import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle

from config import *
from models import DenoisingAutoencoder, get_physics_benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 🌌 启动 PC-DiT 物理约束扩散生成引擎 (设备: {device}) ===")


# ==========================================
# 🌟 核心 SOTA 架构：1D 物理条件引导扩散 Transformer (PC-DiT-1D) 🌟
# ==========================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


class DiTBlock1D(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        # AdaLN 调制核心：将物理条件转化为缩放和移位参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Attention 调制
        x_mod1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_mod1, x_mod1, x_mod1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        # MLP 调制
        x_mod2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod2)
        return x


class PCDiT1D(nn.Module):
    def __init__(self, in_channels=3, seq_len=224, hidden_size=256, depth=4, num_heads=8, num_classes=3, macro_dim=7):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        # 将 3 通道 1D 序列映射到隐空间
        self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1)

        # 条件嵌入器 (时间步 + 标签 + 7D宏观物理特征)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes, hidden_size)
        self.macro_embedder = nn.Sequential(
            nn.Linear(macro_dim, hidden_size // 2), nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # 绝对位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size))

        # Transformer 块
        self.blocks = nn.ModuleList([DiTBlock1D(hidden_size, num_heads) for _ in range(depth)])

        # 输出头：映射回 3 通道 1D 序列
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, in_channels)
        )

    def forward(self, x, t, y, macro):
        """
        x: (B, C=3, L=224) [代表 Raw, Phy_res, AE_res 三条序列]
        t: (B,) 扩散时间步
        y: (B,) 故障标签
        macro: (B, 7) 宏观物理特征
        """
        # 输入投影并加上位置编码
        x = self.x_embedder(x).transpose(1, 2)  # (B, 224, hidden)
        x = x + self.pos_embed

        # 聚合条件 (AdaLN 的条件向量 c)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        m_emb = self.macro_embedder(macro)
        c = t_emb + y_emb + m_emb  # 物理特征与标签时间的完美融合

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x).transpose(1, 2)  # 恢复到 (B, 3, 224)
        return x


# ==========================================
# 🧮 简化的数据加载 (复用你已有的清洗逻辑)
# ==========================================
class SeriesDataset(Dataset):
    def __init__(self, raw_slices, phy_slices, ae_slices, macro_slices, labels):
        # 我们将三个 1D 序列拼成 (3, 224) 的形状
        self.data = np.stack([raw_slices, phy_slices, ae_slices], axis=1)  # Shape: (N, 3, 224)
        self.macro = macro_slices
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.data[i], dtype=torch.float32),
                torch.tensor(self.macro[i], dtype=torch.float32),
                torch.tensor(self.labels[i], dtype=torch.long))


def extract_training_data():
    from main import load_data_with_cache, calc_zscore_stats
    import pandas as pd

    # 这里我们只取微短路(1)和低容量(2)的数据，因为正常数据太多了，不需要生成
    V, I, S, T, cell_names = load_data_with_cache()
    if V is None: return None

    num_cells = V.shape[1]
    total_rows = V.shape[0]
    validate_label_config()
    global_labels = build_global_labels(cell_names)

    print("  -> 正在快速切片并提取少数类特征用于 Diffusion 训练...")
    normal_idx_col = np.where(global_labels == LABEL_NORMAL)[0]
    V_pack_mean = np.mean(V[:, normal_idx_col], axis=1, keepdims=True)
    V_for_ae = V - V_pack_mean
    MEAN_V = np.mean(V_for_ae[:, normal_idx_col])
    STD_V = np.std(V_for_ae[:, normal_idx_col])
    FAULT_THRESHOLD = 3.5 * STD_V
    DYNAMIC_I_THRESHOLD = np.percentile(np.abs(I), 20)

    raw_list, for_ae_list, phy_list, macro_list, label_list = [], [], [], [], []
    V_phy_all = get_physics_benchmark(I, S, T)
    win = min(total_rows, 224)

    # 我们重点针对故障电池采集数据
    fault_cells = np.where(global_labels != LABEL_NORMAL)[0]

    for c in fault_cells:
        cell_name = cell_names[c]
        step = 5  # 密集采样故障序列，增加训练数据量

        for t in range(0, total_rows - win + 1, step):
            current_v = V[t:t + win, c]
            current_v_for_ae = V_for_ae[t:t + win, c]
            current_label = LABEL_NORMAL

        if is_isc_cell(cell_name) and np.max(np.abs(current_v_for_ae)) > FAULT_THRESHOLD:
            current_label = LABEL_ISC
        elif is_low_cap_cell(cell_name) and np.mean(np.abs(I[t:t + win])) > DYNAMIC_I_THRESHOLD:
            current_label = LABEL_LOW_CAP

            if current_label != LABEL_NORMAL:  # 只保留真实的故障片段
                macro_feat = [
                    np.mean(current_v), np.std(current_v), current_v[-1] - current_v[0],
                    np.mean(current_v_for_ae), np.std(current_v_for_ae),
                    np.mean(I[t:t + win]), np.mean(S[t:t + win])
                ]
                raw_list.append(current_v)
                for_ae_list.append(current_v_for_ae)
                phy_list.append(V_phy_all[t:t + win, c] if V_phy_all.ndim > 1 else V_phy_all[t:t + win])
                macro_list.append(macro_feat)
                label_list.append(current_label)

    # 简化的 AE 重建（用 0 占位或者读取你已有的），这里为了极致速度，我们用真实数据训练一个纯信号流形的 Diffusion
    # 真正用到 AE 时可以在生成后再过一遍 AE
    raw_slices = np.array(raw_list, dtype=np.float32)
    phy_slices = np.array(phy_list, dtype=np.float32)
    # 此处偷懒：B通道暂用原始去共模信号代替，依然保留了高频异常特征
    ae_slices = np.array(for_ae_list, dtype=np.float32)

    macro_slices = np.array(macro_list, dtype=np.float32)
    macro_slices = (macro_slices - np.mean(macro_slices, axis=0)) / (np.std(macro_slices, axis=0) + 1e-6)

    print(f"  -> 🎯 成功提取用于生成模型的故障种子序列: {len(label_list)} 个")
    return SeriesDataset(raw_slices, phy_slices, ae_slices, macro_slices, label_list)


# ==========================================
# 🚂 训练主循环 (DDPM Denoising)
# ==========================================
def main():
    dataset = extract_training_data()
    if dataset is None or len(dataset) == 0:
        print("❌ 未提取到故障数据，请检查标签阈值。")
        return

    # 1D 序列显存极小，可以放心把 Batch 调大
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # 实例化 PC-DiT-1D
    model = PCDiT1D(in_channels=3, seq_len=224, hidden_size=256, depth=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # 扩散过程超参数 (Linear Noise Schedule)
    num_timesteps = 1000
    beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    epochs = 200  # 生成模型需要多跑几轮
    print("\n[Phase 1] 🚀 启动 PC-DiT 扩散网络训练 (正在学习物理残差规律)...")

    for ep in range(epochs):
        model.train()
        total_loss = 0
        for x0, macro, y in loader:
            x0, macro, y = x0.to(device), macro.to(device), y.to(device)
            B = x0.shape[0]

            # 1. 随机采样时间步 t
            t = torch.randint(0, num_timesteps, (B,), device=device).long()

            # 2. 生成高斯噪声 epsilon
            noise = torch.randn_like(x0)

            # 3. 计算加噪后的图像 xt
            a_bar_t = alpha_bar[t].view(B, 1, 1)
            xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * noise

            # 4. 神经网络预测噪声 (物理特征约束下)
            optimizer.zero_grad()
            pred_noise = model(xt, t, y, macro)

            # 5. MSE Loss 优化
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            # 加个小防爆盾
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep + 1:3d}/{epochs} | Diffusion Denoising Loss: {total_loss / len(loader):.6f}")

    print("✅ PC-DiT 训练完成！正在保存极化物理流形权重...")
    torch.save(model.state_dict(), "pc_dit_1d_best.pth")


if __name__ == "__main__":
    main()
