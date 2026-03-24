import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import timm
import pickle
import gc

torch.set_float32_matmul_precision('high')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from config import *
from models import DenoisingAutoencoder, get_physics_benchmark, generate_single_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


class AdvancedHybridViT(nn.Module):
    def __init__(self, num_classes=3, macro_dim=7):
        super().__init__()
        # ✅ P1修复: 使用 resnet10t + pretrained=False + Dropout 防止严重过拟合
        self.stem_R = timm.create_model('resnet10t', pretrained=False, in_chans=1, num_classes=0)
        self.stem_G = timm.create_model('resnet10t', pretrained=False, in_chans=1, num_classes=0)
        self.stem_B = timm.create_model('resnet10t', pretrained=False, in_chans=1, num_classes=0)

        self.channel_mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2,
                                                   batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.macro_branch = nn.Sequential(
            nn.Linear(macro_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512 + 32, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, img, macro_feats):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        f_r, f_g, f_b = self.stem_R(r).unsqueeze(1), self.stem_G(g).unsqueeze(1), self.stem_B(b).unsqueeze(1)
        stacked = torch.cat([f_r, f_g, f_b], dim=1)
        attn_out, _ = self.channel_mha(stacked, stacked, stacked)
        micro_features = self.spatial_transformer(attn_out).mean(dim=1)
        return self.fusion_classifier(torch.cat([micro_features, self.macro_branch(macro_feats)], dim=1))


def load_data_with_cache(cache_file='/root/autodl-tmp/battery/processed_data.feather'):
    # ✅ P0修复: ffill/bfill 维持平滑，防止传感器丢失导致的 0V 人造短路
    df = pd.read_feather(cache_file).apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0)
    all_cols = df.columns.tolist()

    I = df['Current'].values.astype(np.float32) * UNIT_CURRENT if 'Current' in all_cols else np.zeros(len(df),
                                                                                                      dtype=np.float32)
    S = (df['SOC'].values.astype(np.float32) / UNIT_SOC_DIVISOR) / CAPACITY_AH if 'SOC' in all_cols else np.zeros(
        len(df), dtype=np.float32)
    T = df['Tmax'].values.astype(np.float32) if 'Tmax' in all_cols else np.zeros(len(df), dtype=np.float32) + 25.0

    import re
    cell_names = [name for name in all_cols if re.match(r'^C\d+-\d+$', str(name)) and not str(name).startswith('C1-')]
    V = df[cell_names].values.astype(np.float32) * UNIT_VOLTAGE
    return V, I, S, T, cell_names


class MultiViewDataset(Dataset):
    def __init__(self, indices, raw_slices, phy_slices, ae_slices, macro_slices, label_slices, stats_phy, stats_ae):
        self.indices, self.raw_slices, self.phy_slices, self.ae_slices = indices, raw_slices, phy_slices, ae_slices
        self.macro_slices, self.label_slices = macro_slices, label_slices
        self.stats_phy, self.stats_ae = stats_phy, stats_ae

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = generate_single_rgb(self.raw_slices[idx], self.phy_slices[idx], self.ae_slices[idx], self.stats_phy,
                                  self.stats_ae)
        return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0, torch.tensor(self.macro_slices[idx],
                                                                                    dtype=torch.float32), \
        self.label_slices[idx]


def main():
    seed_everything(42)
    V, I, S, T, cell_names = load_data_with_cache()
    if V is None: return
    num_cells, total_rows = V.shape[1], V.shape[0]

    # 1. 真实物理 Ground Truth 标签
    global_labels = build_global_labels(cell_names)

    # 2. 严格按电池隔离切分 (Train: 70%, Val: 15%, Test: 15%)
    print("\n[Phase 1] 严格实体隔离划分 (Train/Val/Test)...")
    train_cells_idx, val_cells_idx, test_cells_idx = [], [], []
    for lbl in np.unique(global_labels):
        cells = np.where(global_labels == lbl)[0]
        np.random.shuffle(cells)
        p1, p2 = max(1, int(len(cells) * 0.7)), max(1, int(len(cells) * 0.85))
        train_cells_idx.extend(cells[:p1]);
        val_cells_idx.extend(cells[p1:p2]);
        test_cells_idx.extend(cells[p2:])

    # 3. 计算健康基准线 (严密防泄露: 仅取 Train 里的 Normal)
    print("\n[Phase 2] 计算物理与统计基准...")
    train_normal_cells = [c for c in train_cells_idx if global_labels[c] == LABEL_NORMAL]
    V_pack_mean = np.mean(V[:, train_normal_cells], axis=1, keepdims=True)
    V_for_ae = V - V_pack_mean
    MEAN_V, STD_V = np.mean(V_for_ae[:, train_normal_cells]), np.std(V_for_ae[:, train_normal_cells])
    normalize = lambda x: (x - MEAN_V) / (STD_V + 1e-6)
    denormalize = lambda x: x * (STD_V + 1e-6) + MEAN_V

    # 4. 全量切片与 Ground Truth 标签赋予
    raw_list, for_ae_list, phy_list, macro_list, label_list, cell_idx_list = [], [], [], [], [], []
    V_phy_all = get_physics_benchmark(I, S, T)
    win = min(total_rows, 224)

    for c in range(num_cells):
        step = 10 if global_labels[c] == LABEL_ISC else 100
        for t in range(0, total_rows - win + 1, step):
            raw_list.append(V[t:t + win, c])
            for_ae_list.append(V_for_ae[t:t + win, c])
            phy_list.append(V_phy_all[t:t + win] if V_phy_all.ndim == 1 else V_phy_all[t:t + win, c])
            macro_list.append([
                np.mean(V[t:t + win, c]), np.std(V[t:t + win, c]), V[t:t + win, c][-1] - V[t:t + win, c][0],
                np.mean(V_for_ae[t:t + win, c]), np.std(V_for_ae[t:t + win, c]),
                np.mean(I[t:t + win]), np.mean(S[t:t + win])
            ])
            label_list.append(global_labels[c])  # ✅ P1 彻底废除伪标签阈值
            cell_idx_list.append(c)

    raw_slices, for_ae_slices, phy_slices = np.array(raw_list), np.array(for_ae_list), np.array(phy_list)
    slice_labels, slice_cell_indices, macro_slices = np.array(label_list), np.array(cell_idx_list), np.array(macro_list,
                                                                                                             dtype=np.float32)

    idx_train = np.where(np.isin(slice_cell_indices, train_cells_idx))[0]
    idx_val = np.where(np.isin(slice_cell_indices, val_cells_idx))[0]

    # 宏观特征标准化仅依据 Train
    train_macro_mean = np.mean(macro_slices[idx_train], axis=0)
    train_macro_std = np.std(macro_slices[idx_train], axis=0) + 1e-6
    macro_slices = (macro_slices - train_macro_mean) / train_macro_std

    # 5. AE 底座拟合
    print("\n[Phase 3] 训练去噪自编码器 (仅针对 Train Normal 样本)...")
    train_normal_mask = (slice_labels[idx_train] == LABEL_NORMAL)
    ae_loader = DataLoader(
        TensorDataset(torch.tensor(normalize(for_ae_slices[idx_train][train_normal_mask]), dtype=torch.float32)),
        batch_size=2048, shuffle=True)
    ae = DenoisingAutoencoder(num_cells=win).to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(8):
        ae.train()
        for bx, in ae_loader:
            loss = loss_fn(ae(bx.to(device)), bx.to(device))
            opt.zero_grad();
            loss.backward();
            opt.step()

    ae.eval()
    ae_slices = np.zeros_like(raw_slices)
    inference_loader = DataLoader(TensorDataset(torch.tensor(normalize(for_ae_slices), dtype=torch.float32)),
                                  batch_size=4096)
    ptr = 0
    with torch.no_grad():
        for b_v, in inference_loader:
            out = denormalize(ae(b_v.to(device)).cpu().numpy())
            bl = len(out)
            ae_slices[ptr:ptr + bl] = raw_slices[ptr:ptr + bl] - for_ae_slices[ptr:ptr + bl] + out
            ptr += bl

    # Z-Score 统计基准严防泄露
    tune_idx = np.random.choice(idx_train[train_normal_mask], min(10000, sum(train_normal_mask)), replace=False)
    err_phy = raw_slices[tune_idx] - phy_slices[tune_idx]
    stats_phy = (np.mean(err_phy - np.mean(err_phy, axis=1, keepdims=True)),
                 np.std(err_phy - np.mean(err_phy, axis=1, keepdims=True)))
    stats_ae = (np.mean(raw_slices[tune_idx] - ae_slices[tune_idx]), np.std(raw_slices[tune_idx] - ae_slices[tune_idx]))

    # 保存黑盒验证/测试凭证
    artifacts = {
        "pack_mean": V_pack_mean, "mean_v": MEAN_V, "std_v": STD_V,
        "train_macro_mean": train_macro_mean, "train_macro_std": train_macro_std,
        "stats_phy": stats_phy, "stats_ae": stats_ae,
        "train_cells_idx": train_cells_idx, "test_cells_idx": test_cells_idx,
        "cell_names": cell_names, "win": win
    }
    with open("training_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    torch.save(ae.state_dict(), "ae_denoiser_best.pth")

    # ==========================================
    # ✅ OOM 防护盾：彻底销毁 AE 变量，防止内存/显存溢出
    # ==========================================
    print("  -> 🧹 深度清扫阶段：释放 AE 缓存与显存碎片...")
    del ae, ae_loader, inference_loader, train_normal_mask, tune_idx
    gc.collect()
    torch.cuda.empty_cache()
    # ==========================================

    print("\n[Phase 4] 启动 Advanced Hybrid SOTA 分类引擎...")
    # 动态平滑 Focal Loss 权重
    _, counts = np.unique(slice_labels[idx_train], return_counts=True)
    class_weights = 1.0 / np.sqrt(counts)
    class_weights = (class_weights / np.sum(class_weights) * 3.0).astype(np.float32)

    hybrid_batch_size = 32 * max(1, torch.cuda.device_count())
    train_ds = MultiViewDataset(idx_train, raw_slices, phy_slices, ae_slices, macro_slices, slice_labels, stats_phy,
                                stats_ae)
    val_ds = MultiViewDataset(idx_val, raw_slices, phy_slices, ae_slices, macro_slices, slice_labels, stats_phy,
                              stats_ae)

    train_loader = DataLoader(train_ds, batch_size=hybrid_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=hybrid_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = AdvancedHybridViT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = FocalLoss(alpha=torch.tensor(class_weights).to(device), gamma=2.0)
    scaler = torch.amp.GradScaler(device='cuda')
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=100, steps_per_epoch=len(train_loader))

    best_val_acc = 0.0
    for ep in range(100):
        model.train()
        train_loss = 0
        for bx, bmacro, by in train_loader:
            bx, bmacro, by = bx.to(device), bmacro.to(device), by.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(bx, bmacro), by)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防爆
            scaler.step(optimizer)

            old_scale = scaler.get_scale()
            scaler.update()
            if scaler.get_scale() == old_scale: scheduler.step()
            train_loss += loss.item()

        # OOM 防护盾：推断前清空计算图
        torch.cuda.empty_cache()

        model.eval()
        val_corr, total_val = 0, 0
        with torch.no_grad():
            for bx, bmacro, by in val_loader:
                out = model(bx.to(device), bmacro.to(device))
                val_corr += (out.argmax(1) == by.to(device)).sum().item()
                total_val += by.size(0)

        val_acc = val_corr / total_val if total_val > 0 else 0
        if (ep + 1) % 2 == 0 or val_acc > best_val_acc:
            print(f"  Epoch {ep + 1:3d}: Loss {train_loss / len(train_loader):.4f} | Val_Acc {val_acc:.2%}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "battery_fault_sota_classifier_best.pth")


if __name__ == "__main__":
    main()