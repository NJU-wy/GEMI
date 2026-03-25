import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import gc

from config import *
from models import get_physics_benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size, bias=True))
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
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                                 nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_mod1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_mod1, x_mod1, x_mod1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_mod2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod2)
        return x


class PCDiT1D(nn.Module):
    def __init__(self, in_channels=3, seq_len=224, hidden_size=256, depth=4, num_heads=8, num_classes=3, macro_dim=7):
        super().__init__()
        self.in_channels, self.seq_len = in_channels, seq_len
        self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.t_embedder, self.y_embedder = TimestepEmbedder(hidden_size), nn.Embedding(num_classes, hidden_size)
        self.macro_embedder = nn.Sequential(nn.Linear(macro_dim, hidden_size // 2), nn.SiLU(),
                                            nn.Linear(hidden_size // 2, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock1D(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = nn.Sequential(nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
                                         nn.Linear(hidden_size, in_channels))

    def forward(self, x, t, y, macro):
        x = self.x_embedder(x).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y) + self.macro_embedder(macro)
        for block in self.blocks: x = block(x, c)
        return self.final_layer(x).transpose(1, 2)


class SeriesDataset(Dataset):
    def __init__(self, raw, phy, ae, macro, labels):
        self.data, self.macro, self.labels = np.stack([raw, phy, ae], axis=1), macro, labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, i): return torch.tensor(self.data[i], dtype=torch.float32), torch.tensor(self.macro[i],
                                                                                                   dtype=torch.float32), torch.tensor(
        self.labels[i], dtype=torch.long)


def extract_training_data():
    from main import load_data_with_cache
    V, I, S, T, cell_names = load_data_with_cache()
    if V is None: return None

    global_labels = build_global_labels(cell_names)
    V_pack_mean = np.mean(V[:, global_labels == LABEL_NORMAL], axis=1, keepdims=True)
    V_for_ae = V - V_pack_mean
    V_phy_all = get_physics_benchmark(I, S, T)

    raw_list, for_ae_list, phy_list, macro_list, label_list = [], [], [], [], []
    win = min(V.shape[0], 224)
    fault_cells = np.where(global_labels != LABEL_NORMAL)[0]

    for c in fault_cells:
        for t in range(0, V.shape[0] - win + 1, 5):
            raw_list.append(V[t:t + win, c])
            for_ae_list.append(V_for_ae[t:t + win, c])
            phy_list.append(V_phy_all[t:t + win] if V_phy_all.ndim == 1 else V_phy_all[t:t + win, c])
            macro_list.append(
                [np.mean(V[t:t + win, c]), np.std(V[t:t + win, c]), V[t:t + win, c][-1] - V[t:t + win, c][0],
                 np.mean(V_for_ae[t:t + win, c]), np.std(V_for_ae[t:t + win, c]), np.mean(I[t:t + win]),
                 np.mean(S[t:t + win])])
            label_list.append(global_labels[c])

    macro_slices = np.array(macro_list, dtype=np.float32)
    macro_slices = (macro_slices - np.mean(macro_slices, axis=0)) / (np.std(macro_slices, axis=0) + 1e-6)
    return SeriesDataset(np.array(raw_list, dtype=np.float32), np.array(phy_list, dtype=np.float32),
                         np.array(for_ae_list, dtype=np.float32), macro_slices, label_list)


def main():
    dataset = extract_training_data()
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    model = PCDiT1D().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    num_timesteps = 1000
    alpha_bar = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, num_timesteps).to(device), dim=0)

    print("\n[Phase 1] 🚀 启动 PC-DiT 扩散网络训练 (正在学习物理残差规律)...")
    for ep in range(200):
        model.train();
        total_loss = 0
        for x0, macro, y in loader:
            x0, macro, y = x0.to(device), macro.to(device), y.to(device)
            B = x0.shape[0]
            t = torch.randint(0, num_timesteps, (B,), device=device).long()
            noise = torch.randn_like(x0)
            a_bar_t = alpha_bar[t].view(B, 1, 1)
            xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * noise

            optimizer.zero_grad()
            loss = F.mse_loss(model(xt, t, y, macro), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        gc.collect()
        torch.cuda.empty_cache()

        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep + 1:3d}/200 | Denoising Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), "pc_dit_1d_best.pth")


if __name__ == "__main__":
    main()
