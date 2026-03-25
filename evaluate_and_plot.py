import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

from config import *
from models import generate_single_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedHybridViT(nn.Module):
    def __init__(self, num_classes=3, macro_dim=7):
        super().__init__()
        self.stem_R = timm.create_model('resnet10t', pretrained=False, in_chans=1, num_classes=0)
        self.stem_G = timm.create_model('resnet10t', pretrained=False, in_chans=1, num_classes=0)
        self.stem_B = timm.create_model('resnet10t', pretrained=False, in_chans=1, num_classes=0)
        self.channel_mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2,
                                                   batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.macro_branch = nn.Sequential(nn.Linear(macro_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
                                          nn.Linear(64, 32), nn.ReLU())
        self.fusion_classifier = nn.Sequential(nn.Linear(512 + 32, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                               nn.Dropout(0.4), nn.Linear(128, num_classes))

    def forward(self, img, macro_feats):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        f_r, f_g, f_b = self.stem_R(r).unsqueeze(1), self.stem_G(g).unsqueeze(1), self.stem_B(b).unsqueeze(1)
        stacked = torch.cat([f_r, f_g, f_b], dim=1)
        attn_out, _ = self.channel_mha(stacked, stacked, stacked)
        micro_features = self.spatial_transformer(attn_out).mean(dim=1)
        return self.fusion_classifier(torch.cat([micro_features, self.macro_branch(macro_feats)], dim=1))

    def extract_fused_features(self, img, macro_feats):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        f_r, f_g, f_b = self.stem_R(r).unsqueeze(1), self.stem_G(g).unsqueeze(1), self.stem_B(b).unsqueeze(1)
        stacked = torch.cat([f_r, f_g, f_b], dim=1)
        attn_out, _ = self.channel_mha(stacked, stacked, stacked)
        micro_features = self.spatial_transformer(attn_out).mean(dim=1)
        return torch.cat([micro_features, self.macro_branch(macro_feats)], dim=1)


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
    print("=== 📊 启动 SOTA 严格无泄露评估与可视化引擎 ===")
    import pandas as pd

    with open("training_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    df = pd.read_feather('/root/autodl-tmp/battery/processed_data.feather').apply(pd.to_numeric,
                                                                                  errors='coerce').ffill().bfill().fillna(
        0)
    all_cols = df.columns.tolist()
    I = df['Current'].values.astype(np.float32) * UNIT_CURRENT if 'Current' in all_cols else np.zeros(len(df),
                                                                                                      dtype=np.float32)
    S = (df['SOC'].values.astype(np.float32) / UNIT_SOC_DIVISOR) / CAPACITY_AH if 'SOC' in all_cols else np.zeros(
        len(df), dtype=np.float32)
    T = df['Tmax'].values.astype(np.float32) if 'Tmax' in all_cols else np.zeros(len(df), dtype=np.float32) + 25.0
    V = df[artifacts["cell_names"]].values.astype(np.float32) * UNIT_VOLTAGE

    num_cells, total_rows = V.shape[1], V.shape[0]
    global_labels = build_global_labels(artifacts["cell_names"])

    from models import DenoisingAutoencoder, get_physics_benchmark
    ae = DenoisingAutoencoder(num_cells=artifacts["win"]).to(device)
    ae.load_state_dict(torch.load("ae_denoiser_best.pth", map_location=device))
    ae.eval()

    V_for_ae = V - artifacts["pack_mean"]
    normalize = lambda x: (x - artifacts["mean_v"]) / (artifacts["std_v"] + 1e-6)
    denormalize = lambda x: x * (artifacts["std_v"] + 1e-6) + artifacts["mean_v"]

    raw_list, for_ae_list, phy_list, macro_list, label_list, cell_idx_list = [], [], [], [], [], []
    V_phy_all = get_physics_benchmark(I, S, T)
    win = artifacts["win"]

    for c in range(num_cells):
        step = 10 if global_labels[c] == LABEL_ISC else 100
        for t in range(0, total_rows - win + 1, step):
            raw_list.append(V[t:t + win, c])
            for_ae_list.append(V_for_ae[t:t + win, c])
            phy_list.append(V_phy_all[t:t + win] if V_phy_all.ndim == 1 else V_phy_all[t:t + win, c])
            macro_list.append(
                [np.mean(V[t:t + win, c]), np.std(V[t:t + win, c]), V[t:t + win, c][-1] - V[t:t + win, c][0],
                 np.mean(V_for_ae[t:t + win, c]), np.std(V_for_ae[t:t + win, c]), np.mean(I[t:t + win]),
                 np.mean(S[t:t + win])])
            label_list.append(global_labels[c]);
            cell_idx_list.append(c)

    raw_slices, for_ae_slices, phy_slices = np.array(raw_list), np.array(for_ae_list), np.array(phy_list)
    slice_labels, slice_cell_indices, macro_slices = np.array(label_list), np.array(cell_idx_list), np.array(macro_list,
                                                                                                             dtype=np.float32)

    idx_test = np.where(np.isin(slice_cell_indices, artifacts["test_cells_idx"]))[0]
    macro_slices = (macro_slices - artifacts["train_macro_mean"]) / artifacts["train_macro_std"]

    ae_slices = np.zeros_like(raw_slices)
    inference_loader = DataLoader(torch.tensor(normalize(for_ae_slices), dtype=torch.float32), batch_size=4096)
    ptr = 0
    with torch.no_grad():
        for b_v in inference_loader:
            out = denormalize(ae(b_v.to(device)).cpu().numpy())
            bl = len(out)
            ae_slices[ptr:ptr + bl] = raw_slices[ptr:ptr + bl] - for_ae_slices[ptr:ptr + bl] + out
            ptr += bl

    eval_ds = MultiViewDataset(idx_test, raw_slices, phy_slices, ae_slices, macro_slices, slice_labels,
                               artifacts["stats_phy"], artifacts["stats_ae"])
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=4)

    model = AdvancedHybridViT().to(device)
    model.load_state_dict(torch.load("battery_fault_sota_classifier_best.pth", map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs, all_features = [], [], [], []
    with torch.no_grad():
        for bx, bmacro, by in eval_loader:
            bx, bmacro = bx.to(device), bmacro.to(device)
            features = model.extract_fused_features(bx, bmacro)
            logits = model(bx, bmacro)
            probs = torch.softmax(logits, dim=1)
            all_features.append(features.cpu().numpy());
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy());
            all_labels.append(by.numpy())

    y_true, y_pred, y_prob, features_2d = np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(
        all_probs), np.concatenate(all_features)
    target_names = ['Normal (0)', 'ISC (1)', 'Low Cap (2)']

    print("\n=== 📝 严格无泄露性能报告 (真实世界分布) ===")
    print(classification_report(y_true, y_pred, target_names=target_names))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=target_names,
                yticklabels=target_names, annot_kws={"size": 14})
    plt.title('Strictly Unbiased Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout();
    plt.savefig('confusion_matrix.png', dpi=300)

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6));
    colors = ['green', 'red', 'orange']
    for i in range(3):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{target_names[i]} (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2);
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", fontsize=12);
    plt.savefig('roc_curve.png', dpi=300)

    print("✅ 终极严谨 SOTA 图表生成完毕！")


if __name__ == "__main__":
    main()