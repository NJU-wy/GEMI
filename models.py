import torch
import torch.nn as nn
import numpy as np
from config import R0_BASE

def get_physics_benchmark(current, soc, temp):
    ocv = 2.135 + 0.9 * soc - 0.3 * soc ** 2 + 0.6 * soc ** 3
    ocv = np.where(soc < 0.1, ocv - 0.15 * (0.1 - soc), ocv)
    ocv = np.where(soc > 0.95, ocv + 0.08 * (soc - 0.95), ocv)
    r0_adjusted = R0_BASE * np.exp(-0.04 * (temp - 25))
    v_phy = ocv - current * (r0_adjusted * 1.2)
    return v_phy

class DenoisingAutoencoder(nn.Module):
    def __init__(self, num_cells):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_cells, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, num_cells)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

def physics_aware_gaf(series_zscore):
    s_norm = np.clip(series_zscore / 3.0, -1.0, 1.0)
    phi = np.arccos(s_norm)
    gaf_matrix = np.cos(phi[:, None] + phi[None, :])
    return ((gaf_matrix + 1.0) / 2.0 * 255.0).astype(np.uint8)

def generate_single_rgb(v_real, v_phy, v_ae, stats_phy, stats_ae):
    if isinstance(v_real, torch.Tensor): v_real = v_real.numpy()
    if isinstance(v_phy, torch.Tensor): v_phy = v_phy.numpy()
    if isinstance(v_ae, torch.Tensor): v_ae = v_ae.numpy()
    mu_phy, std_phy = stats_phy
    mu_ae, std_ae = stats_ae

    raw_res_phy = v_real - v_phy
    z_phy = ((raw_res_phy - raw_res_phy.mean()) - mu_phy) / (std_phy + 1e-6)
    r_gaf = physics_aware_gaf(z_phy)

    z_ae = ((v_real - v_ae) - mu_ae) / (std_ae + 1e-6)
    g_gaf = physics_aware_gaf(z_ae)

    v_relative = (v_real - v_real[0]) * 100
    b_gaf = physics_aware_gaf(v_relative)
    return np.stack([r_gaf, g_gaf, b_gaf], axis=-1)