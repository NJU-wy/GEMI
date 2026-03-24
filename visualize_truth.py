import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 引入你的配置
from config import LABEL_NORMAL, LABEL_ISC, LABEL_LOW_CAP, get_cell_label
from models import get_physics_benchmark


def main():
    cache_file = 'processed_data.pkl'
    if not os.path.exists(cache_file):
        print(f"❌ 找不到缓存文件 {cache_file}，请先运行一次 main.py！")
        return

    print("🔍 正在加载缓存数据...")
    with open(cache_file, 'rb') as f:
        V, I, S, T, cell_names = pickle.load(f)
    # ==========================================
    # 🌟 关键修改: 剔除 C1 系列的所有坏数据 🌟
    # ==========================================
    valid_indices = [i for i, name in enumerate(cell_names) if not name.startswith('C1-')]
    V = V[:, valid_indices]
    cell_names = [cell_names[i] for i in valid_indices]
    # ==========================================

    total_len = V.shape[0]
    num_cells = V.shape[1]
    print(f"✅ 数据加载完成！总时间步: {total_len}, 电池数量: {num_cells}")

    # 1. 识别电池索引
    labels = np.zeros(num_cells, dtype=int)
    isc_idx, low_cap_idx, normal_idx = [], [], []
    for idx, name in enumerate(cell_names):
        labels[idx] = get_cell_label(name)
        if labels[idx] == LABEL_ISC:
            isc_idx.append(idx)
        elif labels[idx] == LABEL_LOW_CAP:
            low_cap_idx.append(idx)
        else:
            normal_idx.append(idx)

    if not isc_idx:
        print("❌ 没有找到 ISC 故障电池，无法对比！")
        return

    # 2. 空间去共模处理 (只针对纯电压特征，极速计算)
    print("🧠 正在计算空间去共模基准...")
    # 计算所有正常电池的平均电压，作为每一刻的“健康基准线”
    V_pack_mean = np.mean(V[:, normal_idx], axis=1)

    # 计算每个单体偏离健康基准线的残差
    V_residual = V - V_pack_mean[:, None]

    # 3. 计算 Z-Score 阈值 (基于正常电池的正常波动)
    normal_residuals = V_residual[:, normal_idx]
    mu = np.mean(normal_residuals)
    std = np.std(normal_residuals)
    print(f"📊 正常电池波动统计: Mu = {mu:.5f}V, Std = {std:.5f}V")

    # 定义 5 倍标准差为极度异常的红线
    threshold_3sigma = 3 * std
    threshold_5sigma = 5 * std

    # 4. 为了画图不崩溃，进行降采样 (每 100 个点取 1 个)
    # 因为 100 万个点直接画进一张图会变成实心黑块
    downsample_rate = 100
    time_axis = np.arange(0, total_len, downsample_rate)

    # 选取 1 节正常电池和 1 节 ISC 电池进行解剖
    target_normal = normal_idx[0]
    target_isc = isc_idx[0]

    print(
        f"🎨 正在绘制真相揭秘图谱 (目标正常电池: {cell_names[target_normal]}, 目标故障电池: {cell_names[target_isc]})...")

    plt.figure(figsize=(16, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    # === 子图 1: 原始电压宏观对比 ===
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, V[::downsample_rate, target_normal], color='green', alpha=0.7,
             label=f'Normal Cell ({cell_names[target_normal]})')
    plt.plot(time_axis, V[::downsample_rate, target_isc], color='red', alpha=0.7,
             label=f'ISC Cell ({cell_names[target_isc]})')
    plt.title('Subplot 1: Raw Voltage Comparison (Global View)', fontsize=14, fontweight='bold')
    plt.ylabel('Voltage (V)')
    plt.legend(loc='upper right')

    # === 子图 2: 故障电池的“空间去共模异常得分” (Z-Score等效) ===
    plt.subplot(3, 1, 2)
    isc_res_downsampled = V_residual[::downsample_rate, target_isc]

    plt.plot(time_axis, isc_res_downsampled, color='red', linewidth=1)
    plt.axhline(y=threshold_5sigma, color='black', linestyle='--', label='+5 Sigma (Fault Threshold)')
    plt.axhline(y=-threshold_5sigma, color='black', linestyle='--')
    plt.axhline(y=threshold_3sigma, color='orange', linestyle=':', label='+3 Sigma (Warning)')
    plt.axhline(y=-threshold_3sigma, color='orange', linestyle=':')

    # 填充真正属于“故障”的区域
    fault_mask = np.abs(isc_res_downsampled) > threshold_5sigma
    plt.fill_between(time_axis, isc_res_downsampled, 0, where=fault_mask, color='red', alpha=0.3,
                     label='Actual Fault Occurring')

    plt.title(f'Subplot 2: Spatial Residual of ISC Cell ({cell_names[target_isc]}) - Reveal the TRUTH', fontsize=14,
              fontweight='bold')
    plt.ylabel('Deviation from Pack Mean (V)')
    plt.legend(loc='upper right')

    # === 子图 3: 正常电池的“空间去共模异常得分” ===
    plt.subplot(3, 1, 3)
    normal_res_downsampled = V_residual[::downsample_rate, target_normal]
    plt.plot(time_axis, normal_res_downsampled, color='green', linewidth=1)
    plt.axhline(y=threshold_5sigma, color='black', linestyle='--')
    plt.axhline(y=-threshold_5sigma, color='black', linestyle='--')
    plt.title(f'Subplot 3: Spatial Residual of Normal Cell ({cell_names[target_normal]})', fontsize=14,
              fontweight='bold')
    plt.ylabel('Deviation from Pack (V)')
    plt.xlabel('Time Steps (Downsampled)')

    plt.tight_layout()
    plt.savefig('truth_revealed.png', dpi=300)
    print("✅ 分析图已保存为 'truth_revealed.png'！请立刻打开查看！")


if __name__ == "__main__":
    main()
