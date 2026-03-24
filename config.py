# config.py

# --- 1. 故障标签定义 ---
LABEL_NORMAL = 0
LABEL_ISC = 1
LABEL_LOW_CAP = 2

# --- 2. 故障电池名单 ---
# 请确保这些名字和 Excel 里的列名完全一致 (比如 'C9-17' 还是 'Cell_9-17')
ISC_CELLS = ['C9-17', 'C7-16']
LOW_CAP_CELLS = [
    'C9-15', 'C9-7', 'C8-11', 'C8-5', 'C7-11', 'C7-5',
    'C6-11', 'C6-5', 'C5-11', 'C5-5', 'C4-7', 'C3-15',
    'C3-7', 'C2-7'
]

import numpy as np

ISC_CELLS_SET = set(ISC_CELLS)
LOW_CAP_CELLS_SET = set(LOW_CAP_CELLS)


def is_isc_cell(cell_name: str) -> bool:
    return cell_name in ISC_CELLS_SET


def is_low_cap_cell(cell_name: str) -> bool:
    return cell_name in LOW_CAP_CELLS_SET


def get_cell_label(cell_name: str) -> int:
    if is_isc_cell(cell_name):
        return LABEL_ISC
    if is_low_cap_cell(cell_name):
        return LABEL_LOW_CAP
    return LABEL_NORMAL


def build_global_labels(cell_names):
    return np.array([get_cell_label(name) for name in cell_names], dtype=int)


def validate_label_config():
    overlap = ISC_CELLS_SET.intersection(LOW_CAP_CELLS_SET)
    if overlap:
        raise ValueError(f"Label cell list overlap: {sorted(overlap)}")
    if len(ISC_CELLS) != 2:
        raise ValueError("ISC_CELLS size must be 2")
    if len(LOW_CAP_CELLS) != 14:
        raise ValueError("LOW_CAP_CELLS size must be 14")
    return True

# --- 3. 关键物理参数 ---
# 电池总容量 (你确认是 42Ah)
CAPACITY_AH = 42.0

# --- 4. 单位换算系数 (根据你最新提供的信息) ---
# 电压: 原始值 3000 -> 3.0V => 除以 1000
UNIT_VOLTAGE = 0.001

# 电流: 原始值 8000 -> 8.0A => 除以 1000
UNIT_CURRENT = 0.001

# SOC: 原始值 X。计算逻辑是 X/6 = Ah。
# 所以这里我们只定义原始值的处理系数，后面的除法在 main.py 里做
UNIT_SOC_DIVISOR = 6.0

# --- 5. 物理模型参数 ---
# 42Ah LFP 电池内阻通常在 0.5-2.0 mOhm 之间
R0_BASE = 0.0010
