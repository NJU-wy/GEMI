import numpy as np

# --- 1. 故障标签定义 ---
LABEL_NORMAL = 0
LABEL_ISC = 1       # 微短路
LABEL_LOW_CAP = 2   # 低容量

# --- 2. 真实故障电池 Ground Truth名单 ---
ISC_CELLS = ['C9-17', 'C7-16']
LOW_CAP_CELLS = [
    'C9-15', 'C9-7', 'C8-11', 'C8-5', 'C7-11', 'C7-5',
    'C6-11', 'C6-5', 'C5-11', 'C5-5', 'C4-7', 'C3-15',
    'C3-7', 'C2-7'
]

ISC_CELLS_SET = set(ISC_CELLS)
LOW_CAP_CELLS_SET = set(LOW_CAP_CELLS)

def get_cell_label(cell_name: str) -> int:
    if cell_name in ISC_CELLS_SET: return LABEL_ISC
    if cell_name in LOW_CAP_CELLS_SET: return LABEL_LOW_CAP
    return LABEL_NORMAL

def build_global_labels(cell_names):
    return np.array([get_cell_label(name) for name in cell_names], dtype=int)

# --- 3. 关键物理参数与单位换算系数 ---
CAPACITY_AH = 42.0
UNIT_VOLTAGE = 0.001       # 3000 -> 3.0V
UNIT_CURRENT = 0.001       # 8000 -> 8.0A
UNIT_SOC_DIVISOR = 6.0     # SOC 转换逻辑
R0_BASE = 0.0010           # Thevenin 内阻基准