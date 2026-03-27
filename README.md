# Physics-Informed Battery Fault Diagnosis Framework

工业级锂离子电池微短路 (ISC) 与低容量 (Low Cap) 早期诊断深度学习框架。融合热力学机理、去噪自编码器流形学习与 Hybrid CNN-ViT 视觉架构。

## 📂 模块功能与架构

### 1. `config.py` (全局配置与标签中心)
* **功能**: 管理 Ground Truth 电池名单、物理换算常数（如 `UNIT_VOLTAGE=0.001`）及训练参数阈值。
* **参数意义**: `R0_BASE` (机理模型基准内阻), `CAPACITY_AH` (电池额定容量)。

### 2. `models.py` (物理引擎与网络架构)
* **功能**: 统一管理数学映射与深度学习模型结构。
* **核心函数**:
  * `get_physics_benchmark(current, soc, temp)`: Thevenin 电化学模型计算，自带输入越界截断校验 (SOC $\in [0,1]$)。
  * `physics_aware_gaf(series_zscore, clip_sigma=5.0)`: 动态 5-sigma 截断的 GAF 映射，防御强异常被抹平。
  * `generate_single_rgb()`: 多模态融合契约入口，强制校验 Numpy 格式与 Shape 一致性。
  * `AdvancedHybridViT`: 终极判别器。含 3 条 ResNet10t 独立骨干、MHA 通道博弈层及 Transformer 编码器。

### 3. `main.py` (主训练引擎)
* **功能**: 运行数据流转、实体隔离切分、AE 训练及主分类器训练。
* **参数含义**:
  * `--data_path`: 数据集路径（默认: `./data/processed_data.feather`）。
  * `--batch_size`: 批次大小（默认 32，会自适应多 GPU）。
* **容错与内存管理**: 内置防内存溢出熔断机制（Max Slices 限制），使用 `gc.collect()` 进行跨阶段深度清扫。

### 4. `evaluate_and_plot.py` (评估与可视化引擎)
* **功能**: 严密防泄露测试。加载 `training_artifacts.pkl`，仅在真实的 Test 实体集上推断。
* **输出**: 自动生成 `confusion_matrix.png` (混淆矩阵), `roc_curve.png` (ROC曲线), `tsne_features.png` (流形特征分布)。

### 5. `train_pcdit.py` (物理约束扩散生成器)
* **功能**: (预留扩展) 训练 1D 物理条件引导扩散模型，用于生成极其稀缺的微短路数据，对抗数据不平衡。

## 🚀 快速启动
```bash
# 1. 安装依赖
pip install torch pandas numpy timm matplotlib seaborn scikit-learn tqdm

# 2. 运行主训练流水线
python main.py --data_path ./data/processed_data.feather --batch_size 32

# 3. 运行严格评估
python evaluate_and_plot.py --data_path ./data/processed_data.feather
