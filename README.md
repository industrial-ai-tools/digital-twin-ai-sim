# digital-twin-ai-sim

《数字孪生 × AI 仿真内核》CSDN专栏配套代码

## 目录

- `data.py` — Darcy Flow数据生成（高斯随机场+有限差分求解器）
- `model.py` — FNO2d模型（SpectralConv2d + FNOBlock）
- `train.py` — 训练主流程（含验证集监控）
- `inference.py` — 在线推理引擎（毫秒级实时预测）

## 快速开始

```bash
pip install -r requirements.txt

# 生成数据
python -c "from data import generate_dataset; generate_dataset(n_samples=200)"

# 训练
python train.py

# 推理
python inference.py
