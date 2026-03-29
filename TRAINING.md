# SB-EQ U-Net Stage-1 训练指南

## 快速开始

### 1. 环境检查
```bash
python check_env.py
```

### 2. 准备数据集
目录结构：
```
/data/
  train/
    degraded/  # 退化图像
    clean/     # 干净目标
  val/
    degraded/
    clean/
```

### 3. 启动训练
```bash
# 编辑 run_train.sh 修改数据路径
bash run_train.sh
```

或直接运行：
```bash
python train_stage1_restoration.py \
  --train-input-dir /data/train/degraded \
  --train-target-dir /data/train/clean \
  --val-input-dir /data/val/degraded \
  --val-target-dir /data/val/clean \
  --device cuda \
  --epochs 100 \
  --batch-size 8 \
  --amp
```

## 关键参数

- `--batch-size`: 根据GPU显存调整（8GB→4, 16GB→8, 24GB→16）
- `--amp`: 混合精度训练，节省显存
- `--num-workers`: CPU核心数，建议4-8
- `--save-every`: 每N个epoch保存checkpoint

## 输出

- `checkpoints_stage1/checkpoint_epoch_XXX.pth`: 定期保存
- `checkpoints_stage1/best_stage1.pth`: 验证集最佳模型
- `checkpoints_stage1/train_args.json`: 训练参数记录
