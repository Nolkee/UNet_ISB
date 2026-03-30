# SB-EQ U-Net Stage-1 训练指南

## 快速开始

### 1. 环境检查
```bash
python check_env.py
```

这一步会同时检查 `torchmetrics` 和 `lpips`，避免训练结束后才因为缺少 PSNR / SSIM / LPIPS 依赖而失败。

### 2. 验证数据集
```bash
python stages/stage1/verify_dataset.py
```

### 3. 启动训练
```bash
bash stages/stage1/train.sh
```

兼容旧入口：
```bash
bash run_train.sh
# 或
bash configs/stage1_config.sh
```

### 4. 手动运行 Python 入口
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

- `--batch-size`: 根据 GPU 显存调整（8GB → 4, 16GB → 8, 24GB → 16）
- `--amp`: 混合精度训练，节省显存
- `--num-workers`: CPU 核心数，建议 4-8
- `--save-every`: 每 N 个 epoch 保存 checkpoint
- `--val-save-count`: 每个验证 epoch 保存多少组 `input / pred / target` 三联图，设为 `0` 可关闭
- `--val-save-subdir`: 三联图保存到 `save-dir` 下的哪个子目录

## 输出

- `checkpoints_stage1/checkpoint_epoch_XXX.pth`: 定期保存
- `checkpoints_stage1/best_stage1.pth`: 验证集最佳模型
- `checkpoints_stage1/train_args.json`: 训练参数记录
- `checkpoints_stage1/val_triplets/epoch_XXX/`: 每轮固定少量验证三联图
- `checkpoints_stage1/final_eval_metrics.json`: 训练结束后基于最终评估写出的 PSNR / SSIM / LPIPS 结果

最终评估默认会在训练结束后自动运行一次；若存在 `best_stage1.pth`，则优先评估该 checkpoint。

## 验证三联图说明

默认会在每个验证 epoch 保存一个固定小子集，便于横向比较模型输出变化：

- `*_input.png`
- `*_pred.png`
- `*_target.png`

这个子集按验证集顺序确定，不会每轮随机变化；这样更方便比较收敛过程。

## 监控训练

```bash
watch -n 1 nvidia-smi
```

训练日志默认输出到终端；如果需要落盘，请在服务器上自行重定向，例如：

```bash
bash stages/stage1/train.sh | tee checkpoints_stage1/train_stdout.log
```
