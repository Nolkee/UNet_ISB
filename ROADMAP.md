# SB-EQ U-Net 三段训练路线图

## 当前状态：Stage-1（已完成代码）

### Stage-1：初步训练生成器
**目标**：
- 双分支编码与等变解码的基本映射
- Barycenter/Residual 初步解耦
- 保持输入结构的细节与边缘一致

**特点**：
- ✅ 不使用判别器
- ✅ 不使用完整 barycenter regularization
- ✅ 配对训练：noisy_esc → rss_norm
- ✅ 验证阶段自动保存少量固定样本的 `input / output / GT` 三联图
- ✅ 训练结束后自动对验证/测试集输出 PSNR / SSIM / LPIPS

**数据集**：
- 训练：2420 张 noisy_esc → 2420 张 rss_norm（配对）
- 验证：1000 张 noisy_esc → 1000 张 rss_norm（配对）

**启动命令**：
```bash
bash stages/stage1/train.sh
```

**预期输出**：
- `checkpoints_stage1/best_stage1.pth`
- `checkpoints_stage1/final_eval_metrics.json`
- `checkpoints_stage1/val_triplets/epoch_XXX/*.png`

---

## Stage-2：加入无配对 SB + PatchGAN（待实现）

**需要新增**：
1. PatchGAN 判别器
2. 对抗损失（Adversarial Loss）
3. 无配对 SB 损失
4. 非配对数据加载器

**数据集**：
- Set A: 2420 张 noisy_esc（非配对）
- Set B: 2420 张 rss_norm（非配对）
- 使用 `manifest_train.csv` 避免样本重叠

**占位脚本**：
```bash
bash stages/stage2/train.sh
```

**加载 Stage-1 权重**：
```bash
--load checkpoints_stage1/best_stage1.pth
```

---

## Stage-3：完整 Barycenter Regularization（待实现）

**需要新增**：
1. 完整 barycenter regularization
2. 强化 b 的退化无关属性
3. 强化 r 的退化判别性

**占位脚本**：
```bash
bash stages/stage3/train.sh
```

**加载 Stage-2 权重**：
```bash
--load checkpoints_stage2/best_stage2.pth
```

---

## 立即行动清单

### 1. 验证 Stage-1 数据集
```bash
cd /Users/lucas/Projects/UNet_ISB/UNet_ISB_repo
python stages/stage1/verify_dataset.py
```

### 2. 启动 Stage-1 训练
```bash
cd /Users/lucas/Projects/UNet_ISB/UNet_ISB_repo
bash stages/stage1/train.sh
```

### 3. 监控训练
```bash
watch -n 1 nvidia-smi
bash stages/stage1/train.sh | tee checkpoints_stage1/train_stdout.log
```

### 4. Stage-1 完成后
- 检查 `best_stage1.pth` 是否生成
- 检查 `final_eval_metrics.json` 是否生成，并确认其中的 `checkpoint / psnr / ssim / lpips` 都是有效值
- 检查 `val_triplets/` 中的输入 / 输出 / GT 三联图是否按 epoch 保存
- 开始实现 Stage-2 的判别器
