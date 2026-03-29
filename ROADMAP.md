# SB-EQ U-Net 三段训练路线图

## 📍 当前状态：Stage-1（已完成代码）

### Stage-1：初步训练生成器
**目标**：
- 双分支编码与等变解码的基本映射
- Barycenter/Residual 初步解耦
- 保持输入结构的细节与边缘一致

**特点**：
- ✅ 不使用判别器
- ✅ 不使用完整barycenter regularization
- ✅ 配对训练：noisy_esc → rss_norm

**数据集**：
- 训练：2420张 noisy_esc → 2420张 rss_norm（配对）
- 验证：1000张 noisy_esc → 1000张 rss_norm（配对）

**启动命令**：
```bash
bash configs/stage1_config.sh
```

**预期输出**：
- `checkpoints_stage1/best_stage1.pth`

---

## 🔜 Stage-2：加入无配对SB + PatchGAN（待实现）

**需要新增**：
1. PatchGAN判别器
2. 对抗损失（Adversarial Loss）
3. 无配对SB损失
4. 非配对数据加载器

**数据集**：
- Set A: 2420张 noisy_esc（非配对）
- Set B: 2420张 rss_norm（非配对）
- 使用 manifest_train.csv 避免样本重叠

**加载Stage-1权重**：
```bash
--load checkpoints_stage1/best_stage1.pth
```

---

## 🎯 Stage-3：完整Barycenter Regularization（待实现）

**需要新增**：
1. 完整barycenter regularization
2. 强化b的退化无关属性
3. 强化r的退化判别性

**加载Stage-2权重**：
```bash
--load checkpoints_stage2/best_stage2.pth
```

---

## ✅ 立即行动清单

### 1. 启动Stage-1训练
```bash
cd /Users/lucas/Projects/UNet_ISB/UNet_ISB_repo
bash configs/stage1_config.sh
```

### 2. 监控训练
```bash
watch -n 1 nvidia-smi
tail -f checkpoints_stage1/train.log
```

### 3. Stage-1完成后
- 检查 `best_stage1.pth` 是否生成
- 在测试集上评估 PSNR/SSIM/LPIPS
- 开始实现Stage-2的判别器
