# UNet_ISB

`UNet_ISB` 是一个面向科研复现与增量改造的代码仓库，用于实现参考文档中的 SB-EQ 生成器三段训练策略。

当前仓库以 [`milesial/pytorch-unet`](https://github.com/milesial/pytorch-unet) 为 U-Net 基线，在其上加入第一阶段所需的双分支编码、`b/r` 解耦、动态掩码和 EQ 解码逻辑。仓库同时保留了原始分割版 `UNet`，并新增一条独立的 restoration 训练路径。

## 当前状态

- 已完成：第一阶段生成器代码骨架与训练入口
- 未完成：第二阶段无配对 SB + PatchGAN
- 未完成：第三阶段完整 barycenter regularization
- 重要边界：当前实现是“基于 U-Net 的第一阶段生成器改造”，不是完整的 SB/I2SB 训练框架

这点必须说清楚。参考 PPT 中“骨干网络：SB”这一条，在当前仓库里还没有被完整复现。现阶段实现的是：

- `x(t_i)` 与 `t_i` 条件下的时间条件生成器
- 双分支编码与等变解码
- `b/r` 初步解耦
- 第一阶段细节保持损失

如果你的目标是“完全等价于文档中的 SB 主干训练流程”，那么第二轮代码工作必须继续把 SB 轨迹采样、桥过程训练和后续两阶段补齐。

## 第一阶段审计结论

### 已对齐的点

根据内部参考资料《三段训练策略.docx》与《SB-EQ整体架构.pptx》，当前第一阶段已经实现并且代码上可对应的点如下：

1. 双分支编码与等变解码的基本映射  
   实现位置：
   [sb_eq_unet_model.py](unet/sb_eq_unet_model.py)
   [sb_eq_parts.py](unet/sb_eq_parts.py)

2. `b/r` 初步解耦  
   实现位置：
   [sb_eq_unet_model.py](unet/sb_eq_unet_model.py)
   [sb_eq_parts.py](unet/sb_eq_parts.py)

3. 在输入结构上保持细节与边缘一致  
   实现位置：
   [restoration_losses.py](utils/restoration_losses.py)
   包含：
   - Charbonnier 重建项
   - 高频梯度保持项
   - PatchNCE 结构保持项

4. 第一阶段不启用判别器与完整 barycenter regularization  
   当前仓库中没有接入 PatchGAN，也没有实现 `L_adv` 和 `L_WB`，这与第一阶段要求一致。

5. 动态掩码三头输出  
   当前实现已包含：
   - `mask_eq`
   - `mask_res`
   - `mask_reg`

6. 调制特征与 skip 融合  
   当前实现已包含：
   - `z' = b + alpha * (r * mask_res)`
   - `skip = mask_eq * fe + (1 - mask_eq) * fc`

### 我修复过的正确性问题

这次审计和修补里，已经补掉以下会影响第一阶段结论可信度的实现问题：

- `manifest` 相对路径解析  
  现在优先按 `manifest` 所在目录解析，避免训练清单在服务器上错指路径。  
  位置：
  [restoration_data_loading.py](utils/restoration_data_loading.py)

- `IRC` 在没有标签时静默失效  
  现在如果 `--irc-weight > 0` 但训练数据没有 `degradation_label`，脚本会直接报错，而不是假装在做残差对比学习。  
  位置：
  [train_stage1_restoration.py](train_stage1_restoration.py)

- checkpointing 接口原本是坏的  
  现在已改成真正的 gradient checkpoint wrapper。  
  位置：
  [checkpointing.py](unet/checkpointing.py)
  [unet_model.py](unet/unet_model.py)
  [sb_eq_unet_model.py](unet/sb_eq_unet_model.py)

- `IRC` projector 依赖懒初始化  
  已改成显式特征维度初始化，避免 checkpoint 恢复时出现不透明状态。  
  位置：
  [restoration_losses.py](utils/restoration_losses.py)

- `--load` 只加载模型不恢复训练上下文  
  现在会恢复 `optimizer`、`criterion`、`epoch` 和已有最优验证值。  
  位置：
  [train_stage1_restoration.py](train_stage1_restoration.py)

### 仍然不能虚假宣称“完美无误”的点

下面这些边界我必须明确写出来：

1. 当前不是完整 SB backbone  
   这是最重要的事实。文档说“骨干网络：SB”，而当前仓库仍然是“以 U-Net 为骨架的第一阶段生成器实现”。如果你要求的是完整 SB 训练闭环，这一项还没完成。

2. 当前 EQ 分支是离散 `C4` 权重共享近似  
   这是为了避免额外 steerable/e2cnn 依赖，先把第一阶段逻辑落成可训练代码。它是工程近似，不应写成“与参考论文完全等价的旋转等变实现”。

3. 我没有在本机做训练实跑  
   按你的要求，没有在本机启动训练。代码已经做过静态审计和逻辑修补，但最终正确性仍然需要你在远程 Ubuntu 服务器上用真实数据跑通一次前向和训练。

## 仓库结构

- [train_stage1_restoration.py](train_stage1_restoration.py)
  第一阶段训练入口
- [unet/sb_eq_unet_model.py](unet/sb_eq_unet_model.py)
  第一阶段生成器
- [unet/sb_eq_parts.py](unet/sb_eq_parts.py)
  双分支编码、动态掩码、EQ 上采样等模块
- [utils/restoration_data_loading.py](utils/restoration_data_loading.py)
  配对 restoration 数据集
- [utils/restoration_losses.py](utils/restoration_losses.py)
  第一阶段损失
- [README_stage1_sb_eq.md](README_stage1_sb_eq.md)
  第一阶段简版说明

## 数据格式

最简单的数据组织方式：

```text
train_inputs/
  sample_0001.png
train_targets/
  sample_0001.png
val_inputs/
val_targets/
```

如果你有时间步和退化标签，建议使用 `manifest`：

```text
input_path,target_path,timestep,degradation_type,degradation_level,degradation_label
train_inputs/sample_0001.png,train_targets/sample_0001.png,0.45,noise,25,0
```

注意：

- 如果不提供 `manifest`，所有样本会使用同一个 `--default-timestep`
- 如果 `--irc-weight > 0`，训练集必须提供有效的 `degradation_label`

## 远程 Ubuntu 训练

下面的命令应该在你的远程服务器上运行，而不是本机：

```bash
cd /path/to/UNet_ISB_repo
python train_stage1_restoration.py \
  --train-input-dir /path/to/train/inputs \
  --train-target-dir /path/to/train/targets \
  --val-input-dir /path/to/val/inputs \
  --val-target-dir /path/to/val/targets \
  --save-dir /path/to/checkpoints_stage1 \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --amp
```

如果你有清单文件：

```bash
python train_stage1_restoration.py \
  --train-input-dir /path/to/train/inputs \
  --train-target-dir /path/to/train/targets \
  --train-manifest /path/to/train_manifest.csv \
  --val-input-dir /path/to/val/inputs \
  --val-target-dir /path/to/val/targets \
  --val-manifest /path/to/val_manifest.csv \
  --save-dir /path/to/checkpoints_stage1 \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --amp
```

## 下一步

如果你的目标是继续向参考方案收敛，下一步应该做的是：

1. 在当前第一阶段代码上接入真正的 SB 轨迹/时间步采样流程
2. 实现第二阶段的无配对 SB + PatchGAN
3. 实现第三阶段的完整 barycenter regularization

## 致谢

- U-Net 基线来自 [`milesial/pytorch-unet`](https://github.com/milesial/pytorch-unet)
- 本仓库当前第一阶段实现是在该基线上做研究改造，不代表上游仓库本身提供了 SB-EQ 训练逻辑
