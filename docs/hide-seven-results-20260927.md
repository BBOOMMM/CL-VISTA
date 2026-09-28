# HiDe 七任务结果记录

更新时间：2026-09-27

这是一组**跳过 Space** 的 CL-VISTA 七任务实验结果，不是论文表 2 的完整八任务复现。任务顺序为：

```text
Counting → Traffic → Movie → GUI → Science → Sports → STAR
```

## 训练

| 项目 | 作业号 | 配置 | 状态 |
|---|---:|---|---|
| 原版 HiDe | `29435637` | 4×RTX 5090；每卡 batch 2；梯度累积 8；全局 batch 64；1 epoch/任务 | 七阶段完成 |
| 全层单专家路由变体 | 使用原版七阶段检查点，未重新训练 | 推理时每条样本选择一个专家，所有层使用该专家 LoRA | 已评估 |

原版训练输出：

```text
/cache/hpc_user_alden/outputs/hide-videollava-seven/train-20260926T203107-job29435637/
```

## 最终七任务评估

评估的是最终 `06-STAR` 检查点在七个非 Space 测试集上的分数。每个分数由 Qwen3 裁判给出 0～5 分后归一化到 0～100；MFN 是七个分数的算术平均值。

| 任务 | 原版 HiDe | 全层单专家路由变体 |
|---|---:|---:|
| Counting | 50.01 | 38.07 |
| Traffic | 43.36 | 49.49 |
| Movie | 66.86 | 43.19 |
| GUI | 62.66 | 64.10 |
| Science | 69.68 | 71.84 |
| Sports | 72.34 | 59.67 |
| STAR | 62.11 | 62.53 |
| **MFN（七任务平均）** | **61.00** | **55.56** |

### 原版 HiDe

- 预测作业：`29480241`
- 裁判作业：`29491805`
- 结果目录：

  ```text
  /cache/hpc_user_alden/outputs/hide-eval-29435637-final/
  ```

- 结果文件：`scores.json`
- `judge-COMPLETE.json`：`{"pairs": 7}`
- 注意：Slurm 最终状态显示 `CANCELLED`，但七组结果和完成标记均已生成；需要时应保留这一状态说明。

### 全层单专家路由变体

- 预测作业：`29488207`
- 裁判作业：`29491808`
- 结果目录：

  ```text
  /cache/hpc_user_alden/outputs/hide-hard-route-29435637-final/
  ```

- 结果文件：`scores.json`、`final_metrics.json`
- `judge-COMPLETE.json`：`{"pairs": 7}`
- `final_metrics.json` 中的 MFN：`55.55571428571428`（表中四舍五入为 `55.56`）。

## 解释和限制

- 论文表 2 的完整 HiDe 任务序列包含 Space 和 Reasoning；本实验跳过 Space，因此不能直接声称复现论文表 2。
- 论文表 2 的 HiDe 完整八任务结果为 `MFT=66.61`、`MFN=54.44`、`MAA=60.15`、`BWT=-13.91`。本记录只计算七任务最终分数和七任务 MFN，没有计算 MFT、MAA 或 BWT。
- 原版 HiDe 使用当前仓库的推理融合逻辑；变体在不重新训练的情况下，使用锚点 Top-1 选择一个专家，并让所有层只使用该专家 LoRA，不进行专家融合。
- 两列结果的任务集合相同，但变体改变了推理规则；变体结果不是论文原版 HiDe 结果。
