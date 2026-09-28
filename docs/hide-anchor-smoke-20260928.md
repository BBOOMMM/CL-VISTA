# xz313 锚点 smoke test（2026-09-28）

作业 **29502168**，Slurm **COMPLETED / 0:0**，耗时 **00:11:35**。实际 4×RTX 5090、Video-LLaVA-7B、BF16、DeepSpeed ZeRO-2 + CPU optimizer offload，使用修复后的独立代码快照。

每任务每卡 batch 2，梯度累积 1，执行两个分布式 batch / optimizer steps；每任务共 16 条样本，七任务合计 112 条。沿用不含 Space 的七任务序列。样本在各训练集均匀选取，并在分布式启动前检查文件。

## 结果

七阶段全部通过。下表是当前任务对应锚点相对该任务训练前的 L2 变化量；Counting 相对全零初始状态，其余相对上一阶段保存的同编号锚点。

| 任务 | 视觉锚点变化 L2 | 文本锚点变化 L2 | 视觉/文本最终计数 |
|---|---:|---:|---:|
| Counting | 22.233017 | 17.218695 | 16 / 16 |
| Traffic | 25.348988 | 18.500629 | 16 / 16 |
| Movie | 23.741396 | 19.612244 | 16 / 16 |
| GUI | 22.927311 | 20.382536 | 16 / 16 |
| Science | 22.273357 | 17.821129 | 16 / 16 |
| Sports | 25.976437 | 20.918787 | 16 / 16 |
| STAR | 25.270063 | 22.682735 | 16 / 16 |

每个 optimizer step 后执行断言：
- 当前任务计数严格为 8、16；类型 int64。
- 锚点及累计和 FP32，数值有限；锚点等于累计和除以计数。
- 当前任务视觉及文本锚点均不同于训练前；统计不在模型优化器参数列表中。
- 已学及未来任务的锚点、累计和、计数均保持不变。
- 四个 rank 的全部统计逐元素相同。

每阶段保存后重新读取并检查全部 60 个统计张量；下一阶段由生产加载逻辑恢复上一阶段检查点。最终七任务计数均为 16，未使用槽位计数为 0。训练日志中的两步 loss 均有限。

本测试确认统计更新跨真实 optimizer step 和阶段保存加载保留，不能代表完整数据训练后的路由准确率。仅执行两个 batch 的 GPU 测试未直接覆盖大计数；超过 512 的计数已有先前 CPU int64 测试覆盖。

## 产物

节点结果目录：
```text
/cache/hpc_user_alden/outputs/hide-anchor-smoke-job29502168/
```

- `COMPLETE.json`、`summary.json`、`verification.json`：完成标记及汇总。
- 每阶段的 `anchor-audit-rank0.json`～`rank3.json`：两次优化器更新的断言记录。
- 每阶段的 `anchors.pt`、`anchors.json`：已保存统计，便于直接查看。
- `non_lora_trainables.bin`、`adapter_model.bin`、`trainer_state.json`：实际训练检查点。
- 各任务 `.log` 与 `launch.json`：训练日志、启动参数。

本地机器可查看 `scripts/hide_5090/reports/anchor-smoke-29502168.json` 的结果副本。历史正式训练检查点没有改写。

复跑：
```bash
python slurm_scripts/submit_hide_seven.py --anchor-smoke --test-only
python slurm_scripts/submit_hide_seven.py --anchor-smoke
```
