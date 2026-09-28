# HiDe 锚点统计修复

## 行为

`videollava/model/task_anchors.py` 将任务锚点、累计和及计数器注册为持久 buffer，不再是 Parameter，不进入优化器。统计从零开始，不再用随机锚点和伪计数初始化。

- `image_anchors.*` / `text_anchors.*`：FP32 均值，保留原索引及推理接口。
- `image_anchor_sums.*` / `text_anchor_sums.*`：FP32 累计和。
- `image_boundary.*` / `text_boundary.*`：int64 累计样本数。
- 模型 `.half()`、`.bfloat16()`、`.to(dtype=...)` 只迁移统计 buffer 的设备，不降低统计精度。
- 多模态预处理每次执行一次统计更新，位于 decoder gradient checkpointing 外。各 rank 的本批特征和、整数样本数分别 SUM all_reduce；各 rank 得到相同全局累计值。当前训练流程要求所有 rank 执行相同次数的统计调用；适用于现有数据并行训练，不声明支持混合任务或模型并行分组。

## 保存加载

最终阶段仍写入 `non_lora_trainables.bin`，显式将统计 buffer 加入原有非 LoRA 参数字典。下一任务和推理使用原有 load_state_dict 路径恢复全部统计。

周期检查点另存 `task_anchors.bin`，避免 adapter-only 导出过滤掉 buffer；Trainer 恢复优化器/调度器阶段之后显式恢复该文件。DeepSpeed 完整检查点也包含持久 buffer。

旧格式检查点缺少累计和时，使用保存的均值×计数初始化累计和，并发出警告。这只兼容旧值读取，不修复已经错误的锚点或停在 512 的计数。七任务历史检查点应基于训练数据重新构建统计；本次未改写旧检查点、未提交训练或重评估。

Slurm 的七任务训练、原版评估及变体评估打包入口均显式包含新增模块，支持文件尚未 git add 的工作区。

## 验证

`scripts/hide_5090/tests/test_task_anchors.py` 四项 CPU 测试通过：

1. 优化器不包含统计参数；优化器 step 不改变统计；BF16/FP16 转换保持 FP32 精度；累计计数达到 1200。
2. 统计文件保存加载逐元素一致，恢复后继续累计正确，更新新任务不改变旧任务。
3. 旧格式加载兼容并警告。
4. 两个真实 Gloo 进程、不同本地 batch 大小，验证全局按样本加权均值及计数一致。

另使用真实 HiDe 的微型 Llama 配置验证 buffer 注册、BF16 精度及模型 state_dict 往返；使用真实 LLaVATrainer 构建优化器并验证新增 sidecar 保存恢复（父 Trainer 的完整检查点方法在此检查中 mock）。语法编译及 git diff --check 通过。尚未在 RTX 5090 上进行完整 DeepSpeed 视频训练验证。
