# xz313 HiDe 七任务训练（不含 Space）

这是基于原项目 `Train/train_CVU.sh`、`Task1.sh` 和 `Taskn.sh` 的硬件与路径适配。
不是论文表 2 的完整八任务复现；跳过 Space 会改变后续检查点及遗忘指标。

## 从本地仓库提交

```bash
cd /home/alden/CL-VISTA
# 仅检查调度，不启动训练
python slurm_scripts/submit_hide_seven.py --test-only
# 正式训练，七任务顺序执行，每任务 1 epoch，不自动评估
python slurm_scripts/submit_hide_seven.py
# 可选：七任务各 128 条样本、2 个 optimizer step 的串行冒烟验证
python slurm_scripts/submit_hide_seven.py --mode smoke
```

使用本地 Python 标准库、git、sudo 和 Slurm；不要求本地激活训练环境。
提交身份为 hpc_user_alden，节点 xz313，分区 yzgpu，4 张 RTX5090，16 CPU，128 GB 内存。
正式作业时限 24 小时，冒烟 1 小时。命令输出 Slurm job id。

每次提交打包当前工作树中的已跟踪 HiDe 源文件（包括未提交的修改），以及明确列出的运行入口和配置，
通过 sbatch 脚本传给节点。模型、数据、环境、日志和报告均不包含在快照里。
快照恢复到 `/cache/hpc_user_alden/cl-vista-seven/code/job<JOBID>/`，校验 SHA256 后运行。
本地提交记录位于被 gitignore 忽略的 `.local_runtime/seven-submit/`。
训练使用节点已有独立环境和模型；不下载任何外部文件。

## 配置和阶段衔接

- 顺序：Counting → Traffic → Movie → GUI → Science → Sports → STAR。
- cur_task 连续取 0～6，保留 8 专家（专家 7 未用于训练）。后续评估必须使用同一任务映射。
- 每卡 batch 2，梯度累积 8，4 卡，全局 batch 64。
- rank 256、alpha 512、主学习率 1e-4、投影层 1e-5、1 epoch、warmup 0.03、cosine。
- BF16、TF32、梯度检查点、ZeRO-2 optimizer CPU offload；复用已验证的视频配置（8 帧）。
- 各任务使用 train_path，不使用 replay/router 标注，与原 HiDe 训练脚本一致。
- 每阶段重新加载同一基座；阶段 1～6 通过原代码的 previous_task_model_path 加载上阶段
  adapter_model.bin 和 non_lora_trainables.bin，不合并适配器到基座。
- 正式输出根目录：`/cache/hpc_user_alden/outputs/hide-videollava-seven/`。
  每次运行创建独立目录，内含 run.json、各阶段 launch.json、训练日志及完成标记。
- 任一阶段失败立即终止，不跳过、不自动继续评估。
- 入口目前不支持中断后自动续跑；重新提交会新建一次完整运行，不能当作断点续训命令。

总控日志：`/cache/hpc_user_alden/alden_train-<JOBID>.log` 和 `.err`；
各阶段完整日志位于输出目录 `00-Counting.log` 等文件。

## 范围与限制

此入口仅训练，不调用 Qwen3、不生成测试预测、不计算论文指标。
四卡每卡 batch 2 已经通过 Counting 吞吐量测试；其他任务长文本可能有更高显存需求。
旧 Transformers / 新 DeepSpeed 组合仍可能产生学习率调度调用顺序警告；成功的少量参数更新
并不能单独证明完整训练轨迹与论文原环境一致。

## 已完成的串行验证

2026-09-26：作业 29434469，xz313，4 张 RTX5090，七任务各 128 条样本，
每卡 batch 2，累积 8，每任务 2 个 optimizer step。七阶段均成功保存检查点，
后六阶段的四个 rank 均记录成功加载上一阶段 LoRA；未出现数据读取异常或 CUDA OOM。
Slurm COMPLETED / 0:0，总耗时 13 分 4 秒。

验证输出：
`/cache/hpc_user_alden/outputs/hide-videollava-seven/smoke-20260926T200228-job29434469`。
仅验证启动、少量训练和连续检查点衔接；不代表完整 epoch、评估或表 2 数值已经复现。

正式训练作业名为 `alden_train`，通知参数为 `--mail-type=BEGIN,END,FAIL --mail-user=FT_USER_ALDEN`。
