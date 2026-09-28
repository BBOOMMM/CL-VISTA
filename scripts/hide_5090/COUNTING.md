# xz313：离线 Video-LLaVA HiDe / Counting 首任务

配置目录：`/cache/hpc_user_alden/CL-VISTA/configs/xz313_hide_counting/`。
仅使用 Counting，`cur_task=0`；不使用 Space、replay 或 router 混合标注。

## 模型与数据

- 配置后的模型入口：`/cache/hpc_user_alden/models/Video-LLaVA-7B-HiDe-video-only`。
- 视频编码器：`/cache/hpc_user_alden/models/LanguageBind_Video_merge`。
- 文本编码器：`/cache/hpc_user_alden/models/clip-vit-large-patch14-336`。
- 训练标注：`/cache/hpc_user_alden/datasets/CL-VISTA/Counting/json_row/train_data_20k.json`。
- 视频根目录：`/cache/hpc_user_alden/datasets/CL-VISTA`，与标注中的 `Counting/...` 相拼接。

`prepare_counting.py` 创建独立模型配置，将 `mm_image_tower` 设为 `null`，视频与文本
编码器改为节点本地绝对路径。纯视频训练不会初始化未使用的图像分支，因此不需要
另下载 LanguageBind_Image。权重与 tokenizer 文件通过符号链接复用；原始模型目录、
HDFS 归档与 HiDe 核心代码均不因这一步改变。

## CPU 检查与命令预览（不会训练）

通过 Slurm helper 登录 xz313 后：

```bash
source /cache/hpc_user_alden/CL-VISTA/scripts/hide_5090/counting_env.sh
python "$HIDE_PROJECT_ROOT/scripts/hide_5090/run_counting.py" --mode smoke
python "$HIDE_PROJECT_ROOT/scripts/hide_5090/run_counting.py" --mode train
CUDA_VISIBLE_DEVICES='' python "$HIDE_PROJECT_ROOT/scripts/hide_5090/check_counting.py"
```

默认只打印启动命令，不执行。检查会核对全部 Counting 训练与测试标注引用，加载本地
tokenizer，解析两种启动参数，构造 CPU Trainer，并对首、中、末三个训练样本进行
真实视频解码、分词和组批，确认存在有效监督 token。进程禁用 GPU 并阻止网络连接。

完整 CPU 权重加载检查额外加载主模型、视频塔和冻结文本塔，建议用提供的 64 GB
内存、4 CPU 作业；该作业不申请 GPU，不进行模型前反向或优化：

```bash
sbatch /cache/hpc_user_alden/CL-VISTA/slurm_scripts/check_counting_cpu.sbatch
```

报告位于 `/cache/hpc_user_alden/cl-vista-counting/reports/`。

2026-09-26 CPU 检查作业 `29425141` 已完成，退出码 `0:0`：主模型、视频塔、文本塔
全部从节点本地权重加载成功，图像塔未初始化，网络连接尝试为 0；Counting 数据检查、
样本预处理、两种启动参数解析及 CPU Trainer 构造均通过。报告副本见
[counting-offline-cpu-load.json](reports/counting-offline-cpu-load.json)。
最终轻量检查还覆盖测试问题的 `video_name` 加后缀解析，以及 2,000 条测试问题与
答案的唯一 ID 一一对应，见 [counting-preflight.json](reports/counting-preflight.json)。
环境版本未调整，本次未修改 HiDe 核心代码、运行 GPU 前反向或训练。

## GPU 验证与训练入口

在 xz313 登录任务中，准备验证时可执行：

```bash
# 申请 2 张 RTX5090，进行 2 个优化步骤
sbatch /cache/hpc_user_alden/CL-VISTA/slurm_scripts/counting_5090.sbatch smoke

# GPU 验证通过后，再执行 Counting 一轮训练
sbatch /cache/hpc_user_alden/CL-VISTA/slurm_scripts/counting_5090.sbatch train
```

作业限定 xz313、单节点、2 GPU、8 CPU、96 GB 内存；先进行 CPU 数据检查，再启动
单个 torchrun，进程数必须等于配置 GPU 数和 Slurm 可见 GPU 数。不使用 ssh，不下载
文件，不自动提交后续任务。输出使用时间戳及 job ID 新建目录，拒绝覆盖同名运行。

| 参数 | 两步验证 smoke | Counting 一轮 train |
| --- | --- | --- |
| LoRA rank / alpha | 256 / 512 | 256 / 512 |
| 专家数 / 当前任务 | 8 / 0 | 8 / 0 |
| GPU 数 / 每卡 batch | 2 / 1 | 2 / 1 |
| 梯度累积 / 有效 batch | 1 / 2 | 32 / 64 |
| 停止条件 | 2 步 | 1 epoch |
| 学习率 / projector 学习率 | 1e-4 / 1e-5 | 1e-4 / 1e-5 |
| 帧数 / 精度 | 8 / BF16 | 8 / BF16 |
| 分布式策略 | ZeRO-2 + CPU optimizer offload | 同左 |

正式模式的有效 batch=64 与论文 §4.1 一致；LoRA 设置依据表 B.1。
仓库原始 task1.json 的 `2×4×2=16` 与论文 batch 设置不同，本入口明确使用 `2×1×32=64`。
5090 与论文 8×A100 的硬件不同，显存是否足够以及实际训练稳定性须由 GPU 验证确认；
CPU 加载成功不代表训练已跑通，Counting 单任务也不等于表 2 的八任务复现。

## 已完成的两卡验证

2026-09-26 在 xz313 的两张 RTX5090 上，作业 `29425497` 完成了 2 个训练步骤，
状态 `COMPLETED`、退出码 `0:0`。使用上述 smoke 参数和原有 ZeRO-2 配置；
本次验证无需修改环境、HiDe 核心代码或启动参数。

两步 loss 为 `0.9806`、`0.9237`，平均 `0.9521908760`。
输出位于节点：

```text
/cache/hpc_user_alden/outputs/hide-videollava-counting/smoke-20260926T143529-job29425497
```

`trainer_state.json` 中 `global_step=2`。保存的 `adapter_model.bin`（约 1.30 GB）
和 `non_lora_trainables.bin`（约 42 MB）已在 CPU 上重新读取，所有张量均为有限值；
224 个当前任务 LoRA B 张量由零初始化变为非零，确认参数实际发生更新。
报告见 [counting-gpu-smoke-29425497.json](reports/counting-gpu-smoke-29425497.json)。

日志保留 PyTorch checkpoint 的 `use_reentrant` 提示和 scheduler/optimizer 顺序提示；
运行未因此失败，实际权重更新已核对。这次验证使用有效 batch=2，尚未运行有效 batch=64
的一轮训练，也没有测试断点续训或评估准确率。
