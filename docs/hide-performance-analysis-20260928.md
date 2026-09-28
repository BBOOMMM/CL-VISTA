# HiDe 七任务训练耗时分析（2026-09-28）

## 数据来源与边界

分析对象为 xz313 正式训练作业 `29435637`。通过 Slurm sacct、节点上的 run.json、七个 trainer_state.json、训练日志和作业代码快照核查；没有修改训练代码或启动新训练。通过 slogin 建立短时登录读取记录，因当前分区要求最小 GPU 配额，登录申请了 1 张 GPU，读取完成后已退出。

节点输出：`/cache/hpc_user_alden/outputs/hide-videollava-seven/train-20260926T203107-job29435637/`。
历史代码：`/cache/hpc_user_alden/cl-vista-seven/code/job29435637/`。

没有历史 profiler trace、连续 GPU 利用率/显存曲线或数据等待计时，不能把训练循环耗时进一步精确分配给算子、解码、通信和 CPUAdam。下文明确区分实测结果和待验证假设。

## 实测耗时

4×RTX5090，16 CPU，128 GB 内存；BF16、FlashAttention、梯度检查点、ZeRO-2 optimizer CPU offload；每卡 batch 2，累积 8，全局 batch 64；七任务各 1 epoch。总数据 194726 条。

Slurm 总耗时 **42533 秒，即 11:48:53**。七阶段 Trainer train_runtime 合计 **41876.7617 秒，即 11:37:56.8**，占 **98.46%**；其余差额 **656.2383 秒，即 10:56.2**。差额包含启动、加载、保存和切换等，训练循环内部仍包含数据等待及其他开销，并非纯 GPU 时间。

| 任务 | 样本数 | optimizer steps | train_runtime（小时） | 样本/秒（全局） | 秒/optimizer step |
|---|---:|---:|---:|---:|---:|
| Counting | 19700 | 307 | 1.100 | 4.975 | 12.90 |
| Traffic | 30000 | 468 | 1.615 | 5.159 | 12.43 |
| Movie | 30000 | 468 | 2.796 | 2.980 | 21.51 |
| GUI | 30001 | 468 | 1.657 | 5.028 | 12.75 |
| Science | 30000 | 468 | 1.586 | 5.254 | 12.20 |
| Sports | 30000 | 468 | 1.584 | 5.261 | 12.18 |
| STAR | 25025 | 391 | 1.294 | 5.373 | 11.91 |

Movie 占训练循环约 24%，每步比其他 3 万条任务慢约 1.7 倍。若能达到 Traffic 的吞吐，仅此一项可省约 71 分钟；这是条件估算，不是已验证的优化收益。

## 已确认的计算与数据路径

1. **LLM 输入很长，且梯度检查点需要重算。** 视频配置为 8 帧、224 分辨率、patch 14；视频 tower 保留 CLS，默认 patch dropout 为 0，因此每视频输出 8×257=2056 个视觉 token，再加文本。最终多模态长度上限为 3072，不能把命令中的 model_max_length=2048 误解为实际 LLM 总序列上限。冻结 7B 基座权重仍须通过基座反传激活梯度，供 LoRA 和 projector 学习。
2. **训练仅计算当前专家。** `CoIN/peft/tuners/coinmoelora.py` 的训练分支直接索引 cur_task；rank=256 被 8 个专家分割，每专家实际 rank=32。不是每个 token 计算八个专家。该文件与历史代码快照 SHA256 一致。
3. **未使用专家仍被标成可训练。** `mark_only_lora_as_trainable` 仅冻结不含 lora_ 的参数；全部专家及未使用的 lora_router 保留 requires_grad。全层七类投影的八专家 A/B 总计约 6.40 亿参数，当前单专家约 8000 万。Trainer 优化器按 requires_grad 收集，因此存在精简参数组的明确机会；DeepSpeed 实际分配的状态、梯度缓冲及收益应实测，不能直接声称通信量下降八倍。
4. **两个冻结编码器每个 microbatch 仍执行 forward。** 视频 tower 和 CLIP text tower 均 no_grad，但无特征缓存。文本路径还在 GPU forward 内执行 input_ids.cpu().numpy()、LLM tokenizer 解码、CLIP tokenizer 重编码及传回 GPU，形成主路径上的同步和 CPU 工作。
5. **视频在线重复解码。** Decord 在每个样本创建 VideoReader 并抽 8 帧；每 rank 两个 DataLoader worker，合计 8。VideoReader 没有显式限定解码线程数，另有 OMP_NUM_THREADS=4；增加 worker 前应核对线程竞争和 16 CPU 配额。历史快照的处理代码与当前一致。
6. **已启用长度分桶，但长度估计较粗。** 使用对话空格分词数量，没有用实际 tokenizer 长度。FlashAttention 去掉注意力 padding，但 MLP/线性层仍处理补齐后的 batch 张量。分桶优化有空间，视觉 token 固定开销很大，不能预期仅靠分桶获得数倍收益。

### Movie 的特殊性

以下为实际训练标注统计；文本长度是空格分词数，不是 tokenizer token 数。

| 任务 | 平均文本长度 | P95 | 唯一视频路径数 |
|---|---:|---:|---:|
| Counting | 19.4 | 24 | 4377 |
| Traffic | 51.2 | 93 | 13862 |
| Movie | 192.2 | 284 | 2147 |
| GUI | 70.2 | 104 | 8114 |
| Science | 44.8 | 87 | 12873 |
| Sports | 41.0 | 69 | 11886 |
| STAR | 39.2 | 44 | 2148 |

Movie 3 万条问答复用约 2147 个视频，平均约 14 条/视频；STAR 也约 11.6 条/视频。Movie 文本更长，且视频解码可能更慢；没有实际 token 分布和解码计时，不能把它多出的全部时间都归因于字幕。

## 优化优先级

### 第一批：保持模型、数据量、帧数和全局 batch

- **仅把当前专家及原本需要训练的 projector 放入优化器**，冻结非当前专家和未使用 router，同时保持全部专家注册、保存及跨任务加载。验证当前专家确实更新，其他专家逐元素不变。这个改动优先用于减少优化器状态和显存/内存压力，再尝试取消 CPU offload。
- **比较 ZeRO-2 CPU offload 与 GPU optimizer。** 当前优化器在 CPU，可能受到 PCIe 传输、CPUAdam 和线程竞争影响；但 LoRA 训练不能默认认定 offload 是主要瓶颈。先精简参数组，测量显存余量，再测试，无需首先迁移 ZeRO-3。
- **测试每卡 batch 4、累积 4**，保持全局 batch 64。可减少 microbatch 调度、预处理同步和部分通信调用，并改善小 batch 利用率；总样本计算量不会减半。需要 Movie 长样本验证峰值显存；更改 microbatch 会改变随机数和浮点累加顺序，不能保证逐位一致。
- **预处理文本，避免 forward 内 GPU→CPU→GPU 往返。** 先缓存与现有清洗和截断规则一致的 CLIP token IDs，再考虑冻结 CLIP 文本特征。必须复现当前完整 prompt 的解析方式，不能简单用原始问题替换。
- **缓存固定抽取、缩放后的帧，保留在线随机翻转。** 优先 Movie/STAR；可减少重复 VideoReader 打开和解码。按视频、采样索引、预处理版本构建缓存，使用分片而非海量零碎文件；统计构建时间及复用收益。
- **根据 data wait 调整 worker 和解码线程**，例如 worker 2/4、Decord threads 1/2 的小范围组合。不要无依据把 worker 加到 16；维持全局 CPU 预算并记录 CPU 使用率和 GPU 空闲间隙。

### 第二批：更大改动或需要准确性验证

- **冻结视频 tower 特征缓存**：缓存 projector 之前的倒数第二层特征及锚点使用的 pooled 特征，projector 仍在线训练。当前有 RandomHorizontalFlipVideo(p=0.5)，不能每视频只存一份随机结果并称为等价。可核查其他随机性后缓存原始/翻转两种特征，运行时保持相同抽样分布；还需核查 encoder 的 train/eval、dropout 行为。BF16 的 2056×1024 特征约 4 MiB/视频；七任务唯一路径数之和 55407，对应单视图约 217 GiB，双视图约 435 GiB，尚未计文件开销。离线构建成本须计入首轮总耗时，多轮实验更容易获益。
- **选择性减少梯度检查点**：有显存余量后测试部分层关闭，或比较 batch 1、不重算、累积 16 与当前设置。关闭 checkpoint 以显存换计算，可能 OOM，不应直接全关。
- **按真实 token 长度分桶**：尤其核查 Movie 的 padding 比例、截断情况和 rank 间不均衡。改变样本顺序需纳入实验可比性记录。
- **减少视频帧数、池化视觉 token、减少训练样本/LoRA 容量**：可能明显加速，但会改变训练方法或信息量，应作为独立效果对照实验，不能与原设置混报。直接把多模态上限改为 2048 可能截掉答案监督。

FlashAttention、BF16、TF32 已启用；训练没有内置评估，每阶段步数远小于 save_steps=50000。优先改这几项或合并七次启动不会解决小时级耗时。

## 建议的测量方案

选 Movie 和一个代表性普通任务（Traffic 或 Science），固定同一批样本及顺序；每组预热后测 20～50 个 optimizer step，覆盖长样本。基线应使用准备正式采用的代码版本。

记录 data wait、视频 tower、CLIP 文本预处理/forward、LLM forward/backward/checkpoint 重算、optimizer/CPU offload、NCCL 时间，以及 samples/s、真实非 padding tokens/s、峰值显存、CPU/磁盘吞吐。用短 profiler 窗口定位算子，用不带 profiler 的稳态窗口比较总耗时。

按单变量顺序比较：基线 → 精简优化器参数组 → 关闭 optimizer offload → batch 4/accum 4 → 帧缓存/worker 调优。每轮同时检查 loss、参数更新和锚点计数，不仅看吞吐；缓存构建计入首轮成本，另报复用成本。

当前锚点修复版每 microbatch 有四次小 all_reduce；历史 11h48m 版本没有这些调用，不能用它们解释历史慢速。新版本可单独测合并统计通信的收益，保留 FP32 累计和与 int64 计数以及阶段恢复语义。

当前证据足以确定优先排查 Movie 与训练内循环、排除阶段加载保存为主因；不足以承诺“优化后 6 小时”或给 CPU offload/解码标注确切占比。
