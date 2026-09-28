# 本地 Video-LLaVA HiDe

环境：`conda activate clvista-hide`。本目录针对 xz335 的两张 RTX3090；不适用于 RTX5090。
数据位于 `dataset/CL-VISTA/{STAR,Science}`，训练标注和视频已解压，模型在 `models/`。

从项目根目录运行：

```bash
conda activate clvista-hide
bash scripts/local_hide/train.sh STAR
```

默认运行 2 步，rank=256、8 专家，双卡、每卡 batch=1，使用 ZeRO-3 CPU offload。
Science 可替换 STAR，当前两种选择均作为独立首任务（cur_task=0），不是两个任务顺序训练。
正式持续学习需另配置 Taskn 和前一任务 checkpoint。

所有临时文件、扩展编译缓存均在 `.local_runtime/`，不会修改系统 CUDA。
`activate.sh` 配置 CUDA 库搜索路径；`prepare_model.py` 设置本地模型和文本编码器路径。

环境记录：`environment.freeze.txt`（Python 包）和 `conda-explicit.txt`（Conda 包）。
新环境核心为 Python 3.10、Torch 2.0.1+cu118、Transformers 4.31、DeepSpeed 0.9.5、
FlashAttention 2.5.8。CUDA 开发组件装在环境中，CPUAdam 编译缓存位于项目内部。
Decord 0.6.0 的 wheel 元数据会触发 pip 平台标签警告，但此环境已实测可解码视频。

## 已验证结果

2026-09-24 在 xz335 双 RTX3090 上，STAR 首任务成功完成 2 个优化步骤并正常退出。
loss 分别为 0.3607、0.7842，平均 0.5724333。保留 rank=256 和 8 个专家。
训练状态：`outputs/hide-videollava-STAR-verify4/trainer_state.json`；
日志：`.local_runtime/logs/train-verify4.log`。适配器和非 LoRA 参数已保存到同一输出目录。
这证明本地训练链路可运行，不代表完整 CL-VISTA 复现或八任务评估已经完成。
Science 已解压，训练及测试视频路径核对通过，尚未单独训练。

为 ZeRO-3 修复了两个问题：训练前加载冻结文本编码器；按原公式原地更新任务原型，
避免替换 Parameter 导致分片元数据丢失。原始模型结构及专家数量不变。

重新验证可使用新的输出名称：

```bash
conda activate clvista-hide
HIDE_RUN_ID=my-test bash scripts/local_hide/train.sh STAR
```

默认仍为 2 步。长训练需要自行调整 HIDE_MAX_STEPS，并配置任务顺序及后续任务加载。
