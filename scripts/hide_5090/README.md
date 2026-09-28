# xz313 / RTX5090 环境

节点环境：`/cache/hpc_user_alden/miniforge3/envs/clvista-hide-5090`。
独立克隆节点原 alden 环境后补包，保留 `torch==2.11.0+cu130`，不覆盖原环境或下载环境。
Counting 离线模型配置、CPU 检查及单任务入口见 [COUNTING.md](COUNTING.md)。
系统 Toolkit 为 CUDA 13.1，PyTorch 自带运行库为 CUDA 13.0。
激活脚本设置 `DS_SKIP_CUDA_CHECK=1`：DeepSpeed 0.19.7 尚未列出 CUDA 13.x
小版本兼容关系，需要跳过其版本字符串检查，并以 CPUAdam 实际编译与运行验证。

登录 xz313（使用已有 Slurm helper）后：

```bash
source /cache/hpc_user_alden/cl-vista-5090-setup/activate.sh
```

代码快照：`/cache/hpc_user_alden/CL-VISTA/Video-LLaVA/HiDe`。
工作缓存、临时文件、验证报告均在 `cl-vista-5090-setup/`；不使用 `/tmp`。
运行 GPU 验证或训练需由 Slurm 分配 GPU；下载作业本身没有 GPU 配额。

配置脚本：`slurm_scripts/setup_hide_5090.sbatch` 和 `finish_hide_5090.sbatch`。
旧版 Video-LLaVA 的 pyproject 精确固定 Torch 2.0.1 等，不可直接 `pip install -e .`
覆盖此 5090 环境。代码通过 PYTHONPATH 导入，环境采用新版 CUDA 兼容训练组件，
Transformers 4.31 / PEFT 0.4 保留旧模型接口。

`patch_pytorchvideo.py` 将旧 PyTorchVideo 对已删除的 torchvision functional_tensor
导入替换为公共 functional API，保留源文件备份。重复安装 PyTorchVideo 后需重新应用。

FlashAttention 使用 `2.8.3.post1`，以系统 CUDA 13.1 从本地源码编译 `sm_120`。
HiDe 的 `llama_flash_attn_monkey_patch.py` 兼容新版 `unpad_input` 多出的返回值。
`finish_hide_5090.sbatch` 优先复用已安装包、缓存 wheel 或中断后保留的源码构建目录，
不会自动从外网下载 FlashAttention；没有缓存时会明确退出。

复查环境可从登录节点的仓库根目录提交（只申请一张 GPU 做算子检查）：

```bash
source ~/slurm_helper.sh
_slurm_cmd sbatch slurm_scripts/validate_hide_5090.sbatch
```

验证脚本检查核心模块、Python 标准库、FlashAttention FP16/BF16 前反向、HiDe
注意力适配器有无 padding 的前反向、bitsandbytes CUDA NF4、DeepSpeed CPUAdam、
PyAV 编码及 Decord 解码。视频样本现场生成，不下载模型或数据。

Decord 0.6.0 的 wheel 残留 `cp36` 元数据，`pip check` 会报告平台标签不匹配；
实际 Python 3.10 导入及视频解码另行验证。bitsandbytes 的可选 CPU 4-bit `kernels`
未安装；GPU NF4 使用自带 CUDA 扩展，不为消除 CPU 功能提示额外下载依赖。

本任务只配置环境和进行导入/算子验证，不启动模型训练，不宣称正式训练流程已兼容。

2026-09-26 最终验证作业 `29420997`：`COMPLETED`，退出码 `0:0`，上述检查全部通过。
报告见 `reports/final-validation.txt`；版本清单见 `reports/pip-freeze.txt` 和
`reports/conda-explicit.txt`。本次继续配置完全复用节点环境和 FlashAttention 源码，
没有外网包下载、模型下载或数据下载。编译的约 60 MB wheel 保存在节点
`/cache/hpc_user_alden/cl-vista-5090-setup/wheels/`，仅适用于此 Python/Torch/CUDA/架构组合。
