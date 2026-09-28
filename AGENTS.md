# 项目协作约定

## 重要红线
1. 不要下载任何数据到 /tmp 目录下，临时测试文件就下载到本目录下，挂载的文件系统空间足够

## 项目结构与脚本职责

- `README.md`：数据准备、模型下载、环境安装及实验运行的总体说明。
- `LLaVA/`、`InternVL/`、`Video-LLaVA/`、`VideoLLaMA2/`：按模型划分，各自包含不同方法的实现和训练、评估脚本。运行前先阅读所选方法的说明和脚本，确认工作目录及相对路径。
- `configs/data_configs/`、`configs/model_configs/`、`configs/train_configs/`：分别保存数据、模型和训练配置。实际参数还可能在方法目录的 shell 脚本中设置；修改实验时同时核对配置与启动参数。
- 各方法目录的 `scripts/MCITlib/Train/`：实验训练入口，具体文件和分布式启动方式以所选方法为准。
- `slurm_scripts/submit_download.sh`：从登录节点的仓库根目录提交数据下载任务，选择节点、分区并传递下载参数。
- `slurm_scripts/download_cl_vista.sbatch`：计算节点内的数据下载任务，负责独立下载环境、缓存锁、ModelScope 下载和重试，不启动训练。
- `slurm_scripts/README.md`：当前下载任务的资源、路径、续传和解压操作说明。

本项目目前没有统一的 `submit_slurm.sh`、`run.sbatch` 或环境上传脚本。不要直接套用
其他项目的 `RUN_CONFIGS`、`RUN_MODE`、YAML 路径、模型导出上传和通知链路，也不要假定
本项目已实现代码归档发布、离线训练环境自动恢复或训练前自动预下载。

## Slurm 节点登录与资源

不要用普通 `ssh` 直接登录计算节点。先加载用户目录下的 Slurm helper，再用
`slogin` 以 `hpc_user_alden` 身份创建登录任务或附着已有任务：

```bash
source ~/slurm_helper.sh
slogin xz293 yzgpu
slogin xz312 ds02
slogin -m 16 xz313 yzgpu
```

第一个参数是节点，第二个参数是分区。已知 `xz312` 属于 `ds02`，`xz293`、`xz313`
属于 `yzgpu`；本项目下载脚本也支持 `yzgpu` 的 `xz288`。节点分区、预约、可用资源
可能变化，提交前核对当前调度信息和脚本支持范围。

helper 通过 Slurm 分配交互 PTY，并切换到 `/cache/hpc_user_alden/`。不要假定登录
后仍在仓库根目录。打包并向 HDFS 上传完整环境时，至少申请 16 GB 登录内存；默认
1 GB 可能使 `tar/gzip` 和 HDFS JVM 被 Slurm OOM kill。

## 节点目录与数据准备

- 将工作文件、日志、环境、缓存及临时文件放在自己的 `/cache/hpc_user_alden/` 下。
  不要覆盖其他任务正在使用的环境，也不要通过修改 `HOME` 改变缓存位置。
- `/cache` 按节点本地存储处理，不假定节点之间共享，也不假定登录节点仓库在计算
  节点可见。训练应安排到数据所在节点，并确认代码、模型和配置已在该节点可访问。
- 当前默认数据目录为 `/cache/hpc_user_alden/datasets/CL-VISTA`，下载环境为
  `/cache/hpc_user_alden/envs/cl-vista-download`。提交前节点上的用户根目录必须存在
  且可写，以便 Slurm 进入工作目录并打开日志。
- 下载失败后向同一节点、同一目录重新提交，保留已有文件和 ModelScope 缓存以便
  续传。共享数据目录和环境的准备过程应使用锁，避免并发写入。
- 当前下载任务仅下载归档，解压和完整性检查需要另行完成；Space 视频还需按
  `README.md` 申请 ScanNet 数据权限并转换。启动训练前确认所需标注、媒体文件、
  模型权重和 tokenizer 均已准备好。

从仓库根目录提交下载，例如：

```bash
bash slurm_scripts/submit_download.sh xz313 --test-only
bash slurm_scripts/submit_download.sh xz313
```

第一条只校验调度参数，第二条实际提交。下载不申请 GPU；训练需另外申请 GPU，
并让训练启动器的进程数与 Slurm 分配资源一致。

## 离线环境与 GPU 兼容性

无外网节点应从 HDFS 恢复预构建环境。现有共享 Conda 归档为：

| 用途 | Torch 版本 | HDFS 归档 |
| --- | --- | --- |
| 普通 GPU | `2.6.0+cu124` | `/user/t0/alden/miniforge3_alden.tar.gz` |
| RTX 5090 | `2.11.0+cu130` | `/user/t0/alden/miniforge3_alden_cu130.tar.gz` |

访问上述 HDFS 归档时使用 `export HADOOP_USER_NAME=t0`。这些归档来自共享训练环境，
不代表已经包含 CL-VISTA 各方法的全部依赖。恢复后应核对所选方法的依赖文件、
Python 标准库、核心模块导入及实际 CUDA 运算；尤其检查 Torch、torchvision、
DeepSpeed、FlashAttention 等版本和二进制扩展的兼容性。

已知 RTX 5090 节点包括 `xz290`、`xz306`、`xz312`、`xz313`、`xz319`、`xz323`；
本项目 `slurm_scripts/README.md` 另记录 `xz288` 为 RTX 5090。以实际 GPU 查询结果
为准。RTX 5090 必须使用包含 `sm_120` kernel 的 cu130 环境，不要把普通 cu124
环境直接复制过去，也不要直接照搬 README 中旧版 Torch/CUDA 安装组合。

环境恢复或修复应加锁，避免多个作业同时解压到同一路径。只在依赖更新或归档损坏
时维护环境归档；环境发布与单次实验的代码发布分别处理。重新打包前先验证依赖，
不要用未经验证的环境覆盖共享归档。

当前数据下载脚本优先使用已有 Miniforge Python，并创建独立 venv；缺失时从 HDFS
获取 Miniforge 安装器。它仍需访问 pip 镜像和 ModelScope，并未实现上述完整 Conda
归档的离线恢复；完全离线节点需要提前准备下载环境及数据。

## 分布式训练前的准备

耗时的数据下载、解压、模型下载和缓存构建，应在启动 `torchrun`、DeepSpeed 或
其他分布式启动器之前，以独立任务或单进程完成，并检查成功后再启动训练。
不要把数小时下载放进已初始化的 DDP process group：非主 rank 在 barrier 等待时
可能超时，导致仍在下载的 rank0 被终止。单纯增大分布式超时不能替代预下载。

为本项目新增 Slurm 训练流程时，应保持职责清楚：提交脚本校验配置及资源，作业
脚本准备环境、代码和数据，所选方法的训练入口执行模型逻辑。代码快照应排除数据
缓存、日志和模型产物；缓存应可复用并支持续传。若增加产物上传或通知，应分别报告
训练、上传和通知的结果，避免把通知失败误报为训练失败。
