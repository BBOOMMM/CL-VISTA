# CL-VISTA 数据下载

在项目根目录提交，默认用集群账户 `hpc_user_alden`，下载到 xz288 本地磁盘：

```bash
bash slurm_scripts/submit_download.sh xz288
```

切换到 xz313：

```bash
bash slurm_scripts/submit_download.sh xz313
```

两条命令选一条即可，避免在两个节点重复下载约 749 GB 数据。2026-09-24 查询时，
xz288、xz313 在 `yzgpu` 分区，均有 8 张 RTX5090；xz323 在 `hpc` 分区，处于
`tony_reservevation` 预约中，当前账户不在该预约允许用户列表内，因此脚本不使用它。

下载任务使用 8 CPU、16 GB 内存、最长 48 小时，不申请 GPU。作业在计算节点创建独立
Python venv 并安装 ModelScope，不要求训练环境已安装。参照
`/home/alden/alden-training/run.sbatch`，优先使用自己目录中的 Miniforge Python；
若不存在，则从相同的 HDFS 安装包安装到独立的 `cl-vista-miniforge3` 目录。
pip 使用清华镜像，节点需要能访问该镜像、ModelScope，以及首次安装时的 HDFS。
也可通过 `PYTHON_BIN` 指定节点上带 venv 支持的 Python 3。

默认数据目录：`/cache/hpc_user_alden/datasets/CL-VISTA`。
默认环境目录：`/cache/hpc_user_alden/envs/cl-vista-download`。
工作目录、Slurm 日志、数据、安装环境、pip/ModelScope/Conda 缓存及临时文件
均放在自己的 `/cache/hpc_user_alden` 下，不使用系统临时目录。
缓存位于 `cl-vista-cache/`，临时文件位于 `cl-vista-tmp/<JOBID>/`。
与原训练脚本一样，计算节点上的 `/cache/hpc_user_alden` 须在提交前已存在且可写，
Slurm 才能进入工作目录并打开日志；作业会创建其余子目录。
脚本不依赖提交机的项目目录在计算节点上可见。

只验证调度参数，不实际提交：

```bash
bash slurm_scripts/submit_download.sh xz288 --test-only
```

先下载 STAR 任务进行小规模验证（仍约 16.6 GB）：

```bash
DOWNLOAD_INCLUDE='STAR/*' bash slurm_scripts/submit_download.sh xz288
```

自定义数据目录（必须在目标节点自己的目录下，环境变量值不要包含逗号）：

```bash
DATA_DIR=/cache/hpc_user_alden/datasets/CL-VISTA \
  bash slurm_scripts/submit_download.sh xz288
```

查看作业：

```bash
sudo -u hpc_user_alden env SLURM_CONF_SERVER=xz01 squeue -u hpc_user_alden
```

日志在**计算节点** `/cache/hpc_user_alden/cl_vista_download_<JOBID>.out` 和 `.err`。
可以使用已有的 `slogin xz288 yzgpu` 登录节点查看。脚本自动重试三次；失败或超时后，
向同一节点、同一目录重新提交，保留已有文件及 ModelScope 缓存以便续传。

下载完成后还需解压分卷；保留分卷并解压建议预留至少 1.6 TB，Space 数据和模型另计。
在数据所在节点执行：

```bash
cd /cache/hpc_user_alden/datasets/CL-VISTA
set -euo pipefail
for task in Counting GUI Movie Science Sports STAR Traffic; do
  cat "$task/$task.tar.part"* | tar -xf - -C .
done
tar -xzf CL-VISTA_json.tar.gz -C .
```

标注包解压到 `json_row_unified/`。Space 视频不在下载仓库内，需按项目根目录 README
申请 ScanNet 数据权限，并使用下载得到的 `Space/convert_video.py` 转换视频。

### xz313 七个非 Space 数据集解压

`extract_cl_vista.sbatch` 不申请 GPU、不下载文件，逐个解压 STAR、Science、GUI、
Movie、Traffic、Sports、Counting，并只解压这七个任务的 unified 标注。
分卷通过管道读取，不生成额外的合并 tar，原始分卷保留。

节点上先创建 `/cache/hpc_user_alden/cl-vista-extract/`，将本目录中的
`extract_cl_vista.sbatch` 与 `verify_extracted.py` 放入该目录，再提交 sbatch。
脚本与下载器共用数据目录 `.download.lock`，已成功解压且分卷名称、大小、修改时间
未变化的任务通过 `.extract-state/<task>.done` 跳过；中断的任务重新解压。
若手动删除过解压文件，需先移走对应 done 标记才能重新解压该任务。

任务申请 2 CPU、8 GB 内存，日志与成员清单位于节点 `cl-vista-extract/`。
完成后检查任务原始与 unified JSON 是否可解析、视频是否为空，以及所有标注中的
`video` / `video_path` 引用是否存在，报告为 `reports/verification.json`。
七任务作业显式传入 `--exclude-task Space`：replay/router 中跨任务引用的缺失
Space 视频仍完整记录，但不算作这次七任务解压失败；不删除或改写这些标注。
Counting 原始归档含 3 个未被这七任务标注引用的零字节视频，已核对 tar 成员原始大小。
作业使用 `--allow-unreferenced-empty` 保留并报告这些文件；标注实际引用的空文件
仍会使检查失败。三个原始空文件是 `Counting/youtube/` 下的 `sV4iCDie4fw.mp4`、
`XZPssQgaXig.mp4`、`zgbPVscQqSY.mp4`。
该检查不等于对所有视频进行完整解码，也不等于与远端校验和逐一比对。

`/cache` 按节点本地存储处理，不假定各节点共享。后续训练应固定到数据所在节点，
例如在训练 sbatch 中使用：

```bash
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz288
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX5090:4
```

这里的四卡设置仅说明训练资源；训练启动命令和分布式进程数还需按所选方法配置。
本目录提供数据下载与解压作业，不自动启动训练。

### xz313 HiDe 七任务训练（跳过 Space）

本地提交：`python slurm_scripts/submit_hide_seven.py`。
仅检查：`python slurm_scripts/submit_hide_seven.py --test-only`。
4 张 RTX5090，每卡 batch 2，累积 8，全局 batch 64；训练结束不自动评估。
提交时携带隔离代码快照，不需要手工复制到节点。
详细说明见 [SEVEN.md](../scripts/hide_5090/SEVEN.md)。
