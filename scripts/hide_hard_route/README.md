# HiDe 全层单专家路由推理变体

不重新训练，读取作业 29435637 的已训练专家及锚点。原 HiDe 源码与检查点均不修改。

每个问题在解码器第 0 层之前，由原多模态预处理计算视觉与文本锚点相似度：
对两种相似度取平均，用原温度 0.1 的 softmax 分数取 argmax。仅在已学专家中选择；
最终模型为专家 0～6。平分时取最低编号。这里不使用测试任务名或标准答案选择专家。

全部 32 层的 q/k/v/o、gate/up/down 投影统一使用同一个专家：
`W x + scaling * B_task(A_task(x))`。不求和、不平均、不融合专家，不把 LoRA 合并进基座。
后续生成 token 保留当前样本的选择；下一条问题重新路由。当前仅支持每 GPU 单样本。
这是同一视频+问题级别的一次性路由，不是每层或每个 token 重选专家。

通过实例绑定方法改变独立进程内的推理行为。保留原解码参数与预处理，不改变原项目文件。
原加载器名为 merge_and_unload 的实现没有替换这些专家层；安装时会校验所有层的 7 个 LoRA 模块均存在。
每个分片额外输出 `partN.routes.jsonl`，包含视频、问题、选中的专家及 softmax 分数。

## 本地提交（仅预测最终七任务）

```bash
cd /home/alden/CL-VISTA
python slurm_scripts/submit_hide_hard_route.py
```

默认 4 张 RTX5090，24 小时，xz313/yzgpu，作业名 alden_hard_route_predict。
BEGIN,END,FAIL 通知发送至 FT_USER_ALDEN。输出独立目录：
`/cache/hpc_user_alden/outputs/hide-hard-route-29435637-final`。
不会覆盖原 HiDe 评测目录或修改正在运行的作业。

首次建议先小样本（每任务 4 条）：

```bash
python slurm_scripts/submit_hide_hard_route.py --limit 4 \
  --output-root /cache/hpc_user_alden/outputs/hide-hard-route-29435637-smoke
```

如需等原预测作业结束后才运行，可添加 `--dependency afterany:29480241`；
这只控制调度，不继承原作业的预测结果。

提交裁判评分（将 JOBID 替换为变体预测作业号）：

```bash
python slurm_scripts/submit_hide_hard_route.py --phase judge \
  --judge-python /cache/hpc_user_alden/envs/qwen3-judge/bin/python \
  --dependency afterok:JOBID
```

对小样本评分需同样传入 `--limit 4` 和 smoke 输出目录。
评分生成 scores.json 与 final_metrics.json（七任务分数及 MFN）。
仍使用同一投影层和基座，只有专家使用策略变化；这不是论文原版 HiDe。

## 验证

CPU 单元测试覆盖：所有层选择相同专家、与单专家公式数值一致、缓存 token 保持路由、
新问题重新路由、拒绝多样本混用路由。Slurm --test-only 调度检查通过。
尚未执行变体真实 GPU 视频推理，不把单元测试视为端到端验证。
