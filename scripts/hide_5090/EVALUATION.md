# xz313 七任务 HiDe 评测

默认训练产物为作业 29435637 的七阶段检查点。

## 本地提交

```bash
cd /home/alden/CL-VISTA
python slurm_scripts/submit_hide_eval.py --test-only
# 正式预测：各阶段评测已学任务，28 组，约 56,000 条
python slurm_scripts/submit_hide_eval.py
# 可选：仅最终检查点评测七任务，7 组，约 14,000 条
python slurm_scripts/submit_hide_eval.py --scope final --output-root /cache/hpc_user_alden/outputs/hide-eval-29435637-final
# 推荐首次先小样本：每组 4 条，同样使用全部七阶段
python slurm_scripts/submit_hide_eval.py --limit 4 --output-root /cache/hpc_user_alden/outputs/hide-eval-29435637-smoke
```

4 张 RTX5090、16 CPU、128 GB 内存、24 小时时限；作业名 alden_eval_predict。
邮件 BEGIN,END,FAIL 发往 FT_USER_ALDEN。模型、数据不下载，源码随提交传送。
日志 `/cache/hpc_user_alden/alden_eval_predict-<JOBID>.log/.err`。
阶段和分片日志保存在输出目录 `stage00/Counting/part0.log` 等。

预测和裁判评分是独立阶段。默认命令只生成预测，不代表评分完成。
默认输出 `/cache/hpc_user_alden/outputs/hide-eval-29435637`。
再次提交相同参数会验证并跳过已完成的 stage/task 预测组；未完成组重新生成。
不要并发向同一输出目录提交。不同 scope/limit 必须使用不同目录。

## 裁判评分

训练环境 Transformers 4.31 不支持 Qwen3 MoE。
先准备并验证独立的 Qwen3 Python 环境（Torch 支持 RTX5090、Transformers 支持 qwen3_moe，
且具备原评分器需要的 Accelerate、FlashAttention）。此提交脚本不安装依赖、不修改训练环境。
现已在 xz313 准备 `/cache/hpc_user_alden/envs/qwen3-judge/bin/python`（Transformers 4.57.1），复用已有 Torch/CUDA；已通过 Qwen3 MoE 导入检查。
`--judge-python` 必须是该环境在 xz313 上的 Python 绝对路径，例如：

```bash
python slurm_scripts/submit_hide_eval.py --phase judge \
  --judge-python /cache/hpc_user_alden/envs/qwen3-judge/bin/python
```

上面的 Python 路径已创建。完整评分验证结果另见下文。
若预测使用 final 或 limit，评分需传相同 scope/limit/output-root。
裁判使用原项目提示词和生成参数；包装器使用 batch=8，Qwen3 仅加载一次，跨组复用。
未成功解析的裁判回复会使总控报错，不允许静默将这种情况当作干净结果。
修正后重跑前须移除相应失败的 qwen3 单条缓存（原评分器会缓存错误的零分）。
最终 `scores.json` 是各阶段/任务的归一化平均评分（0～100），不是额外定义的精确匹配准确率。
此入口不自动计算论文 AP/AF 等汇总指标。

## 实现与验证边界

复用原项目 run_inference_video_qa.py 和 eval_video_qa_qwen.py，未修改其模型算法。
预测保持原参数 temperature=0.1、max_new_tokens=256；四个单卡进程分片推理。
检查点别名含 videollava-lora，确保原加载器进入 HiDe LoRA 分支；num-task 取阶段已学任务数，
避免把 Space 或未训练专家纳入路由。基座/视觉/文本模型均为节点本地路径。
按 question_id 对齐答案，核验预测覆盖率，缺视频不允许默默漏评。

已完成 Python 语法、提交参数、节点答案字段检查；GPU 小样本验证进行中。
评测提交器显式加入 Torch cu130 自带 NVRTC 库目录，解决推理 JIT 找不到 libnvrtc-builtins.so.13.0 的问题。
跳过 Space 的七任务评测不等同于论文完整八任务表 2。

## 2026-09-27 四卡端到端验证

预测作业 29477309：28 组 × 4 条，共 112 条，COMPLETED 0:0，33 分 52 秒。
评分作业 29477407：同一批 112 条全部评分和解析成功，COMPLETED 0:0，1 分 7 秒（含 Qwen3 加载）。
输出 `/cache/hpc_user_alden/outputs/hide-eval-29435637-smoke`。
首次预测作业 29477299 因 NVRTC 库搜索路径失败，启动器已修复；不涉及模型算法修改。
Qwen3 环境 Transformers 4.57.1 / tokenizers 0.22.2，安装在独立 venv 中，不覆盖训练环境。
