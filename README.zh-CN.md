<div align="center">

# <img src="docs/assets/logo.png" alt="Relay-OPD" height="40"> 传递接力棒：轨迹接力的在线策略蒸馏

[![arXiv](https://img.shields.io/badge/Paper-arxiv%3A2607.26057-b31b1b)](https://arxiv.org/abs/2607.26057)
[![Daily Paper](https://img.shields.io/badge/Daily%20Paper-huggingface-yellow)](https://huggingface.co/papers/2607.26057)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zju-real.github.io/Relay-OPD/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](relay-opd/LICENSE)

<strong>Haolei Xu<sup>1,2&#42;</sup> · Xiaowen Xu<sup>2&#42;</sup> · Haiwen Hong<sup>2&#42;&dagger;</sup> · Zixuan Ni<sup>1</sup></strong><br>
<strong>Hongxing Li<sup>1,2</sup> · Yiwen Qiu<sup>1</sup> · Weiming Lu<sup>1&Dagger;</sup> · Yongliang Shen<sup>1</sup></strong>

<sup>1</sup>浙江大学 &nbsp; <sup>2</sup>阿里巴巴集团 Yuvion 团队

<sup>&#42;</sup>共同第一作者 &nbsp; <sup>&dagger;</sup>项目负责人 &nbsp; <sup>&Dagger;</sup>通讯作者

</div>

## 🔥 概述

**Relay-OPD** 解决了在线策略蒸馏（on-policy distillation）中的*前缀失败*（prefix failure）问题。一个无需标签的**交接触发器**（handoff trigger）——当教师模型 top-1 的 token 是反思（reflection）token，而学生模型 top-K 中未出现任何反思 token 时——会在学生模型生成过程中在线定位已失败的前缀。随后教师模型短暂接管，生成一小段由 L 个段落组成的**教师片段**（teacher leg），再把轨迹交还给学生；受限的**接力预算 (M, L)** 保证了干预既早又局部。整个 rollout 在同一个投机解码引擎中完成（学生模型作为草稿模型，教师模型作为目标模型），并且学生模型会在接力后的轨迹上（包括接力产生的 token 本身）进行优化。

<div align="center">
<img src="docs/assets/teaser.png" alt="Relay-OPD teaser" width="95%">
</div>

<div align="center">
<img src="docs/assets/method.png" alt="Relay-OPD method overview" width="95%">
</div>

## 📢 动态

- **`2026-07-29`**：🔥🔥 我们发布了[论文](https://arxiv.org/abs/2607.26057)，以及完整的训练、消融与评测代码。
- **`2026-07`**：🔥 项目主页上线。

## 📖 结果

在八个数学推理基准上，以 Qwen3-4B-Instruct-2507 作为教师模型、Qwen3-0.6B/1.7B-Non-Thinking 作为学生模型，Relay-OPD 在每个基准上都取得最佳或第二佳的结果——在 1.7B 规模上，平均比标准 OPD 高 **+5.73%**，比最强基线 FastOPD 高 **+1.49%**，同时将平均训练轨迹长度缩短 **50% 以上**。

## 🛠️ 安装

实现代码位于 [`relay-opd/`](relay-opd/)，构建在 [verl](https://github.com/verl-project/verl) 之上。由于投机解码补丁针对 vLLM 内部实现，**vLLM 0.21.0** 是唯一必须锁定的版本。

### Python 环境

```bash
git clone git@github.com:ZJU-REAL/Relay-OPD.git
cd Relay-OPD/relay-opd

conda create -n relay-opd python==3.12 -y
conda activate relay-opd

pip3 install -c environment/vllm-constraints.txt vllm==0.21.0
pip3 install -e .
pip3 install -r requirements-relay-opd.txt
```

验证安装（检查 CUDA 执行、数学评分器，以及 Relay-OPD 打补丁的所有 vLLM 接口）：

```bash
python environment/verify_install.py
```

如需严格复现我们验证过的环境（Linux x86_64、Python 3.12、CUDA 13.0、PyTorch 2.11、vLLM 0.21.0），我们还提供了可选的锁定安装脚本：

```bash
bash environment/create_locked_env.sh
```

### 训练

各脚本通过环境变量接收路径。主 Relay-OPD 运行（论文默认配置：K=5、M=2、L=3；八张 GPU 分配为四张给学生、四张给教师）：

```bash
export STUDENT_MODEL=/path/to/student
export TEACHER_MODEL=/path/to/teacher
export TRAIN_DATA=/path/to/train.parquet
export BENCH=/path/to/eval_parquets
export OUTPUT_DIR=/path/to/output

bash opd/scripts/relay_opd/train.sh
```

所有基线（SFT、SeqKD、GRPO、OPD、FastOPD、TRD、SKD）以及论文中的每一项消融，都在 `opd/scripts/` 下有专门的脚本。完整的脚本索引、离线数据合成方法以及论文的完整配置，请参见 [`relay-opd/README.md`](relay-opd/README.md)。

### 评测

```bash
RUN_NAME=relay_opd \
STEP=35 \
MODEL=/path/to/checkpoint/actor/huggingface \
DATA_DIR=/path/to/eval_parquets \
OUT_ROOT=/path/to/eval_output \
BENCHES=aime24,aime25,aime26,math500,amc23,olympiad,hmmt_feb_2026,hmmt_nov_2025 \
N_SAMPLES=32 MAX_NEW=32768 MAX_MODEL_LEN=34817 DP_SIZE=4 TP=1 \
bash opd/scripts/evaluation/math.sh
```

## ⭐️ 引用

如果本项目对你有帮助，欢迎引用我们。

```bibtex
@misc{xu2026passbatontrajectoryrelayedonpolicy,
      title={Pass the Baton: Trajectory-Relayed On-Policy Distillation},
      author={Haolei Xu and Xiaowen Xu and Haiwen Hong and Zixuan Ni and Hongxing Li and Yiwen Qiu and Weiming Lu and Yongliang Shen},
      year={2026},
      eprint={2607.26057},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.26057},
}
```

## 🤝 致谢

本项目构建于 [verl](https://github.com/verl-project/verl) 与 [vLLM](https://github.com/vllm-project/vllm) 之上。我们感谢这两个项目的作者。
