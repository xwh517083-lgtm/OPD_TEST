# 本机下载与服务器上传

整个项目目录作为上传根目录，代码在 `relay-opd/`，模型在
`assets/models/`，数据在 `assets/datasets/`。

## 下载状态

模型完成状态见 `assets/models-manifest.json`。只有三个模型的
`complete` 都为 `true` 才表示下载脚本已成功完成三个快照。
下载中断后，在本机 PowerShell 执行：

```powershell
.\download_models.ps1
```

后台下载日志在 `download-logs/models.log`，异常在 `models-error.log`。
在本机 PowerShell 查看实时进度：

```powershell
Get-Content .\download-logs\models.log -Tail 15 -Wait
```

电脑关机、休眠或网络断开会中断/影响下载；重新执行脚本可续传。
每个模型目录内的 `download-manifest.json` 记录固定 revision 和完成状态。
完成前会校验权重文件的官方 SHA256。

不要同时启动多份模型下载脚本。`.incomplete`、`.part` 文件不是完整数据。

训练数据来源：论文 https://arxiv.org/html/2607.26057 明确使用
DAPO-Math-17K 英文子集。官方原始仓库为
https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k ，固定 revision
`65877096c24ffa7abc4e4fa5edb95cf3413a5674`。
原始数据已下载，SHA256 校验结果见 `assets/datasets/data-validation.json`。
`prepare_training_data.py` 已生成 `assets/datasets/processed/train.parquet`。
原文件 1,791,700 行，17,917 个题目 ID，每个 ID 重复 100 次，重复内容已核对。
去重后，按排除汉字、西里尔字母、假名、韩文的规则保留 14,485 条英文候选题。
这只是可复现的语言筛选近似，可能保留拉丁字母的非英文文本，不保证与作者子集完全一致。
原始 Answer: 包装已移除，替换成论文的 boxed system prompt。
详细统计见 `assets/datasets/processed/preprocessing-report.json`。

发现 MATH500 的 4 道题与 5 条训练样本在忽略大小写和空白后完全相同。
已另外生成 `processed/train_no_eval_overlap.parquet`（14,480 条）。
建议自行研究时将基线与新方法都运行在此版本上；它不等同于论文原始训练集。
文本完全匹配检查不能排除改写题目污染。原始转换版本仍完整保留。

评测原始数据及版本记录见 `assets/datasets/benchmarks-manifest.json`。
`prepare_benchmarks.py` 将其转换为训练和评测入口需要的 parquet。
AI-MO 的 AIME 原始文件有多个年份，转换时只保留 URL 中的 2024 年题目。
当前准备的是 AIME 2024、AIME 2025 和 MATH500，不是论文全部八项基准。

## 上传范围

上传 `relay-opd/`、`assets/`、本说明和数据预处理脚本即可。
无需上传 `.download-venv/`、`.download-cache/` 或各模型目录中的 `.cache/`。
不要在下载尚未完成时打包。模型权重通常难以进一步压缩，传目录即可。

服务器必须重新安装 Linux 训练环境，不能使用本机 Windows 虚拟环境。
模型文件和 parquet 数据可跨平台使用。

## 服务器路径

在模型全部下载并校验完成后，在上传根目录执行以下命令：

```bash
ROOT="$(pwd)"
export STUDENT_MODEL="$ROOT/assets/models/Qwen3-0.6B"
# 改为 Qwen3-1.7B 可运行另一学生规模。
export TEACHER_MODEL="$ROOT/assets/models/Qwen3-4B-Instruct-2507"
export TRAIN_DATA="$ROOT/assets/datasets/processed/train_no_eval_overlap.parquet"
export BENCH="$ROOT/assets/datasets/benchmarks"
export OUTPUT_DIR="$ROOT/runs/relay_opd_0.6b"
cd "$ROOT/relay-opd"
python environment/verify_install.py
bash opd/scripts/relay_opd/train.sh
```

训练脚本默认需要 4 张学生侧 GPU 和 4 张教师侧 GPU，正式运行前需按服务器资源调整。
两个学生规模使用不同 OUTPUT_DIR。先做短训练验证，再运行完整实验。
