# Electronic-Nose Drift Toolkit

用于电子鼻长期漂移分析、紧凑阵列选择和通道不可用条件下的识别。包含数值代码、锁定依赖、处理后的特征、逐样本概率、选择记录和可重复计算的指标。

## 快速验证

建议使用Python 3.10。已验证环境为Windows、Python 3.10.20、NumPy 2.0.2、SciPy 1.13.1、scikit-learn 1.6.1。以下命令从仓库根目录执行，适用于PowerShell及常用终端。

```text
python -m venv .venv
```

Windows启用环境：`.venv\Scripts\Activate.ps1`；Linux/macOS：`source .venv/bin/activate`。

```text
python -m pip install -r requirements.lock.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
python scripts/reproduce_results.py --output artifacts/recomputed
```

指标复算不拟合模型，不需要重新下载原始归档。输出包括逐日期与种子分数、日期等权汇总、配对日期bootstrap及核验记录。

## 目录

| 路径 | 内容 |
|---|---|
| `src/enose_drift/` | 特征、日期划分、组稀疏选择、重构与融合算法 |
| `configs/` | 特征、模型、掩码、种子和迭代设置 |
| `scripts/` | 获取数据、预处理、训练、复算与完整性核验入口 |
| `data/processed/` | 584个样本的特征缓存、样本索引及重采样计划 |
| `data/predictions/` | 各条件的标签、类别概率及实验设置 |
| `data/selection/` | 候选参数、验证选择、类别权重和训练掩码 |
| `data/alpha_control/` | 两种α设置的预测和400个选择器检查点 |
| `data/metadata/` | 数据日期、传感器排名和掩码索引 |
| `results/verified/` | 从本目录重新计算并核验的结果 |
| `provenance/` | 源码身份、导出变换和文件核验记录 |
| `docs/` | 数据字典、算法、复现步骤与GitHub使用说明 |

## 方法

1. 以完整采集日期划分训练、验证和测试集；特征缩放参数只由训练样本估计。
2. TCSS在六个时间环境内执行200次整日期重采样，以组稀疏的环境风险目标排列传感器。
3. 紧凑配置保留16个传感器及6个环境变量，再用验证集选择逻辑回归的C。
4. CRG在完整阵列上组合Ridge重构后的RBF-SVM、收缩LDA及多掩码ExtraTrees，根据验证类别召回率计算融合权重。

紧凑选择与CRG使用不同的输入配置，分别执行。CRG结果不表示16传感器子阵列上的联合验证。随机缺失和按影响排序缺失均保留完整输出。缺失掩码是人为设置的计算条件，不代表采集到了真实硬件故障。

## 运行已有配置

检查输入而不训练：

```text
python scripts/train_compact.py --alpha 0.75 --output artifacts/compact_check --check-only
python scripts/train_gate.py --output artifacts/gate_check --check-only
```

执行训练时移除`--check-only`，按本机资源设置`--n-jobs`。两种α分别使用独立输出目录：

```text
python scripts/train_compact.py --alpha 0.75 --output artifacts/compact075 --n-jobs 4
python scripts/train_compact.py --alpha 0 --output artifacts/compact0 --n-jobs 4
python scripts/train_gate.py --output artifacts/crg --n-jobs 4
```

完整CRG训练包含多个基模型和候选设置，耗时明显高于指标复算。默认每个工作进程限制为一个数值计算线程，不要求GPU。

## 数据与结果

原数据为Long-Term Drift Behavior of Electronic Nose，数据集DOI：[10.5281/zenodo.15681119](https://doi.org/10.5281/zenodo.15681119)，许可CC BY 4.0。数据署名和派生处理说明见[NOTICE.md](NOTICE.md)。原始归档的获取和SHA-256验证由`python scripts/fetch_dataset.py`完成。

详细的数据结构见[数据字典](docs/DATA_DICTIONARY.md)，重建特征见[复现步骤](docs/REPRODUCIBILITY.md)，已核验的数值与收敛状态见[结果说明](docs/RESULTS.md)。

代码采用MIT许可；派生数据和数据说明按CC BY 4.0提供。依赖库保留各自许可。本项目未包含依赖安装目录。
