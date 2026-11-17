# 数据字典

## 原始数据与划分

`data/dataset.json`给出数据集身份、许可、下载地址、文件大小及SHA-256。可用日期按时间顺序划为训练24天、验证7天、测试8天。参考样本不进入评分集；评分样本共584个，分别为360、105、119个。

`data/metadata/measurement_manifest.csv`保存采集索引；`scored_sample_ids.csv`和`data/processed/sample_manifest.csv`保存评分样本及划分。`sample_id`是数据集内部相对文件名，`date`是采集日期，`label`是已知物质及浓度类别。环境编号仅用于训练日期。

五类顺序为：`2-Phenylethanol 1000 ppm`、`2-Phenylethanol 200 ppm`、`Diacetyl 0.1 ppm`、`Diacetyl 1 ppm`、`EtOH`。

## 特征缓存

`data/processed/primary_inputs.npz`可用`numpy.load(..., allow_pickle=False)`读取：

| 数组 | 含义 |
|---|---|
| `X_rank` | 360×440，用B=0输入生成，只在训练评分样本上拟合稳健缩放，供排序使用 |
| `X_predict` | 584×440，B=1的未缩放输入，行顺序对应sample_manifest.csv |
| `y_rank`、`train_dates` | 排序输入的标签与日期 |
| `environments` | 360个训练样本的六环境归属 |
| `group_ids`、`group_cols` | 62个传感器及每个传感器的7个响应特征位置 |
| `always` | 6个温湿度特征的位置，保留但不接受传感器组惩罚 |
| `replicate_seeds` | 选择器种子11派生的200个随机数种子 |

`feature_index.json`提供440个变量名称及分组。七个响应特征为幅值、初始斜率、总体斜率、曲线面积、达峰时间、恢复速率和恢复时间参数。16个传感器加6个环境变量共118个预测变量。

B表示用于同日校正的前置参考数量；B=1只允许使用时间上先于目标样本的可用参考。恢复特征敏感性删除每个传感器的两个恢复描述量，仍保留环境变量。

## 预测与选择记录

Parquet文件的公共字段：`method`、`split`、`seed`、`sample_id`、`date_id`、`y_true`、`y_pred`及五个`p_类别名`概率列。部分文件还包括：

- `sensor_budget`、`blank_budget`：传感器数量和参考预算。
- `mask_strategy`、`unavailable_count`、`mask_id`：通道不可用策略、数量及掩码标识。
- `selected_groups`、`always_include_columns`：选中传感器与保留的环境变量。
- `selected_C`、`selected_beta`：由验证集选出的参数。

`budget_predictions.parquet`包含TCSS、混合环境组套索和LDA。`corrected_baseline_predictions.parquet`包含纠正硬预测与选参规则后的mRMR-SVM、RBF-SVM和CORAL-SVM。被替代的旧基线不作为当前有效基线重复计数。

`reconstruction_predictions.parquet`保存三基模型和熵门控输出。`crg_predictions.parquet`、`equal_fusion_predictions.parquet`及`global_fusion_predictions.parquet`使用相同三基模型输出。全局权重共享β=10，并未独立搜索更大的β网格。等权和全局融合由保存概率派生，没有新增模型拟合。

方法名后缀`-unscaled`指概率温度校准前的版本，输入特征仍采用训练集拟合的稳健缩放。验证预测在α对照文件中是校准前概率，测试预测是校准后概率；指标脚本仅评价测试行。

## 选择器检查点

`data/alpha_control/{reference,alpha0}/replicates/*.npz`包含系数、截距、目标轨迹、六环境损失、组大小、入选标志、排名、随机种子、迭代数与停止状态。`reference`对应α=0.75，`alpha0`对应α=0。`signature`是原运行的工作负载身份，不是本次导出目录的身份；本次导出文件由根目录散列清单覆盖。

## 指标

每个日期、种子及条件分别计算固定五类macro-F1、平衡准确率、MCC、NLL和Brier分数。同日先对种子平均，再对日期等权平均。配对bootstrap以整日期为抽样单位，重复10,000次。种子及同日技术重复不增加独立日期数。
