# 复现步骤

## 1. 检查文件

安装锁定依赖后执行`python scripts/verify_release.py`。根目录`SHA256_manifest.json`覆盖静态文件，自身不递归计入。运行后生成的`artifacts/`与`__pycache__/`不属于静态快照。

## 2. 重新计算结果

执行`python scripts/reproduce_results.py --output artifacts/recomputed`。脚本读取全部随附测试预测，重新计算逐日期和种子指标，并核验CRG及等权融合与基模型输出的一致性。α对照、随机缺16组条件下CRG与两种融合方式的配对区间同时重算。

`results/verified/`是从本次独立目录实际复算后导出的结果。部分算法比较为已有测试日期上的探索性分析，输出中保留全部条件。

## 3. 从原始归档重建特征

```text
python scripts/fetch_dataset.py
python scripts/prepare_features.py --output artifacts/rebuilt_features
```

特征重建包含数百个文本轨迹的解析，可能需要数十分钟。它不会覆盖随附缓存。重建后可把`--inputs artifacts/rebuilt_features`传给训练入口。

`preparation.py`从原数值实现抽取特征准备函数，保留函数体，移除调度与账号依赖。其余数值模块保留原可执行AST；少量说明性注释与文档字符串经过清理。源散列和导出散列见`provenance/source_files.json`。

## 4. 训练

TCSS的排序使用B=0训练特征，预测头使用B=1输入。每臂200次环境内整日期重采样；候选C为0.1、1、10；五个预测种子为11、23、37、53、71。训练入口保留500次选择器迭代上限及原相对目标停止条件。预测头仅拟合360训练评分样本，105验证样本用于选参和温度拟合。

`compact_baselines.py`保留紧凑预测头原候选选择规则。`baselines.py`为纠正后的常规分类基线实现，包括概率argmax硬预测与逻辑回归收敛检查；二者使用独立命名空间。

CRG使用62个传感器的训练缩放输入。`train_gate.py`读取固定排名以生成按影响排序的掩码；该排名由训练数据产生。所有基模型、重构和β选择使用训练或验证样本。未来测试标签仅用于最终评价。

训练入口允许显式调用；`--check-only`只检查输入与预处理。此导出版本已执行模块测试、输入检查和全部指标复算，未重复训练完整TCSS或CRG。因此不把入口连通性检查等同于跨平台完整训练复现。

## 5. 数值环境

锁定环境已在Windows/Python 3.10.20验证。历史结果的原运行环境还包含Linux/Python 3.9.18。α=0.75的同环境参照与保存结果具有相同16传感器集合，595条测试类别预测全部一致；概率存在最大约0.001232的环境差异。跨平台结果应同时核对阵列、类别预测和概率差，不要求浮点文件逐字节一致。
