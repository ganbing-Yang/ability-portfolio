# Walmart 周销售额预测

## 任务与方法

基于 45 家门店、143 周、6435 条销售记录，预测门店周销售额。按时间划分 85 周训练、29 周验证、29 周测试；标准化只拟合训练集。比较门店编号直接输入、门店独热编码及门店与月份编码三种线性方案，使用 NumPy 实现均方误差梯度下降，按验证结果选择模型。

## 结果

| 方法 | 测试 R² | MAE | RMSE |
| --- | --- | --- | --- |
| 门店 + 月份编码线性模型 | 0.9529 | 87030.32 | 115755.26 |
| 训练期各门店均值基线 | 0.9587 | 74628.69 | 108420.29 |

最终模型未超过均值基线。总体 R² 很大程度上反映门店规模差异，不能替代单店波动与误差检查。模型系数仅描述统计关系。

![测试集真实值与预测值](results/figures/04_actual_prediction.png)

[研究报告](研究报告.md) · [Notebook](Walmart_线性回归案例.ipynb) · [完整评价记录](results/summary.json)

## 项目输出

- [walmart_analysis.py](walmart_analysis.py)：数据处理、特征设计、训练与评价。
- [Walmart_线性回归案例.ipynb](Walmart_线性回归案例.ipynb)、[研究报告.md](研究报告.md)：方法与图表分析。
- [results/](results/)：模型参数、预测 CSV、损失记录、图表与汇总。
- [Walmart_Sales.csv](Walmart_Sales.csv)：原始数据。

## 复现

```bash
python -m pip install -r requirements.txt
python walmart_analysis.py --output results_reproduced
```

数据与题目来自课程提供的 Walmart_Sales.csv，题面注明 Kaggle mikhail1681/Walmart Sales。实现与运行验证使用 AI 辅助，当前数值对应仓库保存的实验记录。
