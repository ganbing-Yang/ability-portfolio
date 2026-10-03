# Walmart周销售额预测

从45家门店的6435条周记录出发，比较门店编码方式与线性回归，检查简单模型是否真的有效。

## 结果

| 方法 | 测试R² | 测试MAE | 测试RMSE |
| --- | --- | --- | --- |
| 门店与月份编码线性回归 | 0.9529 | 87030.32 | 115755.26 |
| 每家门店的训练期均值 | 0.9587 | 74628.69 | 108420.29 |

模型未超过基线。保留这个结果，重点分析门店差异、时间划分和误差。指标来自results/summary.json，未重新训练。

## 方法与运行

按周划分85周训练、29周验证、29周测试。标准化只拟合训练集。NumPy实现MSE梯度下降，验证集选模型，最后报告测试指标。

```bash
python -m pip install -r requirements.txt
python walmart_analysis.py --output results_reproduced
```

打开[研究报告](研究报告.md)或Notebook查看已有结果。先读make_features、train_linear与scores。原始数据为用户提供的Walmart_Sales.csv，数据集来源说明在脚本中。

学习练习：独立用预测CSV核对MAE；解释模型为什么未超过均值基线。结果不支持因果推断，也不保证未知未来周的预测效果。
