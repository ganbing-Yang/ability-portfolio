# 猫狗分类：MLP、CNN与数据增强对照

基于教师CIFAR10卷积例题，筛选猫狗并改为二分类。彩色32×32图片共12000张，划分8000训练、2000验证、2000官方测试。

| 模型 | 测试准确率 | 参数量 |
| --- | --- | --- |
| MLP | 59.80% | 393473 |
| CNN | 75.80% | 89585 |
| 增强CNN | 74.30% | 89585 |

增强在这次固定划分和训练预算下没有改善。普通CNN猫召回72.6%、狗召回79.0%；错例和混淆矩阵保留。

![对照结果](results/figures/02_metrics.png)

## 运行

```bash
python -m pip install -r requirements.txt
python catdog_case.py --output results_reproduced
python predict.py --image example.jpg --model results_reproduced/catdog_selected.keras
```

首次训练需要下载官方CIFAR10数据。保存的二进制模型不纳入这个精简仓库，可以重新训练生成，或使用此前完整代码包的results/catdog_selected.keras。训练脚本基于课堂代码改编，增加固定划分、MLP对照、数据增强与评估，使用AI辅助完成。

先读load_data、build_model和metrics；输出Sigmoid分数，以0.5为阈值。模型没有未知类别识别或概率校准，其他图片也可能被强制二选一。打开Notebook浏览已保存结果；本次整理不重新训练。

学习练习：手算猫召回率，解释两张错例，再只改一个增强设置并保存自己的实验记录。
