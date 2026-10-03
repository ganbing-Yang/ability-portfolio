# MediaPipe关键点：单图验证与距离计算

使用教师提供的模型文件，读取手、脸和姿态坐标，检查这些结果怎样转换成可用于交互的数据。

| 单图任务 | 关键点结果 |
| --- | --- |
| 手 | 21点 |
| 脸 | 478点 |
| 姿态 | 33点 |
| 全身 | 姿态33点，左右手各21点，脸未检出 |

![手部结果](results/hand.jpg)

## 运行

```bash
python -m pip install -r requirements.txt
python measure_hand.py
python run_landmarks.py
```

measure_hand读取已保存的JSON即可运行，不需要模型权重；它把拇指尖4与食指尖8转换成像素坐标计算距离，再除以腕部0到中指根9的距离，减少尺度影响。

run_landmarks使用IMAGE模式重新推理。运行前把老师media_pipe.zip中的hand_landmarker.task、face_landmarker.task、pose_landmarker.task、holistic_landmarker.task复制到sources/media_pipe/；精简仓库不含这些二进制模型。Linux加载MediaPipe可能需要libgles2运行库。输入出处见inputs/sources.json。

模型已经预训练，此项目完成接口复现、单图输出检查与距离计算，没有训练MediaPipe，也没有验证摄像头、实时捏合或游戏控制。手部距离没有校准触发阈值。脸部形变系数不证明真实情绪。

学习练习：核对x乘图宽、y乘图高，解释为什么直接对归一化坐标求距离会受图像宽高比影响。后续如果实现交互，再测试连续帧稳定性与触发/释放阈值。
