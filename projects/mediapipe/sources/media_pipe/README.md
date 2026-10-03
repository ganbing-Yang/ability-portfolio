# MediaPipe 视觉识别课堂示例

这个文件夹包含 4 个教程版 Python 示例：

- `mp_hand.py`：手部关键点识别，最多识别 2 只手。
- `mp_face.py`：人脸关键点识别，并用 face blendshapes 做简单表情演示。
- `mp_body.py`：身体骨骼 / 姿态关键点识别。
- `mp_holistic.py`：全身综合识别 Holistic Landmarker，同时识别身体、人脸、左右手。

## 1. 环境准备

建议使用 Python 3.10、3.11 或 3.12。Python 3.13 可能因为 MediaPipe 没有对应安装包而安装失败；如果遇到这种情况，建议换 Python 3.12。

本课程代码可以直接安装到当前 Python 环境，也可以安装到虚拟环境。虚拟环境的好处是不会影响电脑上的其它 Python 项目，但不是强制要求。

### Windows 安装方式

```bash
cd 你的课程文件夹\media_pipe
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

运行示例：

```bash
py mp_hand.py
```

### macOS 安装方式

```bash
cd 你的课程文件夹/media_pipe
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

运行示例：

```bash
python3 mp_hand.py
```

### 如果老师要求使用虚拟环境

Windows：

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2. 模型文件

本文件夹已经包含这些模型：

- `hand_landmarker.task`
- `face_landmarker.task`
- `pose_landmarker.task`
- `holistic_landmarker.task`

代码里使用 `Path(__file__).resolve().parent` 自动定位模型文件，所以只要 `.py` 和 `.task` 在同一个文件夹，从哪里运行都不容易报模型路径错误。

如果以后需要重新下载 Holistic 模型：

```bash
curl -L -o holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

## 3. 运行示例

手部识别：

```bash
py mp_hand.py        # Windows
python3 mp_hand.py   # macOS
```

人脸识别：

```bash
py mp_face.py        # Windows
python3 mp_face.py   # macOS
```

身体骨骼识别：

```bash
py mp_body.py        # Windows
python3 mp_body.py   # macOS
```

全身综合识别：

```bash
py mp_holistic.py        # Windows
python3 mp_holistic.py   # macOS
```

运行后按 `q` 退出。

## 4. 修改课堂配置

为了让初学者更容易理解，本课程示例不使用命令行参数。
如果要调整程序，请打开对应 `.py` 文件，修改文件顶部的“课堂可修改配置”常量。

例如 `mp_hand.py` 顶部有这些配置：

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
NUM_HANDS = 2
CONFIDENCE = 0.55
```

如果电脑有多个摄像头，可以把摄像头编号从 0 改成 1：

```python
CAMERA_INDEX = 1
```

如果画面卡顿，可以降低分辨率：

```python
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
```

如果识别不到人、手或脸，可以降低置信度阈值：

```python
CONFIDENCE = 0.4
```

Holistic 示例默认不开启人体分割遮罩。如果想观察“模型认为哪里是人体”，可以开启：

```python
ENABLE_SEGMENTATION_MASK = True
```

## 5. 常见问题

### No module named 'mediapipe'

通常说明依赖没有安装到正在运行代码的那个 Python 里。

Windows 可以检查：

```bash
py --version
py -m pip show mediapipe
py -m pip install -r requirements.txt
```

macOS 可以检查：

```bash
python3 --version
python3 -m pip show mediapipe
python3 -m pip install -r requirements.txt
```

### 无法打开摄像头

检查三件事：

- 摄像头是否被微信、腾讯会议、浏览器等其它软件占用。
- Windows 是否允许当前应用访问相机：设置 -> 隐私和安全性 -> 相机。
- macOS 是否允许当前终端、VS Code、PyCharm 等工具访问相机：系统设置 -> 隐私与安全性 -> 相机。
- 摄像头编号是否正确，可以把脚本顶部的 `CAMERA_INDEX = 0` 改成 `CAMERA_INDEX = 1`。

### 画面能打开但识别很差

优先检查这些条件：

- 光线是否太暗。
- 手或脸是否离摄像头太近，导致关键部位出画。
- 身体是否被桌子、椅子、屏幕边缘遮挡。
- 背景是否特别杂乱。

### 为什么要 BGR 转 RGB

OpenCV 读取摄像头画面时颜色顺序是 BGR，也就是蓝、绿、红。
MediaPipe 模型需要 RGB，也就是红、绿、蓝。
如果不转换，模型看到的颜色和真实画面不一致，识别效果可能变差。

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### 为什么 VIDEO 模式需要 timestamp

MediaPipe 的视频模式会利用上一帧的信息做跟踪。
为了知道“哪一帧在前、哪一帧在后”，每次调用 `detect_for_video()` 都要传入递增的毫秒时间戳。

```python
detection_result = detector.detect_for_video(mp_image, timestamp_ms)
```

## 6. 课堂讲解顺序建议

1. 先讲 `mp_hand.py`：关键点、连接线、左右手、置信度。
2. 再讲 `mp_face.py`：468 点太多，所以只画重点区域；blendshape 是面部动作分数。
3. 再讲 `mp_body.py`：33 个身体点和 visibility，可见度低时线条变暗。
4. 最后讲 `mp_holistic.py`：一个综合模型同时输出身体、人脸、左右手，适合做交互应用。
