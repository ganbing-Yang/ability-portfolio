"""
MediaPipe Tasks API 教程版：手部关键点识别

运行方式：
    1. 先把终端 / 命令提示符切换到本文件所在的文件夹。
       Windows 示例：
           cd 你的课程文件夹\\media_pipe
       macOS 示例：
           cd 你的课程文件夹/media_pipe

    2. 安装依赖库。可以直接安装到当前 Python，也可以安装到虚拟环境。
       本课程代码不强制要求虚拟环境；如果老师已经统一配置好环境，可以跳过安装步骤。
       Windows 常用命令：
           py -m pip install -r requirements.txt
       macOS 常用命令：
           python3 -m pip install -r requirements.txt

    3. 运行程序。
       Windows 常用命令：
           py mp_hand.py
       macOS 常用命令：
           python3 mp_hand.py

常见问题：
    - 如果提示 No module named 'mediapipe'，说明依赖库没有安装到当前 Python 环境。
    - 如果提示找不到 hand_landmarker.task，确认模型文件和本脚本在同一个文件夹。
    - 如果摄像头打不开，Windows 和 macOS 都需要检查系统相机权限，并确认摄像头没有被其它软件占用。
    - MediaPipe 使用 RGB 图像，OpenCV 摄像头读出来的是 BGR 图像，所以代码里必须做颜色转换。
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 手部骨架连接关系
# =========================
# MediaPipe Hand Landmarker 会给每只手输出 21 个关键点。
# 每个关键点都有 x、y、z 坐标：
#   x: 横向位置，0 在画面最左边，1 在画面最右边
#   y: 纵向位置，0 在画面最上方，1 在画面最下方
#   z: 深度方向，课堂入门阶段可以先不画出来
#
# 下面的连接关系告诉 OpenCV：
# “第几个点”和“第几个点”之间应该画一条线。
# 例如 (0, 1), (1, 2), (2, 3), (3, 4) 就是一根大拇指的骨架线。
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 大拇指：手腕 -> 拇指根部 -> 拇指指尖
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),   # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),  # 小拇指
    (0, 17)                                  # 手掌外侧边缘
]


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"

# =========================
# 2. 课堂可修改配置
# =========================
# 初学者先不要急着改函数内部逻辑。
# 如果要换摄像头、调画面大小、调识别灵敏度，只改下面这些常量即可。
#
# CAMERA_INDEX:
#   摄像头编号。大多数电脑内置摄像头是 0。
#   如果外接摄像头没有画面，可以试着改成 1 或 2。
CAMERA_INDEX = 0

# FRAME_WIDTH / FRAME_HEIGHT:
#   希望 OpenCV 从摄像头读取的画面大小。
#   分辨率越高，画面越清楚，但电脑要处理的数据也越多，可能更卡。
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# NUM_HANDS:
#   最多识别几只手。课堂演示通常设为 2，表示左右手都可以识别。
NUM_HANDS = 2

# CONFIDENCE:
#   识别置信度阈值。越高越严格，误识别更少，但光线差时可能识别不到。
#   如果识别不到手，可以试着改成 0.4；如果误识别很多，可以改成 0.7。
CONFIDENCE = 0.55


def normalized_landmarks_to_pixels(landmarks, width, height):
    """
    把 MediaPipe 的归一化坐标转换为 OpenCV 的像素坐标。

    MediaPipe 输出的 x、y 一般在 0~1 之间：
        x = 0.25 表示画面宽度的 25% 处
        y = 0.60 表示画面高度的 60% 处

    OpenCV 画点和画线需要像素坐标，所以要乘以 width / height。
    同时用 max/min 把坐标限制在画面范围内，避免手移动到边缘时画线越界。
    """
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        points.append((x, y))

    return points


def draw_hand_landmarks(frame, detection_result):
    """
    在画面上绘制手部关键点、骨架线和左右手文字。

    detection_result.hand_landmarks:
        一个列表。检测到 1 只手，列表里就有 1 组 21 个关键点。

    detection_result.handedness:
        MediaPipe 对左右手的判断结果，例如 Left / Right 和置信度。
    """
    if not detection_result.hand_landmarks:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        return frame

    height, width, _ = frame.shape

    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
        points = normalized_landmarks_to_pixels(hand_landmarks, width, height)

        # 先画线，再画点，这样红色关键点会覆盖在线条上方，更清楚。
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 210, 0), 2)

        for landmark_index, (x, y) in enumerate(points):
            radius = 7 if landmark_index in [0, 4, 8, 12, 16, 20] else 4
            cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)

        # handedness 可能为空，所以读取前要先判断。
        if detection_result.handedness and hand_index < len(detection_result.handedness):
            handedness = detection_result.handedness[hand_index][0]
            label = handedness.category_name
            score = handedness.score
            wrist_x, wrist_y = points[0]

            cv2.putText(
                frame,
                f"{label} hand  confidence={score:.2f}",
                (wrist_x, max(30, wrist_y - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 80, 0),
                2,
            )

    return frame


def create_hand_landmarker():
    """
    创建 Hand Landmarker 检测器。

    MediaPipe Tasks API 的固定步骤：
        1. BaseOptions 指定模型文件路径。
        2. HandLandmarkerOptions 设置任务参数。
        3. create_from_options 真正创建检测器。

    running_mode=VIDEO 表示我们会一帧一帧处理摄像头画面。
    VIDEO 模式必须在 detect_for_video() 里传入递增的时间戳。
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=CONFIDENCE,
        min_hand_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
    )

    return vision.HandLandmarker.create_from_options(options)


def next_timestamp_ms(last_timestamp_ms):
    """
    生成 MediaPipe VIDEO 模式需要的时间戳。

    time.time() 读取系统时间，极少数情况下相邻两帧可能得到相同毫秒值。
    MediaPipe 要求时间戳严格递增，所以这里做一次保护。
    """
    timestamp_ms = int(time.time() * 1000)

    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1

    return timestamp_ms


def main():
    detector = create_hand_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("无法打开摄像头。请检查摄像头编号、系统相机权限，或是否被其它软件占用。")
        detector.close()
        return

    print("手部关键点识别已启动。按 q 退出。")
    print(f"使用模型：{MODEL_PATH}")

    last_timestamp_ms = 0
    last_fps_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("无法读取摄像头画面。")
                break

            # 镜像翻转：让摄像头画面像照镜子一样，学生举左手时屏幕左侧也动。
            frame = cv2.flip(frame, 1)

            # OpenCV 用 BGR，MediaPipe 用 RGB。颜色顺序不转换会影响模型输入。
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            last_timestamp_ms = next_timestamp_ms(last_timestamp_ms)
            detection_result = detector.detect_for_video(mp_image, last_timestamp_ms)

            frame = draw_hand_landmarks(frame, detection_result)

            now = time.time()
            if now > last_fps_time:
                fps = 1.0 / (now - last_fps_time)
            last_fps_time = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}   q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("MediaPipe Hand Landmarker - tutorial", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
