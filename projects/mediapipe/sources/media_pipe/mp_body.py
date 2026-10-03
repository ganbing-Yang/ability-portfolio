"""
MediaPipe Tasks API 教程版：身体骨骼 / 姿态关键点识别

本示例使用 Pose Landmarker 识别人体 33 个关键点，例如：
    - 鼻子、眼睛、耳朵
    - 肩膀、手肘、手腕
    - 髋部、膝盖、脚踝、脚尖

运行方式：
    1. 先把终端 / 命令提示符切换到本文件所在的文件夹。
       Windows 示例：cd 你的课程文件夹\\media_pipe
       macOS 示例：cd 你的课程文件夹/media_pipe

    2. 安装依赖库。可以直接安装到当前 Python，也可以安装到虚拟环境。
       Windows 常用命令：py -m pip install -r requirements.txt
       macOS 常用命令：python3 -m pip install -r requirements.txt

    3. 运行程序。
       Windows 常用命令：py mp_body.py
       macOS 常用命令：python3 mp_body.py

拍摄建议：
    - 想看完整骨架，最好让上半身或全身尽量出现在画面里。
    - 光线太暗、背景太复杂、身体被桌子遮挡时，关键点容易抖动或缺失。
    - 如果识别不到，可以把文件顶部的 CONFIDENCE 从 0.55 调低到 0.4。
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "pose_landmarker.task"

# =========================
# 1. 可修改配置
# =========================
# 这些常量集中放在文件顶部，不用学习命令行参数也能修改程序行为。
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 识别置信度阈值：
#   数值越高，模型越“谨慎”；数值越低，模型越“容易接受结果”。
#   身体被桌子遮挡或画面较暗时，可以改成 0.4。
CONFIDENCE = 0.55


# =========================
# 2. 人体骨骼连接关系
# =========================
# Pose Landmarker 输出 33 个关键点。
# 这些编号是固定的，例如：
#   0  鼻子
#   11 左肩，12 右肩
#   13 左肘，14 右肘
#   15 左手腕，16 右手腕
#   23 左髋，24 右髋
#   25 左膝，26 右膝
#   27 左脚踝，28 右脚踝
#
# POSE_CONNECTIONS 的作用和手部示例一样：
# 告诉程序哪些点之间应该连线，连起来之后就像一个“火柴人骨架”。
POSE_CONNECTIONS = [
    # 脸部附近
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),

    # 上半身
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    # 手部方向。Pose 模型只给手部大概位置，不像 Hand 模型那样给完整手指。
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),

    # 躯干
    (11, 23), (12, 24),
    (23, 24),

    # 下半身
    (23, 25), (25, 27),
    (24, 26), (26, 28),

    # 脚部
    (27, 29), (29, 31),
    (28, 30), (30, 32),
    (27, 31), (28, 32),
]


IMPORTANT_POINTS = {
    0: "nose",
    11: "L shoulder",
    12: "R shoulder",
    13: "L elbow",
    14: "R elbow",
    15: "L wrist",
    16: "R wrist",
    23: "L hip",
    24: "R hip",
    25: "L knee",
    26: "R knee",
    27: "L ankle",
    28: "R ankle",
}


def normalized_landmarks_to_pixels(landmarks, width, height):
    """
    把 MediaPipe 的归一化坐标转成 OpenCV 像素坐标。

    Pose 关键点通常还带有 visibility：
        visibility 越高，说明这个点越可能真的被摄像头看到了。
    本示例仍然先把点画出来，让学生能观察遮挡时点会如何变化。
    """
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        visibility = getattr(landmark, "visibility", 1.0)
        points.append((x, y, visibility))

    return points


def draw_pose_landmarks(frame, detection_result):
    """
    在画面上绘制人体骨骼关键点和连接线。

    detection_result.pose_landmarks:
        对 Pose Landmarker 来说是一个列表。
        本示例 num_poses=1，所以通常只包含 1 个人的 33 个关键点。
    """
    if not detection_result.pose_landmarks:
        cv2.putText(
            frame,
            "No body detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        return frame

    height, width, _ = frame.shape

    for pose_landmarks in detection_result.pose_landmarks:
        points = normalized_landmarks_to_pixels(pose_landmarks, width, height)

        for start, end in POSE_CONNECTIONS:
            if start < len(points) and end < len(points):
                start_x, start_y, start_visibility = points[start]
                end_x, end_y, end_visibility = points[end]

                # 如果某条线的两个端点都比较可信，就画得更亮。
                # 如果点可能被遮挡，就用暗一点的颜色，帮助学生理解“置信度”。
                average_visibility = (start_visibility + end_visibility) / 2
                color = (0, 255, 0) if average_visibility > 0.5 else (0, 120, 120)

                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)

        for landmark_index, (x, y, visibility) in enumerate(points):
            color = (0, 0, 255) if visibility > 0.5 else (0, 120, 255)
            radius = 6 if landmark_index in IMPORTANT_POINTS else 4
            cv2.circle(frame, (x, y), radius, color, -1)

            if landmark_index in IMPORTANT_POINTS:
                cv2.putText(
                    frame,
                    IMPORTANT_POINTS[landmark_index],
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

        cv2.putText(
            frame,
            f"Pose landmarks: {len(points)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )

    return frame


def create_pose_landmarker():
    """
    创建 Pose Landmarker 检测器。

    num_poses=1 表示只识别画面中最主要的 1 个人。
    如果课堂要多人演示，可以改成 2 或更多，但速度会下降。
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=CONFIDENCE,
        min_pose_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
    )

    return vision.PoseLandmarker.create_from_options(options)


def next_timestamp_ms(last_timestamp_ms):
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms


def main():
    detector = create_pose_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("无法打开摄像头。请检查摄像头编号、系统相机权限，或是否被其它软件占用。")
        detector.close()
        return

    print("身体骨骼识别已启动。按 q 退出。")
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

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            last_timestamp_ms = next_timestamp_ms(last_timestamp_ms)
            detection_result = detector.detect_for_video(mp_image, last_timestamp_ms)

            frame = draw_pose_landmarks(frame, detection_result)

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

            cv2.imshow("MediaPipe Pose Landmarker - tutorial", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
