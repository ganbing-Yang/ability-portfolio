"""
MediaPipe Tasks API 教程版：全身综合识别 Holistic Landmarker

Holistic Landmarker 是一个“综合版”模型：
    - Pose：身体姿态关键点，33 个
    - Face：人脸关键点，468 个
    - Left Hand：左手关键点，21 个
    - Right Hand：右手关键点，21 个

适合做什么：
    - 课堂演示“一个模型同时理解身体、脸和手”
    - 姿态交互、体感游戏、动作分析、手语或表情动作的入门实验

运行方式：
    1. 先把终端 / 命令提示符切换到本文件所在的文件夹。
       Windows 示例：cd 你的课程文件夹\\media_pipe
       macOS 示例：cd 你的课程文件夹/media_pipe

    2. 安装依赖库。可以直接安装到当前 Python，也可以安装到虚拟环境。
       Windows 常用命令：py -m pip install -r requirements.txt
       macOS 常用命令：python3 -m pip install -r requirements.txt

    3. 运行程序。
       Windows 常用命令：py mp_holistic.py
       macOS 常用命令：python3 mp_holistic.py

如果运行较卡：
    - 把文件顶部的 FRAME_WIDTH / FRAME_HEIGHT 改小，例如 960 / 540。
    - 保持 ENABLE_SEGMENTATION_MASK = False，关闭人体分割遮罩。
    - 提高光线亮度，让模型更容易识别

如果提示无法导入 HolisticLandmarker：
    - 请先升级 mediapipe：python -m pip install -U mediapipe
    - Holistic Landmarker 属于较新的 Tasks API，旧版本 mediapipe 可能没有这个类。
"""

import time
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

# 有些 MediaPipe 版本还没有把 HolisticLandmarker 暴露到 vision 命名空间。
# 从具体模块导入更稳，也能让学生看到“类来自哪里”。
try:
    from mediapipe.tasks.python.vision.holistic_landmarker import (
        HolisticLandmarker,
        HolisticLandmarkerOptions,
    )
except ImportError as exc:
    raise ImportError(
        "当前 mediapipe 版本不包含 HolisticLandmarker。"
        "请运行：python -m pip install -U mediapipe"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "holistic_landmarker.task"

# =========================
# 1. 课堂可修改配置
# =========================
# Holistic 同时识别身体、人脸和双手，计算量比单项识别更大。
# 如果电脑较卡，可以优先把 FRAME_WIDTH / FRAME_HEIGHT 调小。
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 识别置信度阈值：
#   0.55 是课堂演示里比较稳的起点。
#   如果识别不到手或脸，可以改成 0.4。
#   如果背景中出现误识别，可以改成 0.7。
CONFIDENCE = 0.55

# 是否开启人体分割遮罩：
#   False：只画关键点和骨架，速度更快，推荐初学时使用。
#   True：额外给人体区域叠加颜色，效果更直观，但会更耗性能。
ENABLE_SEGMENTATION_MASK = False


# =========================
# 2. 手、身体、脸的连接关系
# =========================
# 为了方便学生对比，手和身体的连接关系与前面三个单项示例保持一致。
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32),
    (27, 31), (28, 32),
]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109, 10,
]

LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246, 33,
]

RIGHT_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 398, 384, 385, 386, 387, 388, 466, 263,
]

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 61,
]


FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


@lru_cache(maxsize=8)
def load_font(font_size):
    """加载中文字体，并缓存结果，避免每帧反复读取字体文件。"""
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def draw_chinese_lines(frame, lines, x=20, y=18, font_size=26, color=(255, 255, 0)):
    """在 OpenCV 图像上绘制中文说明面板。"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = load_font(font_size)

    line_height = int(font_size * 1.35)
    for index, line in enumerate(lines):
        draw.text((x, y + index * line_height), line, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def normalized_landmarks_to_pixels(landmarks, width, height):
    """
    把 MediaPipe 关键点转换成像素坐标。

    Holistic 的结果里，pose、face、hand 都使用归一化坐标。
    这让模型不依赖具体分辨率，但绘制时必须乘以当前画面的宽高。
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


def draw_connections(frame, points, connections, color, thickness=2, use_visibility=False):
    """
    根据连接关系画线。

    use_visibility=True 时，会根据关键点可见度把线画亮或画暗。
    Pose 关键点常有 visibility；手和脸通常不需要这个判断。
    """
    for start, end in connections:
        if start < len(points) and end < len(points):
            start_x, start_y, start_visibility = points[start]
            end_x, end_y, end_visibility = points[end]

            line_color = color
            if use_visibility:
                average_visibility = (start_visibility + end_visibility) / 2
                line_color = color if average_visibility > 0.5 else (70, 120, 120)

            cv2.line(frame, (start_x, start_y), (end_x, end_y), line_color, thickness)


def draw_hand(frame, hand_landmarks, label, color):
    """绘制一只手的 21 个关键点。"""
    if not hand_landmarks:
        return 0

    height, width, _ = frame.shape
    points = normalized_landmarks_to_pixels(hand_landmarks, width, height)

    draw_connections(frame, points, HAND_CONNECTIONS, color, 2)

    for index, (x, y, _visibility) in enumerate(points):
        radius = 6 if index in [0, 4, 8, 12, 16, 20] else 4
        cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)

    wrist_x, wrist_y, _ = points[0]
    cv2.putText(
        frame,
        label,
        (wrist_x, max(30, wrist_y - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return len(points)


def draw_pose(frame, pose_landmarks):
    """绘制身体 33 个姿态关键点。"""
    if not pose_landmarks:
        return 0

    height, width, _ = frame.shape
    points = normalized_landmarks_to_pixels(pose_landmarks, width, height)

    draw_connections(frame, points, POSE_CONNECTIONS, (0, 255, 0), 2, use_visibility=True)

    for index, (x, y, visibility) in enumerate(points):
        color = (0, 0, 255) if visibility > 0.5 else (0, 120, 255)
        radius = 6 if index in [0, 11, 12, 15, 16, 23, 24, 27, 28] else 4
        cv2.circle(frame, (x, y), radius, color, -1)

    return len(points)


def draw_face(frame, face_landmarks):
    """
    绘制脸部关键区域。

    Holistic 返回的人脸点很多，直接画 468 个点会太密。
    所以这里只画轮廓、眼睛和嘴巴，再抽样画少量白点。
    """
    if not face_landmarks:
        return 0

    height, width, _ = frame.shape
    points = normalized_landmarks_to_pixels(face_landmarks, width, height)

    draw_connections(frame, points, list(zip(FACE_OVAL, FACE_OVAL[1:])), (0, 255, 0), 2)
    draw_connections(frame, points, list(zip(LEFT_EYE, LEFT_EYE[1:])), (255, 255, 0), 2)
    draw_connections(frame, points, list(zip(RIGHT_EYE, RIGHT_EYE[1:])), (255, 255, 0), 2)
    draw_connections(frame, points, list(zip(LIPS_OUTER, LIPS_OUTER[1:])), (0, 0, 255), 2)

    for index, (x, y, _visibility) in enumerate(points):
        if index % 10 == 0:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    return len(points)


def get_score(blendshape_dict, name):
    return blendshape_dict.get(name, 0.0)


def summarize_face_expression(face_blendshapes):
    """
    从 Holistic 的 face_blendshapes 中提取几个课堂可观察指标。

    注意 HolisticLandmarkerResult.face_blendshapes 不是“多张脸列表”，
    它是一张脸对应的一组分类分数。
    """
    if not face_blendshapes:
        return "表情：未输出 blendshape"

    blendshape_dict = {
        category.category_name: category.score
        for category in face_blendshapes
    }

    smile = (
        get_score(blendshape_dict, "mouthSmileLeft")
        + get_score(blendshape_dict, "mouthSmileRight")
    ) / 2
    jaw_open = get_score(blendshape_dict, "jawOpen")
    blink = (
        get_score(blendshape_dict, "eyeBlinkLeft")
        + get_score(blendshape_dict, "eyeBlinkRight")
    ) / 2

    if blink > 0.45:
        expression = "眨眼 / 闭眼"
    elif jaw_open > 0.45:
        expression = "张嘴"
    elif smile > 0.35:
        expression = "微笑"
    else:
        expression = "中性 / 未明显表情"

    return f"表情：{expression}  微笑 {smile:.2f}  张嘴 {jaw_open:.2f}  眨眼 {blink:.2f}"


def apply_segmentation_overlay(frame, result):
    """
    可选：把人体分割遮罩叠加到画面上。

    segmentation_mask 的每个像素表示“这个位置属于人体”的概率。
    概率高的位置叠加蓝色，学生能直观看到模型认为哪里是人。
    """
    if result.segmentation_mask is None:
        return frame

    mask = result.segmentation_mask.numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    person_area = mask > 0.2

    overlay = frame.copy()
    overlay[person_area] = (180, 90, 20)

    return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)


def draw_holistic_result(frame, result):
    """
    把 Holistic 的四类结果画到同一张画面上：
        1. 身体 pose
        2. 人脸 face
        3. 左手 left hand
        4. 右手 right hand
    """
    frame = apply_segmentation_overlay(frame, result)

    pose_count = draw_pose(frame, result.pose_landmarks)
    face_count = draw_face(frame, result.face_landmarks)
    left_hand_count = draw_hand(frame, result.left_hand_landmarks, "Left hand", (255, 120, 0))
    right_hand_count = draw_hand(frame, result.right_hand_landmarks, "Right hand", (0, 180, 255))

    info_lines = [
        "Holistic Landmarker 全身综合识别",
        f"身体关键点：{pose_count} / 33",
        f"人脸关键点：{face_count} / 468",
        f"左手关键点：{left_hand_count} / 21",
        f"右手关键点：{right_hand_count} / 21",
        summarize_face_expression(result.face_blendshapes),
    ]

    return draw_chinese_lines(frame, info_lines)


def create_holistic_landmarker():
    """
    创建 Holistic Landmarker。

    这些置信度参数分别控制脸、身体和手：
        min_face_detection_confidence：找到脸的最低分数
        min_pose_detection_confidence：找到身体的最低分数
        min_hand_landmarks_confidence：手部关键点可信的最低分数

    output_face_blendshapes=True：
        让模型额外输出表情动作分数，便于课堂观察。

    output_segmentation_mask：
        是否输出人体分割遮罩。默认关闭，速度更快。
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在：{MODEL_PATH}\n"
            "请确认 holistic_landmarker.task 和 mp_holistic.py 在同一文件夹。"
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=CONFIDENCE,
        min_face_landmarks_confidence=CONFIDENCE,
        min_pose_detection_confidence=CONFIDENCE,
        min_pose_landmarks_confidence=CONFIDENCE,
        min_hand_landmarks_confidence=CONFIDENCE,
        output_face_blendshapes=True,
        output_segmentation_mask=ENABLE_SEGMENTATION_MASK,
    )

    return HolisticLandmarker.create_from_options(options)


def next_timestamp_ms(last_timestamp_ms):
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms


def main():
    detector = create_holistic_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("无法打开摄像头。请检查摄像头编号、系统相机权限，或是否被其它软件占用。")
        detector.close()
        return

    print("Holistic 全身综合识别已启动。按 q 退出。")
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
            result = detector.detect_for_video(mp_image, last_timestamp_ms)

            frame = draw_holistic_result(frame, result)

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

            cv2.imshow("MediaPipe Holistic Landmarker - tutorial", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
