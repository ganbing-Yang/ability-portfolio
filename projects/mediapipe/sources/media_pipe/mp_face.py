"""
MediaPipe Tasks API 教程版：人脸关键点与简单表情识别

本示例做三件事：
    1. 用 Face Landmarker 找到人脸上的 468 个关键点。
    2. 把脸部轮廓、眼睛、眉毛、嘴巴、鼻子画出来。
    3. 读取 face blendshapes，用几个简单规则做课堂演示用的表情判断。

运行方式：
    1. 先把终端 / 命令提示符切换到本文件所在的文件夹。
       Windows 示例：cd 你的课程文件夹\\media_pipe
       macOS 示例：cd 你的课程文件夹/media_pipe

    2. 安装依赖库。可以直接安装到当前 Python，也可以安装到虚拟环境。
       Windows 常用命令：py -m pip install -r requirements.txt
       macOS 常用命令：python3 -m pip install -r requirements.txt

    3. 运行程序。
       Windows 常用命令：py mp_face.py
       macOS 常用命令：python3 mp_face.py

重要提醒：
    - 这个脚本的“微笑、眨眼、张嘴”等文字只是根据面部动作分数做的演示判断。
      它不是医学、心理学或身份识别系统，不能用来判断真实情绪。
    - OpenCV 自带的 cv2.putText 对中文支持不好，所以中文文字用 Pillow 画。
    - 如果中文显示失败，请确认系统有中文字体。Windows 通常有微软雅黑，macOS 通常有苹方。
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


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"

# =========================
# 1. 可修改配置
# =========================
# 初学者如果要调整程序，建议先只改这里。
# 这样可以把“配置”和“真正的识别逻辑”分开理解。
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 识别置信度阈值：
#   0.55 表示模型至少有 55% 把握时才接受结果。
#   光线差、脸离摄像头较远时可以降低到 0.4。
#   如果误识别太多，可以提高到 0.7。
CONFIDENCE = 0.55


# =========================
# 2. 常用人脸区域连接关系
# =========================
# Face Landmarker 会输出 468 个关键点。每个数字都是一个固定位置：
#   10、152 等点在脸部轮廓上
#   33、133 等点在左眼周围
#   61、291 等点在嘴唇周围
#
# 完整 468 点全部画出来会很密，不适合课堂观察。
# 所以这里挑出“脸部轮廓、眼睛、眉毛、嘴巴、鼻子”几组最容易理解的连接线。
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

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 61,
]

NOSE = [1, 2, 98, 327, 2, 168]


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
    """
    加载中文字体，并用缓存避免每一帧都重新读取字体文件。

    摄像头程序每秒可能处理 20~60 帧。
    如果每一帧都从硬盘加载字体，会明显拖慢画面。
    """
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)

    # 找不到中文字体时，退回默认字体。默认字体可能无法显示中文，但程序不会崩溃。
    return ImageFont.load_default()


def draw_chinese_lines(frame, lines, x=20, y=20, font_size=28, color=(255, 255, 0)):
    """
    用 Pillow 一次性绘制多行中文文字。

    OpenCV 的图像数组是 BGR 顺序，Pillow 使用 RGB 顺序。
    所以流程是：
        BGR(OpenCV) -> RGB(Pillow) -> 绘制文字 -> BGR(OpenCV)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = load_font(font_size)

    line_height = int(font_size * 1.35)
    for index, line in enumerate(lines):
        draw.text((x, y + index * line_height), line, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def normalized_landmarks_to_pixels(landmarks, width, height):
    """把 0~1 的归一化坐标转换成屏幕上的像素坐标。"""
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        points.append((x, y))

    return points


def draw_polyline(frame, points, indexes, color, thickness=2):
    """
    按照关键点编号画连续线条。

    indexes=[10, 338, 297] 表示：
        先连 10 -> 338，再连 338 -> 297。
    """
    for start_index, end_index in zip(indexes, indexes[1:]):
        if start_index < len(points) and end_index < len(points):
            cv2.line(frame, points[start_index], points[end_index], color, thickness)


def get_score(blendshape_dict, name):
    """
    安全读取某个 face blendshape 分数。

    blendshape 可以理解为“一个面部动作有多明显”：
        mouthSmileLeft 越大，左侧嘴角越像在笑
        jawOpen 越大，下巴越像张开
        eyeBlinkLeft 越大，左眼越像闭上
    """
    return blendshape_dict.get(name, 0.0)


def analyze_expression(blendshape_dict):
    """
    用简单规则把 blendshape 分数转成学生容易理解的文字。

    这不是训练出来的情绪分类器，只是课堂演示规则：
        如果眨眼分数高 -> 显示“眨眼 / 闭眼”
        如果张嘴高且眼睛睁大 -> 显示“惊讶 / 张嘴睁眼”
        如果嘴角上扬高 -> 显示“微笑”
    """
    smile = (
        get_score(blendshape_dict, "mouthSmileLeft")
        + get_score(blendshape_dict, "mouthSmileRight")
    ) / 2

    jaw_open = get_score(blendshape_dict, "jawOpen")

    eye_blink = (
        get_score(blendshape_dict, "eyeBlinkLeft")
        + get_score(blendshape_dict, "eyeBlinkRight")
    ) / 2

    eye_wide = (
        get_score(blendshape_dict, "eyeWideLeft")
        + get_score(blendshape_dict, "eyeWideRight")
    ) / 2

    brow_up = get_score(blendshape_dict, "browInnerUp")

    brow_down = (
        get_score(blendshape_dict, "browDownLeft")
        + get_score(blendshape_dict, "browDownRight")
    ) / 2

    mouth_frown = (
        get_score(blendshape_dict, "mouthFrownLeft")
        + get_score(blendshape_dict, "mouthFrownRight")
    ) / 2

    if eye_blink > 0.45:
        return "眨眼 / 闭眼"
    if jaw_open > 0.45 and eye_wide > 0.25:
        return "惊讶 / 张嘴睁眼"
    if smile > 0.35:
        return "微笑"
    if jaw_open > 0.45:
        return "张嘴"
    if brow_down > 0.35 and mouth_frown > 0.15:
        return "皱眉 / 不高兴"
    if brow_up > 0.35:
        return "挑眉"

    return "中性 / 未明显表情"


def draw_face_landmarks(frame, detection_result):
    """
    绘制人脸关键点，并返回用于显示的状态文字。
    """
    if not detection_result.face_landmarks:
        frame = draw_chinese_lines(frame, ["未检测到人脸：请面对摄像头并保持光线充足"], y=20)
        return frame

    height, width, _ = frame.shape

    for face_index, face_landmarks in enumerate(detection_result.face_landmarks):
        points = normalized_landmarks_to_pixels(face_landmarks, width, height)

        draw_polyline(frame, points, FACE_OVAL, (0, 255, 0), 2)
        draw_polyline(frame, points, LEFT_EYE, (255, 255, 0), 2)
        draw_polyline(frame, points, RIGHT_EYE, (255, 255, 0), 2)
        draw_polyline(frame, points, LEFT_EYEBROW, (255, 0, 255), 2)
        draw_polyline(frame, points, RIGHT_EYEBROW, (255, 0, 255), 2)
        draw_polyline(frame, points, LIPS_OUTER, (0, 0, 255), 2)
        draw_polyline(frame, points, NOSE, (255, 0, 0), 2)

        # 每隔 8 个点画一个小白点，既能让学生看到“很多关键点”，又不会把脸遮满。
        for point_index, (x, y) in enumerate(points):
            if point_index % 8 == 0:
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        info_lines = [f"人脸关键点：{len(points)} 个"]

        if detection_result.face_blendshapes and face_index < len(detection_result.face_blendshapes):
            blendshapes = detection_result.face_blendshapes[face_index]
            blendshape_dict = {
                category.category_name: category.score
                for category in blendshapes
            }

            expression_text = analyze_expression(blendshape_dict)
            smile = (
                get_score(blendshape_dict, "mouthSmileLeft")
                + get_score(blendshape_dict, "mouthSmileRight")
            ) / 2
            jaw_open = get_score(blendshape_dict, "jawOpen")
            blink = (
                get_score(blendshape_dict, "eyeBlinkLeft")
                + get_score(blendshape_dict, "eyeBlinkRight")
            ) / 2
            brow_up = get_score(blendshape_dict, "browInnerUp")

            info_lines.extend([
                f"课堂演示判断：{expression_text}",
                f"微笑分数：{smile:.2f}",
                f"张嘴分数：{jaw_open:.2f}",
                f"眨眼分数：{blink:.2f}",
                f"挑眉分数：{brow_up:.2f}",
            ])
        else:
            info_lines.append("未输出 blendshape：请确认 output_face_blendshapes=True")

        frame = draw_chinese_lines(frame, info_lines, y=20)

    return frame


def create_face_landmarker():
    """
    创建 Face Landmarker。

    output_face_blendshapes=True 是本示例的关键：
    如果不开启这个选项，程序只能画关键点，无法得到微笑、张嘴、眨眼等动作分数。
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=CONFIDENCE,
        min_face_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    return vision.FaceLandmarker.create_from_options(options)


def next_timestamp_ms(last_timestamp_ms):
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms


def main():
    detector = create_face_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("无法打开摄像头。请检查摄像头编号、系统相机权限，或是否被其它软件占用。")
        detector.close()
        return

    print("人脸关键点与表情识别已启动。按 q 退出。")
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

            frame = draw_face_landmarks(frame, detection_result)

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

            cv2.imshow("MediaPipe Face Landmarker - tutorial", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
