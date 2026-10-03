"""从已保存的单图关键点计算指尖距离，不据此判断实时手势。"""
from pathlib import Path
import json,math
from PIL import Image
ROOT=Path(__file__).parent

def measure(points,width,height):
    def distance(a,b):
        return math.hypot((points[a]['x']-points[b]['x'])*width,
                          (points[a]['y']-points[b]['y'])*height)
    tip=distance(4,8)
    palm=distance(0,9)
    if palm<1e-6:raise ValueError('手掌尺度过小，无法计算可靠比值')
    return {'thumb_index_pixels':tip,'wrist_middle_base_pixels':palm,
            'tip_distance_over_palm':tip/palm}

if __name__=='__main__':
    record=json.loads((ROOT/'results/summary.json').read_text())['cases']['hand']
    groups=record['landmarks']['hand_landmarks']
    with Image.open(ROOT/'inputs'/record['input']) as image:width,height=image.size
    measured=[measure(points,width,height) for points in groups]
    result={'image':record['input'],'image_size':[width,height],'hands':measured,
            'scope':'Static-image measurement only; no calibrated gesture threshold or live interaction.'}
    (ROOT/'results/hand_measurement.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
