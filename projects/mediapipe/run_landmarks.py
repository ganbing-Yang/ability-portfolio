"""基于教师MediaPipe例题的单图核对，不进行训练或实时帧率测试。"""
from pathlib import Path
import argparse,json,importlib.util,time
import numpy as np
import cv2
ROOT=Path(__file__).parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def save_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2))

def landmarks_case(out,image_path=None):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    out.mkdir(parents=True,exist_ok=True)
    classes=[('hand','hand.jpg',vision.HandLandmarker,vision.HandLandmarkerOptions,'hand_landmarker.task','hand_landmarks'),
             ('face','face.jpg',vision.FaceLandmarker,vision.FaceLandmarkerOptions,'face_landmarker.task','face_landmarks'),
             ('pose','pose.jpg',vision.PoseLandmarker,vision.PoseLandmarkerOptions,'pose_landmarker.task','pose_landmarks'),
             ('holistic','pose.jpg',vision.HolisticLandmarker,vision.HolisticLandmarkerOptions,'holistic_landmarker.task',None)]
    records={}
    for kind,default,detector_type,options_type,model_name,field in classes:
        path=Path(image_path) if image_path else ROOT/'inputs'/default
        image=cv2.imread(str(path))
        if image is None:raise FileNotFoundError(path)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        settings={'base_options':python.BaseOptions(model_asset_path=str(ROOT/'sources/media_pipe'/model_name),delegate=python.BaseOptions.Delegate.CPU),
                 'running_mode':vision.RunningMode.IMAGE}
        if kind=='hand':settings.update(num_hands=2,min_hand_detection_confidence=.55)
        if kind=='face':settings.update(output_face_blendshapes=True,min_face_detection_confidence=.55)
        if kind=='pose':settings.update(min_pose_detection_confidence=.55)
        with detector_type.create_from_options(options_type(**settings)) as detector:
            start=time.perf_counter();detection=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb));elapsed=time.perf_counter()-start
        if field:
            groups=getattr(detection,field)
            group_fields={field:groups}
        else:
            group_fields={}
            for attr in ['pose_landmarks','face_landmarks','left_hand_landmarks','right_hand_landmarks']:
                points=getattr(detection,attr,[])
                group_fields[attr]=[points] if points else []
        annotated=image.copy();h,w=image.shape[:2];serialized={};counts={}
        for attr,groups in group_fields.items():
            counts[attr]=[len(g) for g in groups]
            serialized[attr]=[[{'x':float(p.x),'y':float(p.y),'z':float(p.z)} for p in g] for g in groups]
            for g in groups:
                pixels=[(int(np.clip(p.x*w,0,w-1)),int(np.clip(p.y*h,0,h-1))) for p in g]
                assert all(np.isfinite([p.x,p.y,p.z]).all() for p in g)
                if 'hand' in attr:
                    t=module('teacher_hand',ROOT/'sources/media_pipe/mp_hand.py');edges=t.HAND_CONNECTIONS
                elif 'pose' in attr:
                    t=module('teacher_body',ROOT/'sources/media_pipe/mp_body.py');edges=t.POSE_CONNECTIONS
                else:edges=[]
                for a,b in edges:
                    cv2.line(annotated,pixels[a],pixels[b],(0,210,0),2)
                for pt in pixels:cv2.circle(annotated,pt,2,(0,0,255),-1)
        cv2.imwrite(str(out/(kind+'.jpg')),annotated)
        records[kind]={'input':path.name,'groups':counts,'seconds':elapsed,'landmarks':serialized}
        if kind=='hand':
            records[kind]['handedness']=[[{'label':c.category_name,'score':float(c.score)} for c in categories] for categories in detection.handedness]
        if kind=='face':
            blend=[[{'name':c.category_name,'score':float(c.score)} for c in cats] for cats in detection.face_blendshapes]
            records[kind]['blendshapes']=blend
        print('MediaPipe',kind,counts,flush=True)
    save_json(out/'summary.json',{'mediapipe':mp.__version__,'mode':'IMAGE','cases':records,
       'scope':'Static images only. No webcam, live gesture recognition, or real-time frame rate validation.'})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--image",type=Path)
    args=p.parse_args();landmarks_case(ROOT/"results",args.image)
