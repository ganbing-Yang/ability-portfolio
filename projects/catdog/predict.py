"""用已训练的猫狗模型预测一张照片：python predict.py --image example.jpg"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

def predict(image_path,model_path):
    model=keras.models.load_model(model_path)
    with Image.open(image_path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB').resize((32,32))
        x=np.asarray(im,dtype='float32')/255
    # 模型输出是二分类分数，不是经过校准的可靠概率。
    p=float(model(x[None,...],training=False).numpy()[0,0])
    label='狗' if p>=.5 else '猫'
    return {'预测类别':label,'狗分数':p,'猫分数':1-p,
            '适用范围':'仅对一张图片中的猫或狗进行二选一；其他物体也会被强行归类。'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=Path(__file__).parent/'results/catdog_selected.keras')
    a=ap.parse_args();print(predict(a.image,a.model))
