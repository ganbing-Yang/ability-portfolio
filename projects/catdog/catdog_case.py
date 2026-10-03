"""猫狗二分类。改编自老师的 cifar10_cnn_teaching_case.ipynb。

保留 RGB 归一化、卷积/池化、BatchNormalization、Dropout、训练曲线、
混淆矩阵和错误样本步骤。改为二分类，并加入 MLP 和增强消融对照。
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
from pathlib import Path
import argparse
import json
import tarfile
import pickle
import time
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED=42


def load_data(archive=None):
    if archive is None:
        (images,labels),(test_images,test_labels)=keras.datasets.cifar10.load_data()
        labels,test_labels=labels.reshape(-1),test_labels.reshape(-1)
    else:
        # 官方 Python 数据包包含 pickle，仅加载官方来源并校验 MD5 后使用。
        p=Path(archive)
        with p.open('rb') as f:
            actual=hashlib.file_digest(f,'md5').hexdigest()
        if actual!='c58f30108f718f92721af3b95e74349a':
            raise ValueError('CIFAR-10 数据包校验失败。请使用官方 Python 版数据包。')
        with tarfile.open(p,'r:gz') as tar:
            def batch(name):
                item=pickle.load(tar.extractfile('cifar-10-batches-py/'+name),encoding='bytes')
                x=item[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
                return x,np.asarray(item[b'labels'])
            batches=[batch('data_batch_'+str(i)) for i in range(1,6)]
            images=np.concatenate([b[0] for b in batches]);labels=np.concatenate([b[1] for b in batches])
            test_images,test_labels=batch('test_batch')
    def select(x,y):
        mask=np.isin(y,[3,5]);return x[mask],(y[mask]==5).astype('int32')
    images,labels=select(images,labels);test_images,test_labels=select(test_images,test_labels)
    # 固定且分层划分。官方测试集从头到尾单独保留，不进入早停和增强选型。
    indices=np.arange(len(labels))
    train_idx,val_idx=train_test_split(indices,test_size=.2,stratify=labels,random_state=SEED)
    return images,labels,test_images,test_labels,train_idx,val_idx


def build_model(kind):
    keras.utils.set_random_seed(SEED)
    if kind=='MLP':
        model=keras.Sequential([layers.Input((32,32,3)),layers.Flatten(),
                               layers.Dense(128,activation='relu'),layers.Dropout(.3),
                               layers.Dense(1,activation='sigmoid')],name='MLP')
    else:
        inputs=keras.Input((32,32,3));x=inputs
        if kind=='CNN_aug':
            x=layers.RandomFlip('horizontal',seed=SEED)(x)
            x=layers.RandomRotation(.06,seed=SEED+1)(x)
            x=layers.RandomZoom(.08,seed=SEED+2)(x)
        for n,filters in enumerate([16,32,64]):
            x=layers.Conv2D(filters,3,padding='same',use_bias=False,name=f'conv_{n+1}')(x)
            x=layers.BatchNormalization()(x);x=layers.Activation('relu')(x)
            x=layers.MaxPooling2D()(x);x=layers.Dropout(.2)(x)
        x=layers.Flatten()(x);x=layers.Dense(64,activation='relu')(x)
        x=layers.Dropout(.3)(x);outputs=layers.Dense(1,activation='sigmoid')(x)
        model=keras.Model(inputs,outputs,name=kind)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),loss='binary_crossentropy',metrics=['accuracy'])
    return model


def dataset(x,y,training=False):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    if training:ds=ds.shuffle(len(y),seed=SEED,reshuffle_each_iteration=True)
    options=tf.data.Options();options.threading.private_threadpool_size=2
    return ds.batch(128).with_options(options).prefetch(1)


def metrics(y,prob):
    pred=(prob>=.5).astype('int32')
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average=None,labels=[0,1],zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1.mean()),'cat_precision':float(precision[0]),'cat_recall':float(recall[0]),
            'dog_precision':float(precision[1]),'dog_recall':float(recall[1]),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def main(archive,output,epochs=20):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    raw,y,raw_test,ytest,ti,vi=load_data(archive)
    x=raw.astype('float32')/255;x_test=raw_test.astype('float32')/255
    assert set(ti).isdisjoint(vi)
    np.savez(out/'split_indices.npz',train=ti,validation=vi)
    histories,models,summary={},{},{}
    for kind in ['MLP','CNN','CNN_aug']:
        keras.backend.clear_session();model=build_model(kind)
        start=time.perf_counter()
        history=model.fit(dataset(x[ti],y[ti],True),validation_data=dataset(x[vi],y[vi]),
                          epochs=epochs,verbose=2,
                          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)])
        val_prob=model.predict(dataset(x[vi],y[vi]),verbose=0).reshape(-1)
        entry={'validation':metrics(y[vi],val_prob),'best_epoch':int(np.argmin(history.history['val_loss'])+1),
               'epochs_run':len(history.history['loss']),'parameters':int(model.count_params()),
               'training_seconds':time.perf_counter()-start}
        models[kind]=model;histories[kind]=history.history;summary[kind]=entry
        (out/(kind+'_history.json')).write_text(json.dumps(history.history,indent=2))
        print(kind,'validation',entry['validation']['accuracy'],flush=True)
    # 选型看验证集交叉熵，与早停的监控指标保持一致；测试集随后一次性评估。
    chosen=min(summary,key=lambda k:min(histories[k]['val_loss']))
    probabilities={}
    for kind,model in models.items():
        prob=model.predict(dataset(x_test,ytest),verbose=0).reshape(-1)
        probabilities[kind]=prob;summary[kind]['test']=metrics(ytest,prob)
    model=models[chosen];model.save(out/'catdog_selected.keras')
    prob=probabilities[chosen];pred=(prob>=.5).astype(int)
    import csv
    with (out/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['test_index','true_label','predicted_label','p_dog','correct'])
        w.writerows((i,int(a),int(b),float(p),int(a==b)) for i,(a,b,p) in enumerate(zip(ytest,pred,prob)))
    cm=confusion_matrix(ytest,pred,labels=[0,1])
    def save(fig,name):fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(11,9))
    for i,kind in enumerate(histories):
        h=histories[kind];axs[i,0].plot(h['loss'],label='Train');axs[i,0].plot(h['val_loss'],label='Validation')
        axs[i,0].set(title=kind+' loss',xlabel='Epoch',ylabel='Binary cross-entropy');axs[i,0].legend()
        axs[i,1].plot(h['accuracy'],label='Train');axs[i,1].plot(h['val_accuracy'],label='Validation')
        axs[i,1].set(title=kind+' accuracy',xlabel='Epoch',ylabel='Accuracy');axs[i,1].legend()
    fig.tight_layout();save(fig,'01_learning_curves')
    fig,axs=plt.subplots(1,2,figsize=(10,4));names=list(summary)
    axs[0].bar(names,[summary[k]['test']['accuracy'] for k in names],color=['#888','#426B89','#648B78'])
    axs[0].set(ylim=(0,1),title='Frozen models: test accuracy')
    for i,k in enumerate(names):axs[0].text(i,summary[k]['test']['accuracy']+.02,f"{summary[k]['test']['accuracy']:.1%}",ha='center')
    axs[1].imshow(cm,cmap='Blues');axs[1].set_xticks([0,1],['Cat','Dog']);axs[1].set_yticks([0,1],['Cat','Dog'])
    axs[1].set(xlabel='Predicted',ylabel='Actual',title=chosen+': confusion matrix')
    for i in range(2):
        for j in range(2):axs[1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=14)
    fig.tight_layout();save(fig,'02_metrics')
    wrong=np.flatnonzero(ytest!=pred);certainty=np.maximum(prob,1-prob)
    order=wrong[np.argsort(certainty[wrong])[::-1]]
    fig,axs=plt.subplots(3,4,figsize=(10,7))
    for ax,idx in zip(axs.flat,order[:12]):
        ax.imshow(raw_test[idx]);ax.set_title(f"#{idx} {'Cat' if ytest[idx]==0 else 'Dog'} -> {'Cat' if pred[idx]==0 else 'Dog'}\np(dog)={prob[idx]:.3f}",fontsize=9);ax.axis('off')
    fig.suptitle('Highest-confidence mistakes');fig.tight_layout();save(fig,'03_wrong_examples')
    fig,ax=plt.subplots(figsize=(8,3.7))
    ax.hist(prob[ytest==0],bins=25,alpha=.6,label='True cat');ax.hist(prob[ytest==1],bins=25,alpha=.6,label='True dog')
    ax.axvline(.5,color='#888',ls='--');ax.set(xlabel='Sigmoid score for dog',ylabel='Count',title='Score distributions (uncalibrated)');ax.legend()
    save(fig,'04_scores')
    fig,axs=plt.subplots(3,6,figsize=(10,5))
    rng=np.random.default_rng(SEED)
    for ax,idx in zip(axs.flat,rng.choice(len(raw_test),18,replace=False)):
        ax.imshow(raw_test[idx]);ax.set_title('Cat' if ytest[idx]==0 else 'Dog');ax.axis('off')
    fig.tight_layout();save(fig,'05_samples')
    result={'dataset':'CIFAR-10 cat/dog subset','source_url':'https://www.cs.toronto.edu/~kriz/cifar.html',
            'seed':SEED,'labels':{'cat':0,'dog':1},'split':{'train':len(ti),'validation':len(vi),'test':len(ytest)},
            'counts':{'train':np.bincount(y[ti]).tolist(),'validation':np.bincount(y[vi]).tolist(),'test':np.bincount(ytest).tolist()},
            'selected_model':chosen,'selection_metric':'minimum validation binary cross-entropy',
            'models':summary,'wrong_count':len(wrong),'worst_error_indices':order[:12].tolist(),
            'warning':'Scores are not calibrated probabilities. Only cat/dog images evaluated; no out-of-distribution detection.',
            'tensorflow_version':tf.__version__}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path);parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results');parser.add_argument('--epochs',type=int,default=20)
    a=parser.parse_args();main(a.archive,a.output,a.epochs)
