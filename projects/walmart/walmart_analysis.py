"""Walmart 周销售额预测。运行：python walmart_analysis.py --data Walmart_Sales.csv

保留课堂中的读数据、标准化、线性预测、MSE训练、评估和作图步骤。
这里用 NumPy 写出梯度下降，便于看清权重怎样更新；不依赖 GPU。
"""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_plot(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_features(df, train_mask, mode):
    numeric = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year']
    category = []
    if mode == 'numeric_store':
        numeric += ['Store', 'Month', 'Week']
    elif mode == 'store_encoded':
        numeric += ['Month', 'Week']
        category = ['Store']
    else:
        numeric += ['Week_sin', 'Week_cos']
        category = ['Store', 'Month']
    scaler = StandardScaler().fit(df.loc[train_mask, numeric])
    x = scaler.transform(df[numeric])
    names = numeric.copy()
    encoder = None
    if category:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoder.fit(df.loc[train_mask, category])
        x = np.column_stack([x, encoder.transform(df[category])])
        names += list(encoder.get_feature_names_out(category))
    return np.column_stack([np.ones(len(df)), x]), ['Intercept'] + names, scaler, encoder


def train_linear(x_train, y_train, x_val, y_val, epochs=8000):
    # y 同样只按训练集均值和标准差缩放，最终预测还原到原始销售额。
    y_mean, y_scale = float(y_train.mean()), float(y_train.std())
    yt, yv = (y_train-y_mean)/y_scale, (y_val-y_mean)/y_scale
    weights = np.zeros(x_train.shape[1])
    gram = x_train.T @ x_train / len(yt)
    rhs = x_train.T @ yt / len(yt)
    lr = 0.45 / np.linalg.eigvalsh(gram).max()
    best_loss, best_weights, best_epoch = np.inf, None, 0
    history = []
    for epoch in range(epochs):
        weights -= lr * 2 * (gram @ weights - rhs)
        if epoch % 10 == 0 or epoch == epochs-1:
            train_loss = float(np.mean((x_train@weights-yt)**2))
            val_loss = float(np.mean((x_val@weights-yv)**2))
            history.append([epoch+1, train_loss, val_loss])
            if val_loss < best_loss:
                best_loss, best_weights, best_epoch = val_loss, weights.copy(), epoch+1
    return best_weights, y_mean, y_scale, np.asarray(history), best_epoch, float(lr)


def scores(true, pred):
    mse = float(mean_squared_error(true, pred))
    return {'MAE': float(mean_absolute_error(true,pred)), 'MSE': mse,
            'RMSE': float(np.sqrt(mse)), 'R2': float(r2_score(true,pred)),
            'WAPE': float(np.abs(true-pred).sum()/np.abs(true).sum())}


def main(data_path, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    figdir = out / 'figures'; figdir.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    audit = {'rows':len(df), 'missing':df.isna().sum().to_dict(),
             'duplicate_rows':int(df.duplicated().sum()),
             'duplicate_store_date':int(df.duplicated(['Store','Date']).sum())}
    if any(audit['missing'].values()) or audit['duplicate_store_date']:
        raise ValueError('源数据存在缺失或重复门店-日期，需要先核对。')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='raise')
    df = df.sort_values(['Date','Store']).reset_index(drop=True)
    df['Year'],df['Month'] = df.Date.dt.year,df.Date.dt.month
    df['Week'] = df.Date.dt.isocalendar().week.astype(int)
    df['Week_sin'] = np.sin(2*np.pi*df.Week/52.18)
    df['Week_cos'] = np.cos(2*np.pi*df.Week/52.18)
    dates = np.sort(df.Date.unique())
    a,b = int(len(dates)*.60),int(len(dates)*.80)
    train = df.Date.isin(dates[:a]).to_numpy()
    val = df.Date.isin(dates[a:b]).to_numpy()
    test = df.Date.isin(dates[b:]).to_numpy()
    y = df.Weekly_Sales.to_numpy()
    split = {}
    for name,mask in [('train',train),('validation',val),('test',test)]:
        split[name] = {'rows':int(mask.sum()),'weeks':int(df.loc[mask,'Date'].nunique()),
                       'start':str(df.loc[mask,'Date'].min().date()),
                       'end':str(df.loc[mask,'Date'].max().date())}
    assert set(dates[:a]).isdisjoint(dates[a:b]) and set(dates[a:b]).isdisjoint(dates[b:])
    candidates = {}
    for mode in ['numeric_store','store_encoded','store_month_encoded']:
        x,names,scaler,encoder = make_features(df,train,mode)
        weights,ym,ys,history,best,lr = train_linear(x[train],y[train],x[val],y[val])
        pred = x@weights*ys+ym
        candidates[mode] = {'x':x,'pred':pred,'weights':weights,'names':names,
                            'history':history,'y_mean':ym,'y_scale':ys,'best_epoch':best,'lr':lr,
                            'scaler':scaler,'encoder':encoder,
                            'val_scores':scores(y[val],pred[val])}
    selected = min(candidates, key=lambda m:candidates[m]['val_scores']['MSE'])
    chosen = candidates[selected]
    # 测试集在模型和停止轮次确定以后才用于报告最终指标。
    for c in candidates.values(): c['test_scores'] = scores(y[test],c['pred'][test])
    store_mean = df.loc[train].groupby('Store').Weekly_Sales.mean()
    base_pred = df.Store.map(store_mean).to_numpy()
    baseline_scores = scores(y[test],base_pred[test])
    prediction_table = df[['Store','Date','Weekly_Sales','Holiday_Flag']].copy()
    prediction_table['Split'] = np.select([train,val,test],['train','validation','test'],default='')
    prediction_table['Predicted_Sales'] = chosen['pred']
    prediction_table['Residual'] = y-chosen['pred']
    prediction_table['Absolute_Error'] = np.abs(prediction_table.Residual)
    prediction_table.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    prediction_table.loc[test].to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(chosen['history'],columns=['epoch','train_mse_scaled','validation_mse_scaled']).to_csv(out/'training_history.csv',index=False)
    weights_in_sales_units = chosen['weights']*chosen['y_scale']
    weights_in_sales_units[0] += chosen['y_mean']
    pd.DataFrame({'feature':chosen['names'],'weight':weights_in_sales_units}).to_csv(out/'coefficients.csv',index=False)
    np.savez(out/'linear_model.npz',weights=chosen['weights'],y_mean=chosen['y_mean'],y_scale=chosen['y_scale'])
    # 下面的图是全数据描述性分析；模型拟合、标准化和选型均仅使用对应训练/验证区间。
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    vars_ = ['Weekly_Sales','Temperature','Fuel_Price','CPI','Unemployment','Holiday_Flag']
    fig,axs = plt.subplots(2,3,figsize=(11,6))
    for col,ax in zip(vars_,axs.flat):
        ax.hist(df[col]/(1e6 if col=='Weekly_Sales' else 1),bins=30,color='#426B89',alpha=.9)
        ax.set_title(col+(' (million)' if col=='Weekly_Sales' else ''));ax.set_ylabel('Count')
    fig.tight_layout();save_plot(fig,figdir,'01_distributions')
    corr = df[['Weekly_Sales','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment']].corr()
    fig,ax=plt.subplots(figsize=(7,5.5));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(corr)),corr.columns,rotation=45,ha='right');ax.set_yticks(range(len(corr)),corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,shrink=.75);fig.tight_layout();save_plot(fig,figdir,'02_correlations')
    h=chosen['history'];fig,ax=plt.subplots(figsize=(9,3.6))
    ax.plot(h[:,0],h[:,1],label='Train');ax.plot(h[:,0],h[:,2],label='Validation')
    ax.axvline(chosen['best_epoch'],color='#888',ls='--',label='Selected epoch')
    ax.set(xlabel='Gradient updates',ylabel='MSE (standardized sales)',title='Linear regression training');ax.legend()
    save_plot(fig,figdir,'03_loss')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].scatter(y[test]/1e6,chosen['pred'][test]/1e6,s=10,alpha=.45)
    limits=[min(y[test].min(),chosen['pred'][test].min())/1e6,max(y[test].max(),chosen['pred'][test].max())/1e6]
    axs[0].plot(limits,limits,'--',color='#888');axs[0].set(xlabel='Actual sales (million)',ylabel='Predicted sales (million)',title='All test stores')
    one=prediction_table.loc[test & (df.Store==1)]
    axs[1].plot(one.Date,one.Weekly_Sales/1e6,label='Actual');axs[1].plot(one.Date,one.Predicted_Sales/1e6,label='Predicted')
    axs[1].set(title='Store 1: held-out weeks',ylabel='Weekly sales (million)');axs[1].tick_params(axis='x',rotation=30);axs[1].legend()
    fig.tight_layout();save_plot(fig,figdir,'04_actual_prediction')
    residual=y[test]-chosen['pred'][test];fig,axs=plt.subplots(1,2,figsize=(11,3.8))
    axs[0].hist(residual/1e3,bins=40,color='#426B89');axs[0].axvline(0,color='#888',ls='--');axs[0].set(xlabel='Actual - predicted (thousand)',ylabel='Count',title='Test residual distribution')
    axs[1].scatter(chosen['pred'][test]/1e6,residual/1e3,s=10,alpha=.4);axs[1].axhline(0,color='#888',ls='--');axs[1].set(xlabel='Predicted sales (million)',ylabel='Residual (thousand)',title='Residual vs fitted value')
    fig.tight_layout();save_plot(fig,figdir,'05_residuals')
    coefs=pd.Series(chosen['weights'][1:]*chosen['y_scale'],index=chosen['names'][1:]);top=coefs.abs().nlargest(14).index
    numeric = [n for n in chosen['names'][1:] if not n.startswith(('Store_','Month_'))]
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    coefs[top].sort_values().div(1000).plot.barh(ax=axs[0],color='#426B89');axs[0].set(title='Largest coefficients',xlabel='Sales change (thousand)')
    coefs[numeric].sort_values().div(1000).plot.barh(ax=axs[1],color='#648B78');axs[1].set(title='Standardized numeric coefficients',xlabel='Sales change per training SD (thousand)')
    fig.tight_layout();save_plot(fig,figdir,'06_weights')
    holiday=df.groupby('Holiday_Flag').Weekly_Sales.agg(['count','mean','median'])
    stores=df.groupby('Store').Weekly_Sales.mean().sort_values()
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].bar(['Non-holiday','Holiday'],holiday['mean']/1e6,color=['#426B89','#9A7151']);axs[0].set(ylabel='Mean weekly sales (million)',title='Holiday comparison (descriptive)')
    axs[1].bar(stores.index.astype(str),stores.values/1e6,color='#426B89');axs[1].set(ylabel='Mean weekly sales (million)',xlabel='Store ID, ordered by sales',title='Differences across stores');axs[1].tick_params(axis='x',labelsize=6,rotation=90)
    fig.tight_layout();save_plot(fig,figdir,'07_holiday_store')
    summary={'data_audit':audit,'date_start':str(df.Date.min().date()),'date_end':str(df.Date.max().date()),
             'stores':int(df.Store.nunique()),'weeks':len(dates),'split':split,'selected_model':selected,
             'models':{k:{'validation':v['val_scores'],'test':v['test_scores'],'best_epoch':v['best_epoch'],'learning_rate':v['lr']} for k,v in candidates.items()},
             'selected_train_metrics':scores(y[train],chosen['pred'][train]),'store_mean_baseline_test':baseline_scores,
             'sales_describe':df.Weekly_Sales.describe().to_dict(),'sales_skew':float(df.Weekly_Sales.skew()),
             'holiday_stats':{str(k):{n:float(v) for n,v in row.items()} for k,row in holiday.to_dict('index').items()},
             'store_min':{'id':int(stores.index[0]),'mean':float(stores.iloc[0])},'store_max':{'id':int(stores.index[-1]),'mean':float(stores.iloc[-1])},
             'correlation_with_sales':corr.Weekly_Sales.to_dict(),
             'residual_mean':float(residual.mean()),'test_negative_predictions':int((chosen['pred'][test]<0).sum()),
             'worst_test_rows':prediction_table.loc[test].nlargest(5,'Absolute_Error').assign(Date=lambda x:x.Date.dt.strftime('%Y-%m-%d')).to_dict('records'),
             'feature_names':chosen['names'],'feature_weights':weights_in_sales_units.tolist(),
             'source':'User supplied Walmart_Sales.csv; assignment specifies Kaggle mikhail1681/Walmart Sales'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected':selected,'test':summary['models'][selected]['test'],'baseline':baseline_scores,'split':split},ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path(__file__).parent/'Walmart_Sales.csv');parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args();main(args.data,args.output)
