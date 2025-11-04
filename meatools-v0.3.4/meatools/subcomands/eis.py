#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os,json
from glob import glob
from datetime import datetime
import matplotlib.pylab as plt
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

# def find_and_sort_load_dta_files(root_folder='.'):
#     files = glob(os.path.join(root_folder, '**', '*.DTA'), recursive=True)
#     file_info = [(os.path.getmtime(f), f) for f in files]
#     file_info.sort(reverse=False)
#     return file_info

def find_and_sort_load_dta_files(root_folder='.', candidates=('EIS','PEIS','GEIS')):
    root = Path(root_folder)
    files = []
    for folder in candidates:
        target_dirs = list(root.rglob(folder))  # 找到叫这些名字的目录
        for d in target_dirs:
            files.extend(d.rglob("*.DTA"))
            files.extend(d.rglob("*.dta"))

    file_info = [(f.stat().st_mtime, str(f)) for f in files if f.stat().st_size > 0]
    file_info.sort(key=lambda x: x[0])
    return file_info


def read_dta_data(file_path):
    data=[]
    with open(file_path,'r',encoding='ISO-8859-1') as f:
        txt_line=f.readlines()
        i=0
        while i<len(txt_line):
            if txt_line[i].startswith('ZCURVE'):
                i+=3
                for j in range(i,len(txt_line),1):
                    line=txt_line[j].strip()
                    values=np.array([float(x) for x in line.split('\t') if x])
                    data.append(values)
                break
            else:
                i+=1
    return np.array(data)


def EIS_calc(data,index,file_path):#tol=50,freq_range=(-2,2)):
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.plot(data[:,3],-data[:,4],'o')
    plt.xlim(0, 0.1)
    plt.ylim(0, 0.2)
    plt.subplot(2,3,2)
    plt.plot(data[:,2],data[:,3],'o')
    plt.plot(data[:,2],-data[:,4],'x')
    plt.ylim(0, 0.2)
    plt.xscale('log')
    plt.grid(True, which='major', linestyle='-', alpha=0.8)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.title(file_path)

    freq,zreal,zimag=data[:,2],data[:,3],data[:,4]
    ### HFR calculation
    hfr_idx_pos=np.argmin(zimag[zimag>0]) 
    hfr_idx_neg=np.argmin(np.abs(zimag[zimag<0])) if np.any(zimag<0) else hfr_idx_pos
    hfr_idx_pos,hfr_idx_neg=np.where(zimag>0)[0][hfr_idx_pos],np.where(zimag<0)[0][hfr_idx_neg]
    hfr=(zreal[hfr_idx_pos]+zreal[hfr_idx_neg])/2
    
    ### R_ion calculation
    # mask=(freq>=1)&(freq<=4)
    # zreal_sel,zimag_sel=zreal[mask],zimag[mask]
    # diffirencials=[0]
    # for i in range(1,len(zreal_sel)):
    #     diff=(zimag[i]-zimag[i-1])/(zreal[i]-zreal[i-1])
    #     diffirencials.append(diff)
    # diffirencials=np.array(diffirencials)
    # slope_mask=np.where(np.abs(diffirencials[1:]-diffirencials[:-1])<tol)[0]+1
    # # print(slope_mask)
    # slope=np.mean(diffirencials[slope_mask])
    # zreal_,zimag_=zreal_sel[slope_mask[0]],zimag_sel[slope_mask[0]]
    # interp=(zreal_*slope-zimag_)/slope
    # r_ion=(interp-hfr)*3

    num_p=5
    p1_1 = np.zeros((5, len(data[:, 0]) - num_p))
    for q in range(len(data[:, 0]) - num_p):
        seq = slice(q, q + num_p + 1)
        x_data = data[seq, 3].reshape(-1, 1)  # Reshape for sklearn  zreal
        y_data = data[seq, 4]                 # zimag
        # Using Huber Regressor (robust to outliers)
        model = HuberRegressor()
        model.fit(x_data, y_data)
        p1_1[0, q] = model.intercept_
        p1_1[1, q] = model.coef_[0]
        p1_1[4, q] = np.min(data[seq, 2])

        y_pred = model.predict(x_data)
        p1_1[2, q] = r2_score(y_data, y_pred)
        p1_1[3, q] = -model.intercept_/model.coef_[0] # find the intercept at x axis
    # downselect focus values
    mask0=p1_1[2,:]>=0.999
    mask1=p1_1[2,:]>=np.quantile(p1_1[2,:],0.9)
    mask2=p1_1[4,:]<=20
    mask=(mask1 | mask0) & mask2
    # print(f"debug: {p1_1[3, mask]}")

    plt.subplot(2,3,4)
    plt.plot(p1_1[4, :],p1_1[0, :],'o')# y intercept
    plt.xscale('log')
    plt.subplot(2,3,5)
    plt.plot(p1_1[4, :],p1_1[3, :],'o', alpha=0.3)# x intercept
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(p1_1[4, mask],p1_1[3, mask],'o')
    plt.grid(True, which='major', linestyle='-', alpha=0.8)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.subplot(2,3,6)
    plt.plot(p1_1[4, :],p1_1[2, :],'o')
    plt.xscale('log')
    plt.savefig(f'results/eis/curve{index}.png')

    median=np.median(p1_1[3, mask])
    std=np.std(p1_1[3, mask])
    # print(median)
    r_ion=(median-hfr)*3
    r_ion_std=std*3
    sample_num=len(p1_1[2, mask])

    # ### Cdl calculation
    # Freq = np.logspace(freq_range[0],freq_range[1],int((freq_range[1]-freq_range[0])/0.1)+1)
    # try:
    #     f_interp=interp1d(freq,zimag,kind='linear',bounds_error=False,fill_value=np.nan)
    #     img_low=f_interp(Freq)
    # except Exception as e:
    #     print("There is something wrong with interpotation for Freq and Zimag")
    #     img_low=np.full_like(Freq,np.nan)
    # with np.errstate(divide='ignore',invalid='ignore'):
    #     Cdl=1.0/(img_low*Freq*2*np.pi)

    return hfr,r_ion,r_ion_std,sample_num

def main():
    os.makedirs('results/eis/',exist_ok=True)
    results={}
    file_info = find_and_sort_load_dta_files('.',candidates=('EIS','PEIS'))
    for i,(filetime,filepath) in enumerate(file_info):
        readable_time=datetime.fromtimestamp(filetime).strftime('%Y-%m-%d %H:%M:%S')
        data=read_dta_data(filepath)
        hfr,r_ion,r_ion_std,sample_num=EIS_calc(data,i,filepath)
        data_dump={"filename":filepath,
                   "filetime":readable_time,
                   "HFR (ohm)":hfr,
                   "R_ion (ohm)":r_ion,
                   "R_ion_std":r_ion_std,
                   "sample_number":sample_num
                }
        results[f"file_{i+1}"]=data_dump
    with open('results/eis/eis_results.json','w') as results_file:
            json.dump(results,results_file,ensure_ascii=False,indent=2,cls=NumpyEncoder)

if __name__=="__main__":
    # os.makedirs('logs',exist_ok=True)
    # log=open('logs/ecsa_normal.log','w')   
    main()
