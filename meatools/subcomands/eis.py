#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re,os,json
from scipy.stats import mstats
from glob import glob
from datetime import datetime
from pathlib import Path
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def find_and_sort_load_dta_files(root_folder='.'):
    files = glob(os.path.join(root_folder, '**', '*.DTA'), recursive=True)
    file_info = [(os.path.getmtime(f), f) for f in files]
    file_info.sort(reverse=False)
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


def EIS_calc(data,tol=50,freq_range=(-2,2)):
    freq,zreal,zimag=data[:,2],data[:,3],data[:,4]

    ### HFR calculation
    hfr_idx_pos=np.argmin(zimag[zimag>0]) 
    hfr_idx_neg=np.argmin(np.abs(zimag[zimag<0])) if np.any(zimag<0) else hfr_idx_pos
    hfr_idx_pos,hfr_idx_neg=np.where(zimag>0)[0][hfr_idx_pos],np.where(zimag<0)[0][hfr_idx_neg]
    hfr=(zreal[hfr_idx_pos]+zreal[hfr_idx_neg])/2
    
    ### R_ion calculation
    mask=(freq>=1)&(freq<=4)
    zreal_sel,zimag_sel=zreal[mask],zimag[mask]
    diffirencials=[0]
    for i in range(1,len(zreal_sel)):
        diff=(zimag[i]-zimag[i-1])/(zreal[i]-zreal[i-1])
        diffirencials.append(diff)
    diffirencials=np.array(diffirencials)
    slope_mask=np.where(np.abs(diffirencials[1:]-diffirencials[:-1])<tol)[0]+1
    # print(slope_mask)
    slope=np.mean(diffirencials[slope_mask])
    zreal_,zimag_=zreal_sel[slope_mask[0]],zimag_sel[slope_mask[0]]
    interp=(zreal_*slope-zimag_)/slope
    r_ion=(interp-hfr)*3

    ### Cdl calculation
    Freq = np.logspace(freq_range[0],freq_range[1],int((freq_range[1]-freq_range[0])/0.1)+1)
    try:
        f_interp=interp1d(freq,zimag,kind='linear',bounds_error=False,fill_value=np.nan)
        img_low=f_interp(Freq)
    except Exception as e:
        print("There is something wrong with interpotation for Freq and Zimag")
        img_low=np.full_like(Freq,np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        Cdl=1.0/(img_low*Freq*2*np.pi)

    return hfr,r_ion,Cdl

def main():
    os.makedirs('results/eis/',exist_ok=True)
    results={}
    file_info = find_and_sort_load_dta_files('EIS')
    for i,(filetime,filepath) in enumerate(file_info):
        readable_time=datetime.fromtimestamp(filetime).strftime('%Y-%m-%d %H:%M:%S')
        data=read_dta_data(filepath)
        hfr,r_ion,Cdl=EIS_calc(data)
        data_dump={"filename":filepath,
                   "filetime":readable_time,
                   "HFR (ohm)":hfr,
                   "R_ion (ohm)":r_ion,
                   "Cdl":Cdl
                }
        results[f"file_{i+1}"]=data_dump
    with open('results/eis/eis_results.json','w') as results_file:
            json.dump(results,results_file,indent=2,cls=NumpyEncoder)

if __name__=="__main__":
    # os.makedirs('logs',exist_ok=True)
    # log=open('logs/ecsa_normal.log','w')   
    main()