#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re,os,json
from scipy import interpolate
from scipy.stats import mstats
from glob import glob
from datetime import datetime
from pathlib import Path

# plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def find_and_sort_load_dta_files(root_folder='.'):
    files = glob(os.path.join(root_folder, '**', '*.DTA'), recursive=True)

    
    # Create list of tuples (mtime, filepath)
    file_info = [(os.path.getmtime(f), f) for f in files]
    
    # Sort by mtime (first element of tuple)
    file_info.sort(reverse=False)
    return file_info

def find_cv_subfolders(root_dir):
    root_path = Path(root_dir)
    csv_folders = {str(p.parent.relative_to(root_path)) 
                  for p in root_path.glob('**/'+searchKey)}
    
    log.write("Subfolders containing CV DTA files:\n")
    csv_folders=sorted(csv_folders)
    # print(csv_folders)
    for folder in csv_folders:
        log.write(f"- {folder}\n")
    return csv_folders

def lsv_calc(u_subfolders):
    results={}
    for jk in range(len(u_subfolders)):
        file_path=u_subfolders[jk]
        file_info = find_and_sort_load_dta_files(file_path)
        plt.figure(jk+1)
        results[f"dir_{jk}"]={}
        for i, (mtime, filepath) in enumerate(file_info, 1):
            results[f"dir_{jk}"][str(i)]={}
            readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            results[f"dir_{jk}"][str(i)]["file"]=filepath
            results[f"dir_{jk}"][str(i)]["time_stamp"]=readable_time

            with open(filepath, 'r') as f:
                for _ in range(59):
                    next(f)
                content_from_line65 = f.read()

            u = re.split('CURVE', content_from_line65)
            plt.subplot(2,3,i)
            dump={}
            for j in range(min(5,len(u))):
                with open(os.path.join(os.getcwd(),'temp'), 'w') as fileID:
                    # print(os.path.join(os.getcwd(),'temp'))
                    fileID.write(u[j])
                A = np.loadtxt(os.path.join(os.getcwd(),'temp'), skiprows=2, usecols=range(8))
                plt.plot(A[:,2],A[:,3])
                if j>0 :
                    scanRate=np.median(np.abs(np.diff(A[:,2])/np.diff(A[:,1])))
                    # print('rate='+str(scanRate)+' V/s')
                    #print('Vmin='+str(np.min(A[:,2]))+' V')
                    CVall = A[:, [2, 3]]
                    startV = CVall[0, 0]
                    updData = CVall[:, :2]
                    mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                    double1 = updData[mask1, :]

                    vol=updData[:,0]
                    index=np.argmin(abs(vol-0.4))
                    H2cx3=updData[:,1][index]

                    
                    # Interpolation
                    x_new = np.arange(0.35, 0.55, 0.001)
                    f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
                    double1_interp = f1(x_new)
                    
                    H2cx = np.quantile(double1_interp,0.99)

                    # print(f"H2cx99%: {H2cx}")
                    n=len(double1[:,0])
                    m =(n*np.sum(double1[:,0]*double1[:,1]) - np.sum(double1[:,0])*np.sum(double1[:,1])) /(n*np.sum(double1[:,0]*double1[:,0]) - np.sum(double1[:,0])**2)
                    b = (np.sum(double1[:,1]) - m*np.sum(double1[:,0])) / n

                    H2cx2=m*0.8+b
                    

                    # print(f"slopeReg: {m}")
                    # print(f"H2cxReg: {H2cx2}")

                    voltage,cur=A[:,2],A[:,3]
                    lsv_membrane=cur[np.argmin(np.abs(voltage-0.4))]


                    dump[f"curve_{str(j)}"]={
                        "rate (V/s)":scanRate,
                        "H2cx99%":H2cx,
                        "slopeReg":m,
                        "H2cx_800mV":H2cx2,
                        "H2cx_0mV":b,
                        "H2cx_400mV":H2cx3,
                        "lsv_membrane*area":lsv_membrane
                    }

                    
                    #plt.title(f"dd: {ECA:.1f}")
                else:
                    
                    plt.title(f"{i}. {filepath}")
                #if os.path.exists('temp'):
                #    os.remove('temp')
                #

            results[f"dir_{jk}"][str(i)]["data"]=dump

    plt.savefig(f'results/lsv/lsv_results.png')
    return results

if __name__=="__main__":
    os.makedirs('logs',exist_ok=True)
    os.makedirs('results/lsv/',exist_ok=True)
    log=open('logs/lsv.log','w')   

    searchKey='LSV/**/*.DTA'
    ECAcutoff = 0.08
    u_subfolders=find_cv_subfolders('.')
    results=lsv_calc(u_subfolders)

    with open('results/lsv/lsv_results.json','w') as results_file:
            json.dump(results,results_file,indent=2,cls=NumpyEncoder)



