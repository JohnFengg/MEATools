#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re,os,json
import os
from scipy import interpolate
from scipy.stats import mstats
from glob import glob
from datetime import datetime
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def find_and_sort_load_dta_files(root_folder='.'):
    files = glob(os.path.join(root_folder, '**', '*cv*.DTA'), recursive=True)
    file_info = [(os.path.getmtime(f), f) for f in files]
    file_info.sort(reverse=False)
    return file_info

def find_cv_subfolders(root_dir):
    root_path = Path(root_dir)
    csv_folders = {str(p.parent.relative_to(root_path)) 
                  for p in root_path.glob('**/'+searchKey)}    
    log.write("Subfolders containing CV DTA files:\n")
    csv_folders=sorted(csv_folders)
    for folder in csv_folders:
        log.write(f"- {folder}\n" if folder != '.' else "- (root directory)\n")
    return csv_folders

def plot_COtripping(file_info):
    data_dump={}
    for i, (mtime, filepath) in enumerate(file_info, 1):
        dump={}
        readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        dump["time_stamp"]=readable_time
        dump["file"]=filepath

        with open(filepath, 'r') as f:
            for _ in range(65):
                next(f)
            content_from_line65 = f.read()

        u = re.split('CURVE', content_from_line65)
        oldUpper = x_CO*0
        
        for j in range(len(u)):
            with open('temp', 'w') as fileID:
                fileID.write(u[j])
            A = np.loadtxt('temp', skiprows=2, usecols=range(8))
            plt.subplot(2,3,j+1)
            plt.plot(A[:,2],A[:,3])
            if j>0 and j<(len(u)-1):
                scanRate=np.median(np.abs(np.diff(A[:,2])/np.diff(A[:,1])))
                # print('rate='+str(scanRate)+' V/s')
                # print('Vmin='+str(np.min(A[:,2]))+' V'

                CVall = A[:, [2, 3]]
                startV = CVall[0, 0]
                updData = CVall[:, :2]
                mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                double1 = updData[mask1, :]
                mask2 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (np.concatenate(([0], np.diff(updData[:, 0]))) < 0)
                double2 = updData[mask2, :]

                double1 = double1[np.argsort(double1[:, 0])]
                double2 = double2[np.argsort(double2[:, 0])]
                double1 = double1[np.concatenate(([True], np.diff(double1[:, 0]) != 0)), :]
                double2 = double2[np.concatenate(([True], np.diff(double2[:, 0]) != 0)), :]

                # CO stripping area
                maskCO = (updData[:, 0] > 0.5) & (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                topLimits = updData[maskCO, :]
                fCO = interpolate.interp1d(topLimits[:, 0], topLimits[:, 1], bounds_error=False)
                newUpper = fCO(x_CO)
                plt.subplot(2,3,1)
                plt.plot(x_CO,newUpper)
                oldUpper =np.maximum(oldUpper,newUpper)

                # Interpolation
                x_new = np.arange(0.35, 0.451, 0.001)
                f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
                double1_interp = f1(x_new)
                f2 = interpolate.interp1d(double2[:, 0], double2[:, 1], bounds_error=False)
                double2_interp = f2(x_new)

                ddouble = np.abs(double1_interp - double2_interp)
                ddouble = ddouble[~np.isnan(ddouble)]
                doubleMean = np.median(ddouble)
                # print(f"dd: {doubleMean}")

                # Process updData
                updData = updData[np.concatenate(([True], np.diff(updData[:, 0]) > 0)), :]
                base = mstats.mquantiles(updData[(updData[:, 0] > 0.4) & (updData[:, 0] < 0.6), 1], 0.25)
                updData[:, 1] = updData[:, 1] - base
                updData = updData[updData[:, 1] > 0, :]
                updData = updData[(updData[:, 0] <= 0.4) & (updData[:, 0] > ECAcutoff), :]
                updData = updData[np.argsort(updData[:, 0]), :]

                area = np.sum(np.diff(updData[:, 0]) * updData[:-1, 1])  # in mAV
                QH = area / scanRate 
                ECA = QH / 2.1e-4
                # print(f"ECA: {ECA}")
                dump[f"curve_{str(j)}"]={}
                dump[f"curve_{str(j)}"]["rate (V/s)"]=scanRate
                dump[f"curve_{str(j)}"]["Vmin (V)"]=np.min(A[:,2])
                dump[f"curve_{str(j)}"]["dd"]=doubleMean
                dump[f"curve_{str(j)}"]["ECA"]=ECA

            elif j==0:
                CVall = A[:, [2, 3]]
                updData = CVall[:, :2]
                maskCO = (updData[:, 0] > 0.5) & (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                topLimits = updData[maskCO, :]
                fCO = interpolate.interp1d(topLimits[:, 0], topLimits[:, 1], bounds_error=False)
                COdesorb = fCO(x_CO)
            if os.path.exists('temp'):
                os.remove('temp')
        data_dump[f"file_{str(i)}"]=dump

    return oldUpper, COdesorb, data_dump


if __name__=="__main__":
    os.makedirs('logs',exist_ok=True)
    os.makedirs('results/ecsa_dry/',exist_ok=True)
    log=open('logs/ecsa_normal.log','w')   
    index=1
    results={}    

    searchKey='Cathode CO*/*cv*.DTA'
    ECAcutoff = 0.08
    x_CO = np.arange(0.5, 0.95, 0.001)
    u_subfolders=find_cv_subfolders('.')

    for i, folderName in enumerate(u_subfolders,start=1):
        results[f"dir_{str(i)}"]={}

        plt.figure(index)
        index+=1
        file_info = find_and_sort_load_dta_files(u_subfolders[i-1])
        [oldUpper,COdesorb,data]=plot_COtripping(file_info)
        plt.savefig(f'results/ecsa_dry/ECSA_Dry_{i}-1.png')

        plt.figure(index)
        index+=1
        plt.plot(x_CO,oldUpper,'r--')
        plt.plot(x_CO,COdesorb,'b-')
        COECA=np.nansum( np.mean(np.diff(x_CO))*(COdesorb-oldUpper))
        plt.savefig(f'results/ecsa_dry/ECSA_Dry_{i}-2.png')

        results[f"dir_{str(i)}"]["data"]=data
        results[f"dir_{str(i)}"]["COECA"]=COECA
        with open('results/ecsa_dry/ecsa_results.json','w') as results_file:
            json.dump(results,results_file,indent=2,cls=NumpyEncoder)

    log.close()