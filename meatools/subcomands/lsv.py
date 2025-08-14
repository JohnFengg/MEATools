#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from scipy import interpolate
from scipy.stats import mstats
import sys
from glob import glob
from datetime import datetime
from pathlib import Path


# os.chdir(os.path.dirname(sys.argv[0]))

searchKey='LSV/*.DTA'

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
    
    print("Subfolders containing CV DTA files:")
    csv_folders=sorted(csv_folders)
    for folder in csv_folders:
        print(f"- {folder}" if folder != '.' else "- (root directory)")
    return csv_folders

# Example usage
u_subfolders=find_cv_subfolders('.')
#file_info = find_and_sort_load_dta_files(u_subfolders[1])

ECAcutoff = 0.08
#CVfileName = file_infoTest


for jk in range(len(u_subfolders)):
    file_path=u_subfolders[jk]
    file_info = find_and_sort_load_dta_files(file_path)
    plt.figure(jk+1)
    for i, (mtime, filepath) in enumerate(file_info, 1):
        readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {filepath}")
        

        with open(filepath, 'r') as f:
            for _ in range(59):
                next(f)
            content_from_line65 = f.read()

        u = re.split('CURVE', content_from_line65)
        plt.subplot(2,3,i)
        for j in range(min(5,len(u))):
            with open('temp', 'w') as fileID:
                fileID.write(u[j])
            A = np.loadtxt('temp', skiprows=2, usecols=range(8))
            plt.plot(A[:,2],A[:,3])
            if j>0 :
                scanRate=np.median(np.abs(np.diff(A[:,2])/np.diff(A[:,1])))
                print('rate='+str(scanRate)+' V/s')
                #print('Vmin='+str(np.min(A[:,2]))+' V')
                CVall = A[:, [2, 3]]
                startV = CVall[0, 0]
                updData = CVall[:, :2]
                mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                double1 = updData[mask1, :]
                
                # Interpolation
                x_new = np.arange(0.35, 0.55, 0.001)
                f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
                double1_interp = f1(x_new)
                
                H2cx = np.quantile(double1_interp,0.99)

                
                print(f"H2cx99%: {H2cx}")
                n=len(double1[:,0])
                m =(n*np.sum(double1[:,0]*double1[:,1]) - np.sum(double1[:,0])*np.sum(double1[:,1])) /(n*np.sum(double1[:,0]*double1[:,0]) - np.sum(double1[:,0])**2)
                b = (np.sum(double1[:,1]) - m*np.sum(double1[:,0])) / n

                H2cx2=m*0.8+b
                print(f"slopeReg: {m}")
                print(f"H2cxReg: {H2cx2}")


                
                #plt.title(f"dd: {ECA:.1f}")
            else:
                
                plt.title(f"{i}. {filepath}")
            #if os.path.exists('temp'):
            #    os.remove('temp')
            #
            
plt.show()
