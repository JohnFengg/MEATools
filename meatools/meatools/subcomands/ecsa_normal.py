#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re,os,json
from scipy import interpolate
from scipy.stats import mstats
from datetime import datetime
from pathlib import Path
from collections import Counter
from glob import glob

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def find_and_sort_load_dta_files(root_folder):
    files = glob(os.path.join(root_folder, '**', searchKey), recursive=True)
    file_info = [(os.path.getmtime(f), f) for f in files]
    file_info.sort(reverse=False)
    return file_info

def find_cv_subfolders(root_dir='/ECSA'):
    root_path = Path(root_dir).resolve()
    csv_folders = {str(p.parent) for p in root_path.glob('**/' + searchKey)}
    csv_folders = sorted(csv_folders)
    log.write("Subfolders containing CV DTA files:\n")
    for folder in csv_folders:
        log.write(f"\t{folder}\n")
        log.write("**" * 80+'\n')
    return csv_folders

def process_curve_data(A, ECAcutoff):
    scanRate = np.median(np.abs(np.diff(A[:, 2]) / np.diff(A[:, 1])))
    CVall = A[:, [2, 3]]
    updData = CVall[:, :2]

    # extract double1&double2
    mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & \
            (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
    double1 = updData[mask1, :]

    mask2 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & \
            (np.concatenate(([0], np.diff(updData[:, 0]))) < 0)
    double2 = updData[mask2, :]

    # ranking
    double1 = double1[np.argsort(double1[:, 0])]
    double2 = double2[np.argsort(double2[:, 0])]
    double1 = double1[np.concatenate(([True], np.diff(double1[:, 0]) != 0)), :]
    double2 = double2[np.concatenate(([True], np.diff(double2[:, 0]) != 0)), :]

    # Interpolation
    x_new = np.arange(0.35, 0.451, 0.001)
    f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
    double1_interp = f1(x_new)
    f2 = interpolate.interp1d(double2[:, 0], double2[:, 1], bounds_error=False)
    double2_interp = f2(x_new)

    ddouble = np.abs(double1_interp - double2_interp)
    ddouble = ddouble[~np.isnan(ddouble)]
    doubleMean = np.median(ddouble)


    # calculating ECSA
    updData = updData[np.concatenate(([True], np.diff(updData[:, 0]) > 0)), :]
    base = mstats.mquantiles(updData[(updData[:, 0] > 0.4) & (updData[:, 0] < 0.6), 1], 0.25)
    updData[:, 1] -= base
    updData = updData[updData[:, 1] > 0, :]
    updData = updData[(updData[:, 0] <= 0.4) & (updData[:, 0] > ECAcutoff), :]
    updData = updData[np.argsort(updData[:, 0]), :]

    area = np.sum(np.diff(updData[:, 0]) * updData[:-1, 1])  # in mAV
    QH = area / scanRate
    ECA = QH / 2.1e-4

    # print(f'rate={scanRate} V/s')
    # print(f'Vmin={np.min(A[:, 2])} V')
    # print(f"dd: {doubleMean}")
    # print(f"ECA: {ECA}")

    dump={
        "rate (V/s)":scanRate,
        "Vmin (V)":np.min(A[:, 2]),
        "dd":doubleMean,
        "ECA":ECA
    }

    return dump

def parse_dta_format1(filepath, ECAcutoff):
    """running when len(u) < 2"""

    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Find the line where CURVE1 starts
    curve_start = next((i for i, line in enumerate(lines) if line.strip().startswith('CURVE')), None)
    data_lines = lines[curve_start + 3:]
    data = [line.split()[-1] for line in data_lines if line.strip() and len(line.split()) >= 10]

    value_counts = Counter(data)
    log.write("\nValue counts in the last column:\n")
    for value, count in value_counts.items():
        log.write(f"Value {value}: {count} occurrences\n")

    data_dump={}
    for value in value_counts.keys():
        if 0 < float(value) < len(value_counts) - 1:
            with open('temp', 'w') as f:
                for line in data_lines:
                    if line.strip() and len(line.split()) >= 10 and line.split()[-1] == value:
                        f.write(line.strip() + '\n')

            A = np.loadtxt('temp', skiprows=2, usecols=range(8))
            plt.subplot(2, 3, int(value))
            plt.plot(A[:, 2], A[:, 3])
            dump=process_curve_data(A, ECAcutoff)
            data_dump[f"curve_{str(value)}"]=dump
            os.remove('temp')
    data_dump["file_path"]=filepath
    return data_dump

def parse_dta_format2(filepath, ECAcutoff):
    """running when len(u) >= 2 """
    with open(filepath, 'r') as f:
        for _ in range(65):
            next(f)
        content_from_line65 = f.read()

    u = re.split('CURVE', content_from_line65)

    data_dump={}
    for j in range(len(u)):
        with open('temp', 'w') as fileID:
            fileID.write(u[j])
        A = np.loadtxt('temp', skiprows=2, usecols=range(8))
        plt.subplot(2, 3, j + 1)
        plt.plot(A[:, 2], A[:, 3])
        if 0 < j < len(u) - 1:
            dump=process_curve_data(A, ECAcutoff)
            data_dump[f"curve_{str(j)}"]=dump
        os.remove('temp')
    data_dump["file_path"]=filepath
    return data_dump

if __name__ == "__main__":
    ECAcutoff = 0.08
    searchKey = '*cv*.DTA'
    os.makedirs('logs',exist_ok=True)
    os.makedirs('results/ecsa_normal/',exist_ok=True)
    log=open('logs/ecsa_normal.log','w')
    u_subfolders = find_cv_subfolders('./ECSA/')
    results={}
    for jk, file_path in enumerate(u_subfolders,start=1):
        file_info = find_and_sort_load_dta_files(file_path)
        plt.figure(jk)
        results[f"dir_{str(jk)}"]={}
        
        for i, (mtime, filepath) in enumerate(file_info, 1):
            with open(filepath, 'r') as f:
                for _ in range(65):
                    next(f)
                content_from_line65 = f.read()

            if len(re.split('CURVE', content_from_line65)) < 2:
                log.write('different format'+str(len(re.split('CURVE', content_from_line65))))
                data_dump=parse_dta_format1(filepath, ECAcutoff)
            else:
                data_dump=parse_dta_format2(filepath, ECAcutoff)

            readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            results_jk_i={
                "time_stamp":readable_time,
                "data":data_dump
                }
            results[f"dir_{str(jk)}"][f"file_{str(i)}"]=results_jk_i
        plt.savefig(f'results/ecsa_normal/ECSA_{jk}.png')


    with open('results/ecsa_normal/ecsa_results.json','w') as results_file:
        json.dump(results,results_file,indent=2,cls=NumpyEncoder)
    log.close()