#!/usr/bin/env python
import json,os,re
import numpy as np 

def read_json(filename,index=None):
    try:
        with open(f'results/{filename}','r') as f:
            data = json.load(f)
        if index is not None:
            data=data[index]
    except:
        data={}
    return data 


def extract_row(data,key,target):
    arr=np.array(data[key])
    idx=np.argmin(np.abs(arr - target))
    for k,v in data.items():
        if isinstance(v,list):
            data[k]=v[idx]
    return data


def strip_filename(filename: str) -> str:
    """
    sample:
    'HRL_D048_05_Polarisation02_80C_H2_60%O2_2_5_250kpa - 20250816 1528.csv'
    -> 'HRL_D048_05_Polarisation02_80C_H2_60%O2_2_5_250kpa'
    """
    return re.split(r" - \d{8} \d{4}",filename)[0]

def pairwise_average(big_dict):
    items=list(big_dict.items())
    length=len(items)
    results={}

    for i in range(0,length,2):
        if i+1>=length:
            name,data=items[i]
            results[strip_filename(name)]=data
            raise Warning("Odd files number is detected, please check the test results of polarization")

        name1,dict1=items[i]
        name2,dict2=items[i+1]
        new_name = strip_filename(name1)

        avg_dict = {}
        for key in dict1.keys():
            v1,v2=dict1[key],dict2[key]
            if isinstance(v1, list):
                arr = np.array([v1, v2])
                avg_dict[key]=arr.mean(axis=0).tolist()
            else:
                avg_dict[key]=float((v1+v2)/2)

        results[new_name] = avg_dict

    return results



if __name__=="__main__":
    result_tol={}

    sample_name=os.path.basename(os.getcwd())

    match=re.search(r"\((.*?)\)",sample_name)
    if match:
        station=match.group(1)
    else:
        station=None
    # print(read_json('test_sequence/all_csv_results_in_timeline.json')["A2"]["data"].keys())
    sample_area=read_json('test_sequence/all_csv_results_in_timeline.json')["A2"]["data"]["cell_active_area"][0]

    test_seq=read_json('test_sequence/test_order_in_timeline.json')

    esca=read_json('ecsa_normal/ecsa_results.json')
    for dir,results in esca.items():
        for file,info in results.items():
            eca,dd=[],[]
            for curve,value in info["data"].items():
                if isinstance(value, dict) and "ECA" in value:
                    eca.append(value["ECA"])
                    dd.append(value["dd"])
            avg_eca=float(np.mean(eca))
            avg_dd=float(np.mean(dd))
            info["avg_ECA"]=avg_eca
            info["avg_dd"]=avg_dd
    
    lsv=read_json('lsv/lsv_results.json')
    for dir,results in lsv.items():
        slopereg=[]
        for file,info in results.items():
            for curve,value in info["data"].items():
                if isinstance(value, dict) and "slopeReg" in value:
                    slopereg.append(value["slopeReg"])
    avg_reg=float(np.mean(slopereg))
    results["avg_slopereg"]=avg_reg

    pol=read_json('polarization/polarization_results.json')
    for file,data in pol.items():
        try:
            data=extract_row(data,"current density (A cm^(-2))",1)
        except:
            pass
    try:
        pol=pairwise_average(pol)
    except:
        pass

    otr=read_json('impedence/final_results.json')

    ecsa_dry=read_json('ecsa_dry/ecsa_results.json')
    for dir,results in ecsa_dry.items():
        for file,info in results["data"].items():
            eca,dd=[],[]
            for curve,value in info.items():
                if isinstance(value, dict) and "ECA" in value:
                    eca.append(value["ECA"])
                    dd.append(value["dd"])
            avg_eca=float(np.mean(eca))
            avg_dd=float(np.mean(dd))
            info["avg_ECA"]=avg_eca
            info["avg_dd"]=avg_dd
    
    result_tol["sample"]=sample_name
    result_tol["station_num."]=station
    result_tol["sample_area (cm^2)"]=sample_area
    result_tol["Test_Sequence"]=test_seq
    result_tol["ECSA"]=esca 
    result_tol["ECSA_Dry"]=ecsa_dry
    result_tol["LSV"]=lsv
    result_tol["Polarization"]=pol
    result_tol["O_Transfer_Resistance"]=otr

    with open('results.json','w') as f:
        json.dump(result_tol,f,indent=2)
    

