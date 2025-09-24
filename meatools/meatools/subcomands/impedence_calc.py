#!/usr/bin/env python
import numpy as np 
from test_squence import extract_data_from_file as edf
import os,re,json
from collections import defaultdict 
import cantera as ct
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class r_total_calc():
    def __init__(self,root_path,temp=80):
        self.root=os.path.abspath(root_path)
        self.temp=temp+273.15
        

    def parse_data_title(self):
        results=defaultdict(lambda:defaultdict(dict))
        for o2_dir in os.listdir(self.root):
            if o2_dir.startswith('._'):
                continue
            o2_path=os.path.join(self.root,o2_dir)
            if os.path.isdir(o2_path):
                match=re.search(r"(\d+(?:\.\d+)?)%O2",o2_path)
                if match:
                    o2_fraction=float(match.group(1))/100
                for file in os.listdir(o2_path):
                    if file.startswith('._') or not file.endswith('.csv'):
                        continue
                    match=re.search(r"(\d+)\s*kPa[a]?",file,re.IGNORECASE)
                    if match:
                        pressure=match.group(1)
                        filepath=os.path.join(o2_path,file)
                        results[o2_fraction][pressure]['data']={}
                        results[o2_fraction][pressure]['file_path']=filepath

        self.init_results=self.recursive_to_dict(results)

        return self.init_results
    
    def parse_data(self,long_out=False):
        for conc,info in self.init_results.items():
            for pressure,data in info.items():
                file_path=data['file_path']
                results=edf(file_path)
                data['data']['current']=list(results['data']['current'])
                data['data']['current_density']=list(
                                                        np.array(results['data']['current'])/
                                                        np.array(results['data']['cell_active_area'])
                                                        )
                o_conc,dry=self.concentration_calc(self.temp,float(pressure),float(conc))
                data['data']['o_concentration']=o_conc
                data['data']['dry_pressure']=dry
        if long_out:
            with open('results/impedence/raw_data.json','w') as f:
                json.dump(self.init_results,f,indent=2)
                f.close()

        return self.init_results

    def r_calc(self,fit_plot=False):
        results=defaultdict(lambda:defaultdict(list))
        for conc,info in self.init_results.items():
            for pressure,data in info.items():
                cd=np.array(data['data']['current_density'])
                cut_off=int(len(cd)*0.8)
                cd_avg=np.mean(cd[:cut_off])
                results[pressure]['current_density'].append(cd_avg)
                results[pressure]['o_concentration'].append(data['data']['o_concentration'])
                results[pressure]['dry_pressure'].append(data['data']['dry_pressure'])
        # print(results)
        for pressure,v in results.items():
            o_conc=v['o_concentration']
            cur_d=v['current_density']
            fitted=self.fitting(o_conc,cur_d,prefix=pressure,plot=fit_plot)
            r_total=4*96485/1000/fitted['a']
            results[pressure]['r_total'].append(r_total)
        with open('results/impedence/fitted_r_total.json','w') as f:
            json.dump(results,f,indent=2)
            f.close()
        
        self.fitted_results=results

        return  self.fitted_results

    def run_calc(self,long_out=False,fit_plot=False):
        self.parse_data_title()
        self.parse_data(long_out)
        self.r_calc(fit_plot)
        pressures,rs_total=[],[]
        for pressure,data in self.fitted_results.items():
            pressures.append(data['dry_pressure'][0])
            rs_total.extend(data['r_total'])
        results=self.fitting(pressures,rs_total,prefix='final',plot=fit_plot)
        r_diff=(101*results['a']+results['b'])*100
        r_other=101*results['b']
        results["r_diff (s m^-1)"]=r_diff
        results["r_other (s m^-1)"]=r_other
        results=self.recursive_to_dict(results)
        # print(results)
        with open('results/impedence/final_results.json','w') as f:
            json.dump(results,f,indent=2)
            f.close()

    @staticmethod
    def recursive_to_dict(d):
        if isinstance(d, defaultdict):
            d = {k: r_total_calc.recursive_to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: r_total_calc.recursive_to_dict(v) for k, v in d.items()}
        return d                    

    @staticmethod
    def concentration_calc(temp,pressure,ratio):
        # water=ct.Water()
        # water.TQ=temp,0
        # water_pressure=water.vapor_pressure
        water_pressure=47379         # unit-> Pa
        dry_pressure=pressure*1000-water_pressure  # unit-> Pa
        gas=ct.Solution('air.yaml')
        gas.TPX=temp,dry_pressure,{'O2':ratio,'N2':1-ratio}
        conc=gas.concentrations[gas.species_index('O2')] # unit-> mol/L
        return conc,dry_pressure/1000

    @staticmethod
    def fitting(x,y,prefix,plot=False):
        linear_func=lambda x,a,b: a*x+b
        x,y=np.array(x),np.array(y)
        popt,pcov=curve_fit(linear_func,x,y)
        a,b=popt
        y_pred=linear_func(x, a, b)
        residuals=y-y_pred
        mse=np.mean(residuals**2)
        r2=1-np.sum(residuals**2)/np.sum((y-np.mean(y))**2)
        results={'function':'a*x+b',
                 'a':a,
                 'b':b,
                 'a_err':pcov[0][0],
                 'b_err':pcov[1][1],
                 'mse':mse,
                 'r2':r2
                    }
        if plot:
            plt.figure()
            plt.scatter(x,y,label='Data')
            plt.plot(x,y_pred,'r-',label=f'Fit:y={a:.3f}x+{b:.3f}')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Linear Fit (MSE={mse:.4g},R2={r2:.4f})")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/impedence/{prefix}_fitting.png')

        return results

    
if __name__ == "__main__":
    os.makedirs('logs',exist_ok=True)
    os.makedirs('results/impedence',exist_ok=True)
    root_path='OTR/'
    calc=r_total_calc(root_path=root_path)
    # calc.parse_data_title()
    # calc.parse_data()
    # calc.r_calc()
    calc.run_calc(long_out=True,fit_plot=True)