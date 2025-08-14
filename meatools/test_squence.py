#!/usr/bin/env python
import os,json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from glob import glob
from datetime import datetime
import pandas as pd
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  


"""
When I wrote this code, only God and I know it.
Now ...
ONLY God knows...
"""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def find_csv_files(root_path):
    """Find all CSV files recursively with relative paths"""
    return [os.path.join(root, f) 
            for root, _, files in os.walk(root_path) 
            for f in files if f.lower().endswith('.csv')]

def find_csv_pol_files(root_path):
    files = glob(os.path.join(root_path, '**', searchKey), recursive=True)
    
    # Create list of tuples (mtime, filepath)
    #file_info = [(os.path.getmtime(f), f) for f in files]
    
    # Sort by mtime (first element of tuple)
    #file_info.sort(reverse=False)
    
    return files #file_info

def extract_start_times(csv_files):
    """Extract start times from files with validation"""
    results = []
    
    for filepath in csv_files:
        try:
            with open(filepath, 'r', encoding='ISO-8859-1') as f:
                lines = [line.strip() for line in f.readlines()]
                
                for line_num in [5, 6, 7]:  # Lines 6-8 (0-indexed 5-7)
                    if line_num >= len(lines):
                        continue
                        
                    parts = lines[line_num].split(',') or lines[line_num].split('\\t')
                    if len(parts) >= 2 and parts[0].strip() == 'Start time':
                        time_str = parts[1].strip()
                        try:
                            time_obj = datetime.strptime(time_str, '%m/%d/%y %H:%M:%S')
                            results.append({
                                'path': filepath,
                                'time': time_obj,
                                'time_str': time_str,
                                'line': line_num + 1
                            })
                            break
                        except ValueError:
                            continue
        except Exception as e:
            log.write(f"\nError processing {os.path.basename(filepath)}: {e}\n")
    
    return sorted(results, key=lambda x: x['time'])

def extract_data_from_file(filepath):
    #print('active area col #'+str(find_column_number(filepath,'cell_active_area')))
    with open(filepath, 'r', encoding='ISO-8859-1') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Find "Time stamp" line (case-insensitive)
    data_start = None
    for i, line in enumerate(lines):
        if line.lower().startswith('time stamp'):
            data_start = i
            break
    # print(data_start)       
    if data_start is None:
        log.write(f"\nNo 'Time stamp' line found in {filepath}\n")
        return None
    
    # Load data with numpy
    data = np.genfromtxt(filepath, 
                     delimiter=',', 
                     skip_header=data_start+1, invalid_raise='false',encoding='ISO-8859-1')
    
    # Get header line to find column indices
    header = lines[data_start].lower().split(',')
    col_indices = {}
    
    # Find our target columns
    targets = [
        'elapsed time',
        'current',
        'current_set',
        'cell_voltage_001',
        lambda x: x.endswith('.resistance'),  # For any resistance column
        'temp_coolant_inlet',
        'temp_cathode_dewpoint_gas',
        'temp_anode_dewpoint_gas',
        'pressure_cathode_inlet',
        'cell_active_area'
    ]
    
    for i, col in enumerate(header):
        col = col.strip()
        for target in targets:
            if (callable(target) and target(col)) or (col == target):
                col_name = 'resistance' if callable(target) else col.replace(' ', '_')
                col_indices[col_name] = i
                break
    
    # Extract the columns we found
    extracted = {}
    for name, idx in col_indices.items():
        extracted[name] = data[:, idx]
        
    return {
        'file_info': filepath,
        'data': extracted,
        'columns_found': list(col_indices.keys())
    }

def plot_voltages(all_results):
    """Generate subplots for each file's current vs elapsed time"""
    num_files = len(all_results)
    if num_files == 0:
        return
    
    # Determine subplot grid size
    cols = min(3, num_files)
    rows = (num_files + cols - 1) // cols
    
    # Create figure with subplots
    #plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Flatten axes array if needed
    if num_files > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    
    for idx, (key, result) in enumerate(all_results.items()):
        ax = axs[idx]
        data = result['data']
        filename = os.path.basename(result['file_info'])
        
        if 'elapsed_time' in data and 'current' in data:
            ax.plot(data['elapsed_time'], data['cell_voltage_001'])
            ax.set_title(filename, fontsize=10)
            ax.set_xlabel('Elapsed Time')
            ax.set_ylabel('Voltage')
            ax.grid(True)
            ax1=ax.twinx()
            ax1.plot(data['elapsed_time'], data['current'],'r-')
        else:
            ax.text(0.5, 0.5, 'Missing required columns', 
                   ha='center', va='center')
            ax.set_title(filename, fontsize=10)
    
    # Hide unused subplots
    for j in range(len(all_results), len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/test_sequence/IV_vs_time.png')
    
def plot_Tcells(all_results):
    """Generate subplots for each file's current vs elapsed time"""
    num_files = len(all_results)
    if num_files == 0:
        return
    
    # Determine subplot grid size
    cols = min(3, num_files)
    rows = (num_files + cols - 1) // cols
    
    # Create figure with subplots
    #plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Flatten axes array if needed
    if num_files > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    
    for idx, (key, result) in enumerate(all_results.items()):
        ax = axs[idx]
        data = result['data']
        filename = os.path.basename(result['file_info'])
        
        if 'elapsed_time' in data and 'current' in data:
            ax.plot(data['elapsed_time'], data['temp_anode_dewpoint_gas'],'b-')
            ax.plot(data['elapsed_time'], data['temp_cathode_dewpoint_gas'],'y-')
            ax.plot(data['elapsed_time'], data['temp_coolant_inlet'],'r-')
            ax.set_title(filename, fontsize=10)
            ax.set_xlabel('Elapsed Time')
            ax.set_ylabel('Temperature')
            ##ax.set_ylabel('Pressure')
            ax.grid(True)
            ax1=ax.twinx()
            ax1.plot(data['elapsed_time'], data['pressure_cathode_inlet'],'g-')
        else:
            ax.text(0.5, 0.5, 'Missing required columns', 
                   ha='center', va='center')
            ax.set_title(filename, fontsize=10)
    
    # Hide unused subplots
    for j in range(len(all_results), len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/test_sequence/TP_vs_time.png')

def plot_Pol(all_results,sampleArea):
    """Generate subplots for each file's current vs elapsed time"""
    num_files = len(all_results)
    if num_files == 0:
        return
    
    # Determine subplot grid size
    rows = num_files
    cols = 5
    
    # Create figure with subplots
    #plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Flatten axes array if needed
    if num_files > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    
    Pols={}
    for idx, (key, result) in enumerate(all_results.items()):
        ax = axs[idx]
        data = result['data']
        filename = os.path.basename(result['file_info'])
        
        if 'elapsed_time' in data and 'current' in data:
            time=data['elapsed_time']
            voltage=data['cell_voltage_001']
            ocv=data['cell_voltage_001'][-1]
            current=data['current']
            HFR=data['resistance']
            currentSet=data['current_set']
            voltageIR=voltage+HFR*current/1000

            plt.subplot(rows,5,5*idx+1)
            plt.plot(current,voltage)#, 'b-', linewidth=1, label='Current')
            plt.title('IV curve')

            plt.subplot(rows,5,5*idx+2)
            plt.plot(time,current)#, 'b-', linewidth=1, label='Current')
            plt.title('current v.s. time curve')
            #plt.plot(np.diff(currentSet))

            positions = np.where(np.diff(currentSet) < -0.95)[0]+1
            diffs=np.diff(positions)
            split_indices = np.where(diffs > 50)[0] + 1
            segs=positions[split_indices]
            # print('load points:'+str(len(segs)))
            # print(np.diff(time[segs]))
            plt.subplot(rows,5,5*idx+3)
            plt.plot(time,voltage)
            plt.plot(time[segs],voltage[segs],'o')#, 'b-', linewidth=1, label='Current')
            plt.title('voltage v.s. time curve')


            segsVol=np.zeros((len(segs),5))
            s=0


            for steps in segs:
                volAVG=np.average(voltage[(time<time[steps])&(time>time[steps]-polReportAVG)])
                volIRAVG=np.average(voltageIR[(time<time[steps])&(time>time[steps]-polReportAVG)])
                segsVol[s,1]=volAVG
                segsVol[s,2]=volIRAVG
                segsVol[s,3]=volIRAVG-volAVG
                segsVol[s,0]=np.average(current[(time<time[steps])&(time>time[steps]-polReportAVG)])
                s=s+1
            segsVol[:,4]=segsVol[:,0]/sampleArea
            # print(segsVol)  
            Pols[filename]={
                "OCV (V)":ocv,
                "current (A)":segsVol[:,0],
                "voltage (V)":segsVol[:,1],
                "voltage+IR (V)":segsVol[:,2],
                "IR (V)":segsVol[:,3],
                "current density (A cm^(-2))":segsVol[:,4]
            }

            for row in segsVol:
                if abs(row[-1]-1)<0.1:
                    log.write(' '.join(map(str,row))+'\n') 

            plt.subplot(rows,5,5*idx+4)
            plt.plot(segsVol[:,0],segsVol[:,1])
            plt.title('voltage v.s. current curve')

            plt.subplot(rows,5,5*idx+5)
            plt.plot(segsVol[:,3],segsVol[:,2])
            plt.title('resistance v.s. IR voltage curve')

            plt.xscale('log')
            plt.grid(True,which='both')
        
        else:
            ax.text(0.5, 0.5, 'Missing required columns', 
                   ha='center', va='center')
            ax.set_title(filename, fontsize=10)

    # Hide unused subplots
    #for j in range(len(all_results), len(axs)):
    #    axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/polarization/polarization_curves.png')
    
    with open('results/polarization/polarization_results.json','w') as Polresults:
        json.dump(Pols,Polresults,indent=2,cls=NumpyEncoder)

def find_column_number(file_path, target_column):
    # Read the file to find where the data starts
    with open(file_path, 'r',encoding='ISO-8859-1') as f:
        lines = f.readlines()
    
    # Find the header line
    header_line = None
    for i, line in enumerate(lines):
        if line.startswith('Time stamp'):
            header_line = i
            break
    
    if header_line is None:
        raise ValueError("Could not find header line starting with 'Time stamp'")
    
    # Read just the header row
    df = pd.read_csv(file_path, skiprows=header_line, nrows=0, encoding='ISO-8859-1')
    column_names = df.columns.tolist()
    
    # Find the target column (case-sensitive)
    try:
        column_number = column_names.index(target_column) + 1  # +1 because Python uses 0-based indexing
        print(df[10,column_number-1])
        #return column_number  
    except ValueError:
        return None  # Column not found

def areaDetermine(all_results):
    num_files = len(all_results)
    if num_files == 0:
        return
    areaAll=np.zeros(num_files)
    for idx, (key, result) in enumerate(all_results.items()):
        data = result['data']
        areaAll[idx]=np.nanmedian(data['cell_active_area'])
    # print(areaAll)
    return np.average(areaAll)
    
def plot_step_timeline(steps, title="Process Timeline", figsize=(20,10)):
    """
    Plots a clean linear timeline of steps with durations.
    
    Args:
        steps (list): List of step dictionaries with:
                     - "step" (int/str): Step identifier
                     - "start" (str): ISO format datetime string
                     - "duration" (float): Duration in minutes
        title (str): Plot title
        figsize (tuple): Figure dimensions
    """
    # Convert strings to datetime objects
    for step in steps:
        step['start_dt'] = datetime.strptime(step['start'], '%Y-%m-%d %H:%M:%S')
        step['end_dt'] = step['start_dt'] + timedelta(minutes=step['duration'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each step
    for i, step in enumerate(steps):
        # Main timeline segment
        ax.plot([step['start_dt'], step['end_dt']], 
                [i, i], 
                'o-', 
                linewidth=3,
                markersize=8,
                label=step['name'])
        
        # Duration label
        mid_point = step['start_dt'] + timedelta(minutes=step['duration']/2)
        ax.text(mid_point, i+0.15, 
                f"{step['duration']} mins",
                ha='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_title(title, pad=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.yaxis.set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
        # Move legend to bottom (with 2 columns if many steps)
    #ncol = 2 if len(steps) > 3 else 1
    ax.legend(
        loc='upper left',
        #bbox_to_anchor=(0.5, -0.15),  # Position below plot
        ncol=1,
        frameon=False
    )

    
    plt.tight_layout()
    plt.savefig('results/test_sequence/test_sequence.png')

def main():
    log.write("CSV Start Time Analyzer\n")
    log.write("=" * 40)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ##############################################################
    ############### test squence analyzer#########################
    ## detect  all of the csv files and find start time from them ##
    ##############################################################
    csv_files = find_csv_files(script_dir)
    if not csv_files:
        log.write("\nNo CSV files found in directory tree\n")
    else:
        log.write(f"\nFound {len(csv_files)} CSV files\n")
    
    results = extract_start_times(csv_files)
    if not results:
        log.write("No valid start times found in lines 6-8\n")
    else:
        log.write(f"Found {len(results)} files with valid start times:\n")
        log.write("=" * 80)
    

    all_results = {}
    steps_data = []
    for i, res in enumerate(results, 1):
        rel_path = os.path.relpath(res['path'], script_dir)
        result = extract_data_from_file(rel_path)
        all_results[f"A{i}"] = result
        data = result['data']
        duration=np.max(data['elapsed_time'])/60

        log.write(f"\n{i}. {rel_path}")
        # log.write(f"   Line {res['line']}: Start time, {res['time_str']}")
        log.write(f"\tParsed: {res['time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"\tDuration: {duration:.1f} mins\n")
        log.write(f"\tFound columns: {', '.join(result['columns_found'])}\n")
        log.write("-" * 80)
        # Append structured data to list
        steps_data.append({
            "step": i,
            "start": res['time'].strftime('%Y-%m-%d %H:%M:%S'),
            "duration": float(f"{duration:.1f}"),
            "name": f"Step {i}: {rel_path}"
        })
    with open('results/test_sequence/all_csv_results_in_timeline.json','w') as all_json:
        json.dump(all_results,all_json,indent=2,cls=NumpyEncoder)
    all_json.close()

    time_line={}
    total_time=0
    for step in steps_data:
        starttime = datetime.strptime(step['start'], '%Y-%m-%d %H:%M:%S')
        endtime = starttime + timedelta(minutes=step['duration'])
        endtime_str = endtime.strftime('%Y-%m-%d %H:%M:%S')
        step["end"]=endtime_str
        time_line[str(step["step"])]={"start_time":step["start"],
                                      "end_time":step["end"],
                                      "duration (mins)":step["duration"],
                                      "file_path":step["name"]}
        total_time+=step["duration"]
    time_line["total_time (mins)"]=total_time
    with open('results/test_sequence/test_order_in_timeline.json','w') as timeline:
        json.dump(time_line,timeline,indent=2,cls=NumpyEncoder)
    timeline.close()
    
    #plot test sequence
    plot_step_timeline(steps_data, title="Test sequence")
    plot_voltages(all_results)
    plot_Tcells(all_results)

    ##########################################################
    ############### polarization analyzer#####################
    ##########################################################
    csv_files = find_csv_pol_files(script_dir)
    log.write('\n'+"**" * 80)
    log.write(f"\nFound {len(csv_files)} Pol CSV files")
    results = extract_start_times(csv_files)

    all_pol_results={}
    step_pol_data={}
    log.write(f"Found {len(results)} Pol files with valid start times:\n")
    log.write("=" * 80)
    for i, res in enumerate(results, 1):
        rel_path = os.path.relpath(res['path'], script_dir)
        result = extract_data_from_file(rel_path)
        data = result['data']
        duration=np.max(data['elapsed_time'])/60
        all_pol_results[f"A{i}"] = result
        log.write(f"\n{i}. {rel_path}")
        log.write(f"\tLine {res['line']}: Start time, {res['time_str']}\n")
        log.write(f"\tParsed: {res['time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"\tFound columns: {', '.join(result['columns_found'])}\n")
        log.write("-" * 80)
        endtime = res['time'] + timedelta(minutes=duration)
        endtime_str = endtime.strftime('%Y-%m-%d %H:%M:%S')
        step_data={
            "start_time":res['time'].strftime('%Y-%m-%d %H:%M:%S'),
            "end_time":endtime_str,
            "duration (mins)":float(f"{duration:.1f}"),
            "file_path":f"Step {i}: {rel_path}"
        }
        step_pol_data[str(i)]=step_data
    #plt.figure(1)
    sampleSize=areaDetermine(all_pol_results)
    step_pol_data["sampleSize (cm2)"]=sampleSize

    log.write(f"\nConfirmed the sample size is {sampleSize} cm2")
    with open('results/polarization/all_pol_results.json','w') as all_pol_json:
        json.dump(all_pol_results,all_pol_json,indent=2,cls=NumpyEncoder)
    all_json.close()

    with open('results/polarization/polarization.json','w') as pol:
        json.dump(step_pol_data,pol,indent=2)

    plot_Pol(all_pol_results,sampleSize)


if __name__ == '__main__':
    #input zone#
    searchKey='*Pol*-*.csv'
    polReportAVG=30                # secs
    os.makedirs('logs',exist_ok=True)
    os.makedirs('results/test_sequence',exist_ok=True)
    os.makedirs('results/polarization',exist_ok=True)
    log=open('logs/test_sequence.log','w')
    main()
    log.close()
