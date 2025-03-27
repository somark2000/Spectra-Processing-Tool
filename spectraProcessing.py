import tkinter as tk
from tkinter import  ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn import preprocessing
import numpy as np
from collections import defaultdict
import re
import csv
import os

class Measurement:
    def __init__(self, name, value, std, wave, alias):
        self.name = name
        self.value = value
        self.std = std
        self.wave = wave
        self.alias = alias
        self.afw=[]
        self.af=[]
        self.af_std=[]
    
    def find_max(self):
        try:
            max_index = np.argmax(self.value)
        except ValueError:
            return 0,0,0
        max_index = np.argmax(self.value)
        return self.value[max_index], self.wave[max_index], self.std[max_index]

class Fluorophor:
    def __init__(self,name,auto):
        self.name = name
        self.auto = auto

def extract_nume_label(filename):
    """Extracts the NUME_LABEL from a filename."""
    match = re.search(r"\_\_\[\[(.*?)\]\]\_", filename)
    if match:
        return match.group(1)
    else:
        return None

def is_control(filename):
    match = re.search(r"\_\_\[\[(.*?)\]\]\_", filename)
    if match:
        return match.group(1)
    else:
        return None
    
# Select input files
def select_input_files():
    files = filedialog.askopenfilenames(title="Select Input Files",filetypes=[("Text files", "*.txt")])
    input_files_var.set(", ".join(files))
    file_paths = [x.strip() for x in files]
    
    # Group files by their basename prefix
    grouped_files = defaultdict(list)
    for file_path in file_paths:
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        grouped_files[basename].append(file_path)
    
    for i,group in enumerate(grouped_files.keys()):
        tk.Label(root, text=f"{group.split('/')[-1]}: ").grid(row=6+i, column=0, padx=10, pady=5, sticky="e")
        entry = tk.Entry(root, width=80)
        entry.grid(row=6+i, column=1, padx=10, pady=5, columnspan=2)
        # Store entry widget in dictionary with filename as key
        alias_entries[group] = entry
    submit_button.grid(row=7+len(grouped_files), column=0, columnspan=3, pady=10)

# Select autofluorescence files
def select_autofluorescence_files():
    if fluorophors[solution_var.get()] == "True":
        files = filedialog.askopenfilenames(title="Select Autofluorescence Files",filetypes=[("Text files", "*.txt")])
        autofluorescence_var.set(", ".join(files))

# Update autofluorescence field based on solution selection
def on_solution_change(event):
    print(f"solution: {solution_var.get()} fd {fluorophors[solution_var.get()]}")
    if solution_var.get()!="":
        # Enable or disable autofluorescence field based on solution selection
        if fluorophors[solution_var.get()] == "True":
            autofluorescence_button.config(state="normal")
            autofluorescence_entry.config(state="readonly")
        else:
            autofluorescence_button.config(state="disabled")
            autofluorescence_entry.config(state="disabled")
        
# Save new fluorophor
def save_new(new_name_var,check_var):
    # Append-adds at last
    documents_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(documents_path, "Spectra Processing\Executable")
    if os.path.exists(folder_path):
        file1 = open(f"{folder_path}/fluorophor_data.txt", "a")  # append mode
        file1.write(f"\n{new_name_var.get()}, {check_var.get()} ")
        file1.close()

# Add new fluorophor
def add_fluorophor():
    # Toplevel object which will 
    # be treated as a new window
    newWindow = tk.Toplevel(root)
    newWindow.title("Add new Fluorophor")
    # Variables
    new_name_var = tk.StringVar()
    check_var = tk.BooleanVar()
 
    # Name field
    tk.Label(newWindow, text="Fluorophor Name:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    name_entry = tk.Entry(newWindow, textvariable=new_name_var, width=40)
    name_entry.grid(row=0, column=1, padx=10, pady=5, columnspan=2)
    
    # Checkbox for autofluorescence property
    checkbox = tk.Checkbutton(newWindow, text="Autofluorescence ",variable=check_var)
    checkbox.grid(row=1, column=0, padx=10, pady=5)
    
    # Submit button
    new_submit_button = ttk.Button(newWindow, text="Save", command=lambda: save_new(new_name_var,check_var))
    new_submit_button.grid(row=2, column=0, columnspan=3, pady=10)
    
# Find the folder with the given name in the user's Pictures directory
def find_folder(folder_name):
  documents_path = os.path.join(os.path.expanduser("~"), "Pictures")
  folder_path = os.path.join(documents_path, folder_name)
  if os.path.exists(folder_path):
    return folder_path
  else:
    return None

# Retrieve data from UI
def submit():
    solution = solution_var.get()
    input_files = input_files_var.get()
    autofluorescence_files = autofluorescence_var.get()
    min_spectra = min_spectra_var.get()
    max_spectra = max_spectra_var.get()
    output_name = name_var.get()
    if min_spectra == "": min_spectra = 0.0
    else: min_spectra = float(min_spectra)
    if max_spectra == "": max_spectra = 0.0
    else: max_spectra = float(max_spectra)
    aliases = {filename: entry.get() for filename, entry in alias_entries.items()}
    peaks = peak_display_var.get()
    normalize = normalize_var.get()
    denoise = denoise_var.get()
    
    # Collect and print the values for debugging purposes
    print("Solution:", solution)
    print("Input Files:", input_files)
    print("Autofluorescence Files:", autofluorescence_files)
    print("Spectra Limits:", f"Min: {min_spectra}, Max: {max_spectra}")
    print("Output Name:", output_name)
    print("Aliases:", aliases)
    base_dir = find_folder("Spectra Processing")
    base_dir=os.path.join(base_dir, output_name)
    try:
        os.mkdir(base_dir)
    except OSError:
        pass
    
    process_files(solution,input_files,autofluorescence_files,min_spectra, max_spectra,output_name,base_dir,aliases,peaks,normalize,denoise)

# Open a Tk window to select multiple txt files
def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
    return file_paths

# Function to normalize a spectrum
def normalize_spectrum(wavelengths, intensities):
    max_intensity = max(intensities)
    if max_intensity == 0:
        return intensities
    return [intensity / max_intensity for intensity in intensities]

# Process files and perform tasks
def process_files(solution,input_files,autofluorescence_files,min_spectra, max_spectra,output_name,basedir,aliases,peaks,normalize,denoise):
    # Split the input files string into a list of file paths
    file_paths = input_files.split(',')
    file_paths = [x.strip() for x in file_paths]
    file_paths_af = autofluorescence_files.split(',')
    file_paths_af = [x.strip() for x in file_paths_af]
    
    # Group files by their basename prefix
    grouped_files = defaultdict(list)
    grouped_files_baseline = defaultdict(list)
    for file_path in file_paths:
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        grouped_files[basename].append(file_path)
    print(grouped_files)
        
    for file_path in file_paths_af:
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        grouped_files_baseline[basename].append(file_path)
    grouped_files_baseline.pop('', None)

    headers = []
    data = []
    max_ctr_nom = 1
    for group, files in grouped_files.items():
        print(f"Processing group: {group}")
        headers.append(group)
        
        wave = []
        samples = []
        skip_lines = 8
        delimiter = ';'
        skip_footer = 0
        if solution == "SERS_BWTeK": skip_lines = 1
        elif solution == "SERS_Avantes": skip_lines = 0
        elif solution == "FT-IR": 
            skip_lines = 19
            delimiter = '	'
            skip_footer = 42
        elif solution == "UV-Vis": 
            skip_lines = 19
            delimiter = '	'
            skip_footer = 42
        for file_path in files:
            # Read the file
            file_data = np.genfromtxt(file_path, encoding="UTF-8", dtype = np.float64, skip_header = skip_lines, delimiter = delimiter, skip_footer=skip_footer)
            # Extract data
            wave.append(file_data[:,0])
            sample = file_data[:,1]
            samples.append(sample)

        # Convert the list of lists to a NumPy array
        array = np.array(samples)
        arrayw = np.array(wave)
        print(array,arrayw)

        # Calculate the mean and standard deviation along the first axis (rows)
        means = np.mean(array, axis=0)
        meansw = np.mean(arrayw, axis=0)
        stds = np.std(array, axis=0)

        # Combine the means and standard deviations into a list of tuples
        result = Measurement(group,means,stds,meansw,aliases[group])
        print("current group is ",group)
        
        # check for autofluorescence
        print("af groups: ",len(grouped_files_baseline.keys()))
        if len(grouped_files_baseline.keys())>0: 
            gr = list(grouped_files_baseline.keys())[0]
            fl = grouped_files_baseline[gr]
            w = []
            s = []
            print("the groups are ",gr,group)
            for file_path in fl:
                # Read the file and extract the second column using numpy
                file_data = np.genfromtxt(file_path, encoding="UTF-8", dtype=np.float64, skip_header=8,delimiter=';')

                # Extract data from the second column
                w.append(file_data[:,0])
                s.append(file_data[:,1])
                
            # Convert the list of lists to a NumPy array
            array = np.array(s)
            arrayw = np.array(w)
            result.af = np.mean(array, axis=0)
            result.afw = np.mean(arrayw, axis=0)
            result.af_std = np.std(array, axis=0)
        
        if len(result.af) == 0:
            if solution == "SERS_BWTeK":
                result.afw = result.wave
                result.af = np.polyval(np.polyfit(result.wave, result.value, 1), result.wave)
                result.af_std = np.zeros(len(result.value))
            else:
                result.afw = result.wave
                result.af = np.zeros(len(result.value))
                result.af_std = np.zeros(len(result.value))
        # print("af shape ",len(result.af))
        data.append(result)
    
    for measurement in data:
        if measurement.wave[0] > measurement.wave[-1]:
            measurement.wave = measurement.wave[::-1]
            measurement.value = measurement.value[::-1]
            measurement.std = measurement.std[::-1]
            measurement.af = measurement.af[::-1]
            measurement.afw = measurement.afw[::-1]
            measurement.af_std = measurement.af_std[::-1]
    
    # Find indexes for plotting
    for measurement in data:
        index_start = 0
        index_end = len(measurement.value)-1
        if min_spectra != 0.0:
            index_start = (np.abs(measurement.wave - min_spectra)).argmin()
        if max_spectra != 0.0:
            index_end = (np.abs(measurement.wave - max_spectra)).argmin()
        # if index_start > index_end: index_start,index_end = index_end,index_start
        # print(index_start,measurement.wave[index_start],min_spectra,index_end,measurement.wave[index_end],max_spectra)
        
        # print("test1 ",len(measurement.value),measurement.name,len(measurement.af))
        measurement.value = measurement.value[index_start:index_end]
        measurement.wave = measurement.wave[index_start:index_end]
        measurement.std = measurement.std[index_start:index_end]
        index_start = 0
        index_end = len(measurement.af)-1
        if min_spectra != 0.0:
            index_start = (np.abs(measurement.afw - min_spectra)).argmin()
        if max_spectra != 0.0:
            index_end = (np.abs(measurement.afw - max_spectra)).argmin()
        measurement.afw = measurement.afw[index_start:index_end]
        measurement.af = measurement.af[index_start:index_end]
        measurement.af_std = measurement.af_std[index_start:index_end]
        # print("test2 ",len(measurement.value),measurement.name,len(measurement.af))
        
    # Subtract autofluorescence and adjust std
    for measurement in data:
        # print(len(measurement.value),measurement.name,len(measurement.af))
        measurement.value -= measurement.af
        measurement.std += measurement.af_std
        if "_contr_" in measurement.name: 
            # print(measurement.name)
            max_ctr_nom,_,_ = measurement.find_max()
            # print(max_ctr_nom)
    
    # Normalize the data
    if solution == "SERS_BWTeK":
        for measurement in data:
            intensities_array = np.array(measurement.value).reshape(-1, 1)  # Reshape to a column vector
            measurement.value = preprocessing.normalize(intensities_array, axis=0).flatten()
            intensities_array = np.array(measurement.std).reshape(-1, 1)  # Reshape to a column vector
            measurement.std = preprocessing.normalize(intensities_array, axis=0).flatten()
    elif solution in ("UV-Vis","FT-IR"): pass
    else:
        for measurement in data: 
            a = np.array([float(x)/max_ctr_nom for x in measurement.value])
            measurement.value = a
            a = np.array([float(x)/max_ctr_nom for x in measurement.std])
            measurement.std = a
    
    # Transpose and export data to output.xls
    # Create a list of dictionaries to store the data for each measurement
    output = []
    for measurement in data:
        output.append({
        'Name': measurement.alias,
        **dict(zip(measurement.wave, measurement.value))
        })
        if solution not in ("UV-Vis","FT-IR","SERS_BWTeK"):
            output.append({
            'Name': f"{measurement.alias}_raw",
            **dict(zip(measurement.wave, measurement.std * max_ctr_nom))
            })
        output.append({
        'Name': f"{measurement.alias}_std",
        **dict(zip(measurement.wave, measurement.std))
        })

    # Create a DataFrame from the data
    df = pd.DataFrame(output)

    # Write the DataFrame to an Excel file
    df.to_excel(f'{basedir}/output_{output_name}.xlsx', index=False)
    print(f"Data exported to {basedir}/output_{output_name}.xlsx")

    # Save relevant data to a CSV file
    with open(f'{basedir}/max_values_{output_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Name', 'Max Wave', 'Max Value', 'Max Std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # save data
        for measurement in data:
            max_value, max_wave, max_std = measurement.find_max()
            writer.writerow({'Name': measurement.alias,
                          'Max Value': max_value,
                          'Max Wave': max_wave,
                          'Max Std': max_std})
    
    # Plot the data
    cumulative_height = np.zeros(len(data[0].value))
    plt.figure(figsize=(20,14))
    for measurement in data:
        # name = extract_nume_label(measurement.name)
        # plt.plot(measurement.wave, measurement.value, label=name)
        # Plot the mean values
        try:
            plt.plot(measurement.wave, measurement.value+cumulative_height[:len(measurement.wave)], label=measurement.alias)
        except ValueError:
            messagebox.showerror("Error", "Please adjust the domain to be plotted !")
        # plt.plot(measurement.wave, measurement.value+cumulative_height[:len(measurement.wave)], label=measurement.alias)
        
        # Plot the upper and lower bounds (mean +/- standard deviation)
        try:
            plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)
        except ValueError:
            if len(measurement.value) > len(cumulative_height):
                cumulative_height = np.full(len(measurement.value),cumulative_height[0])
            plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)

        if solution == "SERS_BWTeK" or solution == "FT-IR": 
            cumulative_height += 1.5 * max(measurement.value)  # Update cumulative height for next spectrum
    
    if solution == "SERS_BWTeK":
        plt.xlabel("Raman Shift [cm-1]", fontsize = 18)
        plt.ylabel("Normalized Intensity [a.u]", fontsize = 18)
        plt.gca().set_yticklabels([])
    elif solution == "UV-Vis":
        plt.xlabel("Wavelength [nm]", fontsize = 18)
        plt.ylabel("Intensity [a.u]", fontsize = 18)
    elif solution == "FT-IR":
        plt.xlabel("Wavenumber [cm-1]", fontsize = 18)
        plt.ylabel("Transmittance [a.u]", fontsize = 18)
        plt.yticks([])
    else:
        plt.xlabel("Wavelength [nm]", fontsize = 18)
        plt.ylabel("Intensity [a.u]", fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.tick_params(axis='both', direction='in')
    plt.legend(fontsize=18, loc='upper right', frameon=True, framealpha=0.5, edgecolor="black",bbox_to_anchor=(1.1, 1))
    # plt.show()
    plt.savefig(f"{basedir}/plot_{output_name}.png", bbox_inches="tight")

    if solution not in ("UV-Vis","FT-IR","SERS_BWTeK"):
        plt.clf()
        plt.cla()
        cumulative_height = np.zeros(len(data[0].value))
        # Plot unnormalized data
        plt.figure(figsize=(20,14))
        for measurement in data:
            measurement.value = measurement.value * max_ctr_nom
            measurement.std = measurement.std * max_ctr_nom
            # Plot the mean values
            try:
                plt.plot(measurement.wave, measurement.value+cumulative_height[:len(measurement.wave)], label=measurement.alias)
            except ValueError:
                messagebox.showerror("Error", "Please adjust the domain to be plotted !")
            # Plot the upper and lower bounds (mean +/- standard deviation)
            try:
                plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)
            except ValueError:
                if len(measurement.value) > len(cumulative_height):
                    cumulative_height = np.full(len(measurement.value),cumulative_height[0])
                plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)
    
        plt.xlabel("Wavelength [nm]", fontsize = 18)
        plt.ylabel("Intensity [a.u]", fontsize = 18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.tick_params(axis='both', direction='in')
        plt.legend(fontsize=18, loc='upper right', frameon=True, framealpha=0.5, edgecolor="black",bbox_to_anchor=(1.1, 1))
        # plt.show()
        plt.savefig(f"{basedir}/plot_raw_{output_name}.png", bbox_inches="tight")


def read_fluorophor(l):
    documents_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(documents_path, "Spectra Processing","Executable")
    if os.path.exists(folder_path):
        with open(f"{folder_path}/fluorophor_data.txt", "r") as file:
            for line in file:
                l[line.strip().split(',')[0].strip()] = line.strip().split(',')[1].strip()
    else: print("help!")

def updatelist():
    read_fluorophor(fluorophors)
    solution_dropdown["values"]=[f for f in fluorophors.keys()]

def replace_dots_in_filenames(directory):
    for root_dir, _, files in os.walk(directory):
        for filename in files:
            old_path = os.path.join(root_dir, filename)
            # Split filename and extension
            name, ext = os.path.splitext(filename)
            
            # Check for version suffix (e.g., p1.1.txt)
            match = re.search("([0-9].[0-9])$", name)
            print(name,match)
            if match:
                base_name = name[:match.start()]  # Extract the part before the version suffix
                new_name = base_name.replace('.', '_')
                new_name = new_name.replace(' ', '_')
                new_name += match.group(1)
                new_name += ext
            else:
                new_name = name.replace('.', '_')
                new_name = name.replace(' ', '_')
                new_name += ext
            
            new_path = os.path.join(root_dir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f'Renamed: "{old_path}" -> "{new_path}"')

def rename_files():
    folder_path = filedialog.askdirectory()
    if folder_path:
        replace_dots_in_filenames(folder_path)
    
if __name__ == "__main__":
    # Create main application window
    root = tk.Tk()
    root.title("Spectra processing")
    # Variables
    solution_var = tk.StringVar()
    input_files_var = tk.StringVar()
    autofluorescence_var = tk.StringVar()
    min_spectra_var = tk.StringVar()
    max_spectra_var = tk.StringVar()
    name_var = tk.StringVar()
    fluorophors = dict() #for easier search
    fluorophors[''] = "False"
    read_fluorophor(fluorophors)
    # Dictionary to store aliases
    alias_entries = {}    
    
    # Solution dropdown
    tk.Label(root, text="Spectra:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    solution_dropdown = ttk.Combobox(root, textvariable=solution_var, state="readonly", postcommand=updatelist)
    solution_dropdown["values"] = [f for f in fluorophors.keys()]
    solution_dropdown.grid(row=0, column=1, padx=10, pady=5)
    solution_dropdown.bind("<<ComboboxSelected>>", on_solution_change)
    
    # Insert new Fluorophor
    new_spectra_button = ttk.Button(root, text="Add new Spectrometer", command=add_fluorophor)
    new_spectra_button.grid(row=0, column=2, pady=10)

    # Rename files to naming convention
    rename_files_button = ttk.Button(root, text="Rename Files", command=rename_files)
    rename_files_button.grid(row=0, column=3, pady=10)

    # Input files field
    tk.Label(root, text="Input Files:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    input_files_button = ttk.Button(root, text="Select Files", command=select_input_files)
    # input_files_button = ttk.Button(root, text="Select Files", command=select_input_files, postcommand=get_aliases)
    input_files_button.grid(row=1, column=1, padx=10, pady=5)
    input_files_entry = tk.Entry(root, textvariable=input_files_var, state="readonly", width=40)
    input_files_entry.grid(row=1, column=2, padx=10, pady=5, columnspan=2)

    # Autofluorescence field
    tk.Label(root, text="Autofluorescence:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    autofluorescence_button = ttk.Button(root, text="Select Files", command=select_autofluorescence_files)
    autofluorescence_button.grid(row=2, column=1, padx=10, pady=5)
    # autofluorescence_button.config(state="disabled")
    autofluorescence_entry = tk.Entry(root, textvariable=autofluorescence_var, state="disable", width=40)
    autofluorescence_entry.grid(row=2, column=2, padx=10, pady=5, columnspan=2)
    # autofluorescence_entry.config(state="disabled")

    # Spectra limits
    tk.Label(root, text="Spectra Min:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    min_spectra_entry = tk.Entry(root, textvariable=min_spectra_var, width=10)
    min_spectra_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    tk.Label(root, text="Spectra Max:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    max_spectra_entry = tk.Entry(root, textvariable=max_spectra_var, width=10)
    max_spectra_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

    # Peak display checkbox
    peak_display_var = tk.BooleanVar()
    peak_display_checkbox = tk.Checkbutton(root, text="Show Peak Values", variable=peak_display_var)
    peak_display_checkbox.grid(row=3, column=2, padx=10, pady=5, sticky="w")

    # Normalize checkbox
    normalize_var = tk.BooleanVar()
    normalize_checkbox = tk.Checkbutton(root, text="Normalize", variable=normalize_var)
    normalize_checkbox.grid(row=4, column=2, padx=10, pady=5, sticky="w")

    # Denoise checkbox
    denoise_var = tk.BooleanVar()
    denoise_checkbox = tk.Checkbutton(root, text="Denoise", variable=denoise_var)
    denoise_checkbox.grid(row=3, column=3, padx=10, pady=5, sticky="w")

    # Name field
    tk.Label(root, text="Output Name:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
    name_entry = tk.Entry(root, textvariable=name_var, width=40)
    name_entry.grid(row=5, column=1, padx=10, pady=5, columnspan=2)

    # Submit button
    submit_button = ttk.Button(root, text="Submit", command=submit)
    submit_button.grid(row=6, column=0, columnspan=3, pady=10)

    # Set initial state for autofluorescence
    on_solution_change(None)

    # Start the application
    root.mainloop()
    # process_files()
    # "C:\Users\smark\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\pyinstaller.exe" --onefile --noconsole ..\spectraProcessing.py
    