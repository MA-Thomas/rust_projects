import argparse
import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import time
import ast

import immunogenicity_rust



def create_evenly_spaced_list(L, U, N):
    return np.linspace(L, U, N).tolist()

def create_log_spaced_list(L, U, N):
    return np.logspace(np.log10(L), np.log10(U), N).tolist()


def create_evenly_spaced_list(L, U, N):
    return np.linspace(L, U, N).tolist()
def create_log_spaced_list(L, U, N):
    return np.logspace(np.log10(L), np.log10(U), N).tolist()


def create_subfolders(file_path):
    """
    Creates all necessary subfolders in the given file path if they don't exist.

    Parameters:
    file_path (str): The full path where subfolders need to be created.
    """
    # Get the directory path from the file path
    directory_path = os.path.dirname(file_path)

    # Create the directories if they do not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directories: {directory_path}")
    else:
        print(f"Directories already exist: {directory_path}")



# # The parameter sets used in Rust.
gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 7),4)) )))
gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 12),4))  )))

gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(1e-1, 5, 6),4)) + [1e-8, 1e-100])))
gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 10),4))  +[1e-8, 1e-100])))
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-base_directory", default="/sc/arion/projects/FLAI/marcus/PDAC_Rust_Results/d_ub_100_d_lb_0")
    parser.add_argument("-counter", type=int, default=0) # The index of the self param set

    args = parser.parse_args()

    start_time = time.time()

    ##%% GATHER PATHS TO PICKLE FILES STORING IMMUNOGENICITY INFO
    pkl_list = []

    # Define the pattern for the folder names
    pattern = re.compile(r'HLA-[A-Z]\d{2}:\d{2}')

    # Iterate over all items in the base directory
    for item in os.listdir(args.base_directory):
        item_path = os.path.join(args.base_directory, item)
        # Check if the item is a directory and matches the pattern
        if os.path.isdir(item_path) and pattern.match(item):
            print(f"Processing folder: {item}")

            # Add your code to process the folder here
            # For example, you can list files within the folder
            for file_name in os.listdir(item_path):
                if file_name.endswith('.pkl'):
                    pkl_file_path = os.path.join(item_path, file_name)
                    pkl_list.append(pkl_file_path)
                    print(f"  Appended file: {file_name}")

###############################################################################
    # # Generate sep Imm_dict for each self parameter set (each using all .pkls)
    print("[python] Now to generate sep Imm_dict for each self parameter set (each using all .pkls) ...", flush=True)

    # # IT'S IMPORTANT TO FORMAT THE STRINGS CONTAINING THE SELF PARAMET CORRECTLY.
    import re
    def remove_leading_zeros_scientific_notation(input_str):
        if 'e' in input_str:
            # Use regular expression to find and remove leading zeros after 'e'
            return re.sub(r'e-0+', 'e-', input_str)
        else:
            return input_str  # Return input_str unchanged if 'e' is not present

    counter = 0
    for gamma_d_self in gamma_d_self_values:
        for gamma_logKd_self in gamma_logkd_self_values:
            if counter == args.counter:
                s1 = remove_leading_zeros_scientific_notation(str(gamma_d_self))
                s2 = remove_leading_zeros_scientific_notation(str(gamma_logKd_self))

                print("[python] processing self params: ",s1,s2)
                rst1 = time.time()
                imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(pkl_list, (), s1, s2)

                first_few_items = list(imm_dict.items())[:3]  # Change the slice range as needed
                # print("[python] first few items: ", first_few_items)

                # # Save dict as pkl file.
                save_name = 'Imm_dict__gamma_d_self_'+s1+'_gamma_logKd_self_'+s2+'.pkl'
                save_pkl = os.path.join(args.base_directory,'Immunogenicity_Dicts',save_name)
                create_subfolders(save_pkl)
                with open(save_pkl,"wb") as pk:
                    pickle.dump(imm_dict, pk)
                print("[python] Time to generate/load/save immunogenicity_dict at current self param: ", time.time()-rst1, flush=True)
            counter += 1

    print("[python] Time to generate/load/save all immunogenicity_dicts from rust: ", time.time()-start_time, flush=True)






###############################################################################
