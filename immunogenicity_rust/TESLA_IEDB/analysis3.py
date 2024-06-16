import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import immunogenicity_rust 

def full_pep_from_ninemers(epi_list):
    if not epi_list:
        return ""
    full_peptide = epi_list[0]
    for i in range(1, len(epi_list)):
        full_peptide += epi_list[i][-1]
    return full_peptide

def get_nth_element(input_str, n):
    elements = input_str.strip('()').split(',')
    return elements[n].strip()

def string_to_numeric_tuple(input_str):
    elements = input_str.strip('()').split(',')
    return tuple(map(float, elements))

def create_evenly_spaced_list(L, U, N):
    return np.linspace(L, U, N).tolist()

def create_log_spaced_list(L, U, N):
    return np.logspace(np.log10(L), np.log10(U), N).tolist()

def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_and_preprocess_data(args):
    # Batched loading
    tesla_files = [os.path.join(args.tesla_variables_dir, f'TESLA_variables_{args.distance_metric_type}.pkl')]
    iedb_files = [os.path.join(args.iedb_variables_dir, f'Koncz_IEDB_variables_{args.distance_metric_type}.pkl')]
    
    # Load and preprocess data in batches
    tesla_data_batches = [load_data(file) for file in tesla_files]
    iedb_data_batches = [load_data(file) for file in iedb_files]

    # Combine and preprocess batches
    combined_tesla_data = pd.concat([batch[0] for batch in tesla_data_batches])
    combined_iedb_data = pd.concat([batch[0] for batch in iedb_data_batches])

    combined_tesla_data['assay'] = 'TESLA'
    combined_iedb_data['assay'] = 'IEDB'

    combined_data = pd.concat([combined_tesla_data, combined_iedb_data]).sort_values(by='peptides')
    hla_df = combined_data[combined_data['allele'] == args.allele]

    return hla_df

def process_row(idx, row, args, start_time):
    nb_records = []
    # print("intermediate time 1: ", time.time() - start_time)
    query_epi_list = row['peptides']
    full_query_epi = full_pep_from_ninemers(query_epi_list)
    strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
    inputs = load_data(os.path.join(args.pkl_dir, args.allele, strg))

    logKInv_entropy_self_dict, logCh_dict, log_rho_dict = inputs[0], inputs[1], inputs[6]
    # print("intermediate time 2: ", time.time() - start_time)
    for query_epi in log_rho_dict:
        for SELF_params, foreign_dict in log_rho_dict[query_epi].items():
            SELF_params_numeric = string_to_numeric_tuple(SELF_params)
            gamma_logKd_str = get_nth_element(SELF_params, 1)
            log_Ch = logCh_dict[gamma_logKd_str if gamma_logKd_str != '1.0' else '1']
            log_non_rho = log_Ch + logKInv_entropy_self_dict[query_epi][SELF_params][0] - logKInv_entropy_self_dict[query_epi][SELF_params][1]

            for FOREIGN_params, log_rho in foreign_dict.items():
                FOREIGN_params_numeric = string_to_numeric_tuple(FOREIGN_params)
                nb = np.exp(log_non_rho + log_rho)
                nb_records.append((full_query_epi, row['assay'], row['immunogenicity'], SELF_params_numeric, FOREIGN_params_numeric, nb))
    return nb_records

def calculate_auc(pos, neg):
    y = np.array([2] * len(pos) + [1] * len(neg))
    pred = np.concatenate([pos, neg])
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=2)
    return round(metrics.auc(fpr, tpr), 12)

# The parameter sets used in Rust.
gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 12),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 3),4)) )))
gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  )))

gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 12),10)) + list(np.round(create_log_spaced_list(1, 5, 15),4)) )))
gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  )))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-pkl_dir", default="/Users/marcus/Work_Data/rust_outputs_local/immunogenicity_outputs/d_ub_70.0_d_lb_3.14")
    parser.add_argument("-distance_metric_type", default="all_tcr_all_combos_model")
    parser.add_argument("-tesla_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/TESLA")
    parser.add_argument("-iedb_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/IEDB")
    parser.add_argument("-allele", default='A0301')
    parser.add_argument("-max_workers", type=int, default=12, help="Maximum number of worker threads")
    args = parser.parse_args()

    start_time = time.time()
    hla_df = load_and_preprocess_data(args)
    
    # hla_df = hla_df[:40]
    
    nb_tesla_records = []
    nb_iedb_records = [] 
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_row, idx, row, args, start_time): idx for idx, (df_index,row) in enumerate(hla_df.iterrows())}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result[0][1] == 'TESLA':
                    nb_tesla_records.extend(result)
                else:
                    nb_iedb_records.extend(result)
            except Exception as e:
                print(f"Error processing row {futures[future]}: {e}")

    print("loading runtime:", time.time() - start_time)
    
#%%  COMPUTE AUCs IN RUST
    start_time2 = time.time()
    tesla_auc_dict = immunogenicity_rust.calculate_auc_dict(nb_tesla_records)
    iedb_auc_dict = immunogenicity_rust.calculate_auc_dict(nb_iedb_records)
    print("rust auc_dict runtime:", time.time() - start_time2)
    # assert(1==2)
#%%

    start_time3 = time.time()
    keys_ranking_list = []
    for key in tesla_auc_dict:
        for key2 in tesla_auc_dict[key]:
            # Sort the values (lists) within each key based on the fourth element (koncz_auc)
            tesla_auc_dict[key][key2].sort(key=lambda x: x[1], reverse=True)

            # Convert the list to a DataFrame
            tesla_df = pd.DataFrame(tesla_auc_dict[key][key2], columns=['gamma_d_nonself', 'tesla_auc', 'koncz_auc'])

            iedb_auc_dict[key][key2].sort(key=lambda x: x[1], reverse=True)

            # Convert the list to a DataFrame
            iedb_df = pd.DataFrame(iedb_auc_dict[key][key2], columns=['gamma_d_nonself', 'tesla_auc', 'koncz_auc'])

 
            '''
            Provide a ranking for [key=gamma_logKd_nonself, key2=SELF_params].
            A higher ranking implies that:  
            a particular gamma_d_nonself value does very well on tesla, 
            a possibly different gamma_d_nonself value does very well on iedb, 
            both in the context of [key=gamma_logKd_nonself, key2=SELF_params]
            - this is achieved with the .max() - 
            Additionally, rankings are improved if all gamma_d_nonself values  
            do well for both tesla and iedb
            - this is achieved with the .mean() -
            
            '''
            
            ranking_statistic = tesla_df['tesla_auc'].max() * tesla_df['koncz_auc'].max() * iedb_df['tesla_auc'].mean() * iedb_df['koncz_auc'].mean()
            
            keys_ranking_list.append([key,key2,ranking_statistic])

                
    keys_ranking_list.sort(key=lambda x: x[2], reverse=True)
    print("keys_ranking_list[:10]: ", keys_ranking_list[:10])
    print("key_list runtime:", time.time() - start_time3)
#%%

