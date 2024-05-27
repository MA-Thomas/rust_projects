import immunogenicity_rust 

import argparse
import os
import pickle
import copy 

import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

import time 

def create_evenly_spaced_list(L, U, N):
    return np.linspace(L, U, N).tolist()
def create_log_spaced_list(L, U, N):
    return np.logspace(np.log10(L), np.log10(U), N).tolist()


if __name__ == "__main__":

    # # LOCAL
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-cTEC", action='store_true', default=False, help="whether to use cTEC gene expression")
    parser.add_argument("-cTEC_conc", action='store_true', default=False)
    parser.add_argument("-save_pkl_dir", default="/Users/marcus/Work_Data/rust_outputs_local/immunogenicity_outputs")
    parser.add_argument("-distance_metric_type", default="all_tcr_all_combos_model")

    parser.add_argument("-data_matrix_dir", default="/Users/marcus/Work_Data/Conifold_editing/CFIT/cfit/data/matrices/")
    parser.add_argument("-self_fasta_path", default="/Users/marcus/Work_Data/Self_Epitopes/self_peptides.fasta")
    parser.add_argument("-immunogenic_Koncz_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Koncz_peptides.fasta")
    parser.add_argument("-nonimmunogenic_Koncz_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Koncz_peptides.fasta")
    parser.add_argument("-immunogenic_Ours_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Ours_peptides.fasta")
    parser.add_argument("-nonimmunogenic_Ours_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Ours_peptides.fasta")

    parser.add_argument("-csv_S_dir", default="/Users/marcus/Work_Data/Self_Epitopes")
    parser.add_argument("-csv_F_dir", default="/Users/marcus/Work_Data/Foreign_Epitopes")

    parser.add_argument("-tesla_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/TESLA")
    parser.add_argument("-iedb_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/IEDB")
    
    parser.add_argument("-allele", default='A0201')

    parser.add_argument("-inclusive_start_ind", default="0")
    parser.add_argument("-inclusive_end_ind", default="0")
    args = parser.parse_args()


    

    '''
    distance metric types (ninemer) 
    ['all_tcr_all_combos_model',
     'hamming',
     'epidist_blosum62_distance']
    
    '''
    
    '''
    Load TESLA and IEDB data separately generated in /TESLA and /IEDB.
    This data consists of the 2 dataframes (merged into a single dataframe here)
    as well as the auxilliary variables like dictionaries

    '''
    #epidist_blosum62_distance #all_tcr_all_combos_model #hamming
    distance_metric = "all_tcr_all_combos_model" 
    max_target_num = 20000000 

    save_query_distance_files = False 
    load_precomputed_distances = False 

    inclusive_start_ind = int(args.inclusive_start_ind)
    inclusive_end_ind = int(args.inclusive_end_ind) 

    data_matrix_dir = args.data_matrix_dir
    self_fasta_path = args.self_fasta_path

    immunogenic_Koncz_fasta_path = args.immunogenic_Koncz_fasta_path
    nonimmunogenic_Koncz_fasta_path = args.nonimmunogenic_Koncz_fasta_path

    immunogenic_Ours_fasta_path = args.immunogenic_Ours_fasta_path
    nonimmunogenic_Ours_fasta_path = args.nonimmunogenic_Ours_fasta_path


    start_time = time.time()

    with open(os.path.join(args.tesla_variables_dir, 'TESLA_variables_'+args.distance_metric_type+'.pkl'),'rb') as pyt:
        W=pickle.load(pyt)
        
    tesla_data=copy.deepcopy(W[0]) 
    tesla_data['assay'] = 'TESLA'

    with open(os.path.join(args.iedb_variables_dir, 'Koncz_IEDB_variables_'+args.distance_metric_type+'.pkl'),'rb') as pyt:
        V=pickle.load(pyt)
        
    iedb_data=V[0] 
    iedb_data['assay'] = 'IEDB'

    # # COMBINED DATASET
    tesla_iedb_data = pd.concat([tesla_data, iedb_data])
    tesla_iedb_data = tesla_iedb_data.sort_values(by='peptides')

    allele = args.allele 
    hla = "HLA-"+allele

    # pos_df = tesla_iedb_data[(tesla_iedb_data['immunogenicity'] == 1) & (tesla_iedb_data['allele'] == allele)]
    # neg_df = tesla_iedb_data[(tesla_iedb_data['immunogenicity'] == 0) & (tesla_iedb_data['allele'] == allele)]

    hla_df = tesla_iedb_data[tesla_iedb_data['allele'] == allele]

    
    csv_S_kds_file = os.path.join(args.csv_S_dir, "Kds/self_epitopes_Kd_values_"+hla+".csv")
    csv_KImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Koncz_Imm/epitopes_Kd_values_"+hla+".csv")
    csv_KNImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Koncz_nonImm/epitopes_Kd_values_"+hla+".csv")
    csv_OImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Ours_Imm/epitopes_Kd_values_"+hla+".csv")
    csv_ONImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Ours_nonImm/epitopes_Kd_values_"+hla+".csv")
    
    for idx, (row_index,row) in enumerate(hla_df.iterrows()):
        print("idx = ",idx)
        if idx < inclusive_start_ind:
            continue
        if idx > inclusive_end_ind:
            break


        allele = row['allele']
        query_epi_list = row['peptides']
        print(query_epi_list, allele)

        if save_query_distance_files:
            csv_S_dists_file = os.path.join(args.csv_S_dir, "Distances/QUERYEPISTR_self_distances.csv")
            csv_KImm_dists_file = os.path.join(args.csv_S_dir, "Distances/QUERYEPISTR_Koncz_imm_distances.csv")
            csv_KNImm_dists_file = os.path.join(args.csv_S_dir, "Distances/QUERYEPISTR_Koncz_nonimm_distances.csv")
            csv_OImm_dists_file = os.path.join(args.csv_S_dir, "Distances/QUERYEPISTR_Ours_imm_distances.csv")
            csv_ONImm_dists_file = os.path.join(args.csv_S_dir, "Distances/QUERYEPISTR_Ours_nonimm_distances.csv")
        else:
            csv_S_dists_file = ""
            csv_KImm_dists_file = ""
            csv_KNImm_dists_file = ""
            csv_OImm_dists_file = ""
            csv_ONImm_dists_file = ""                


        
        compute_logKinv_and_entropy = True

        # # PARAMETER SETS FOR ZACH'S METRIC
        gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 12),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 3),4)) )))
        gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  )))

        gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 12),10)) + list(np.round(create_log_spaced_list(1, 5, 15),4)) )))
        gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  )))


        #################################################################################################
        #################################################################################################
        ##                                 Self settings.
        compute_logCh = True
        d_PS_threshold = 70.0 # # distances above this threshold do not contirbute to Kinv_self (models positive selection)
        d_NS_cutoff = 3.14 # # distances below this cutoff do not contribute to Kinv_self (partially models negative selection)
        #################################################################################################
        #################################################################################################

        logKInv_entropy_self_dict, logCh_dict, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_S_dists_file,load_precomputed_distances,self_fasta_path)],csv_S_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_self_values,
            gamma_logkd_self_values,
            d_PS_threshold, d_NS_cutoff,
            compute_logKinv_and_entropy,
            compute_logCh,
        )
        print("len(logKInv_entropy_self_dict): ",len(logKInv_entropy_self_dict))
        print("A few items from logKInv_entropy_self_dict: ")
        for key, value in itertools.islice(logKInv_entropy_self_dict.items(), 3):
            print(f"{key}: {value}")
        print("[python] logKInv_entropy_self_dict/logCh_dict runtime: ",runtime)

        print("len(logCh_dict): ",len(logCh_dict))
        print("A few items from logKIlogCh_dictnv_entropy_self_dict: ")
        for key, value in itertools.islice(logCh_dict.items(), 5):
            print(f"{key}: {value}")

        #################################################################################################
        #################################################################################################
        ##                                 Foreign settings.
        compute_logCh = False
        d_PS_threshold = 1e10 # # not relevant for K_inv evaluated on foreign target set
        d_NS_cutoff = 0 # # not relevant for K_inv evaluated on foreign target set
        #################################################################################################
        #################################################################################################

        logKInv_entropy_Koncz_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_KImm_dists_file,load_precomputed_distances,immunogenic_Koncz_fasta_path)], csv_KImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold, d_NS_cutoff,
            compute_logKinv_and_entropy,
            compute_logCh,
        )
        print("len(logKInv_entropy_Koncz_imm_epi_dict): ",len(logKInv_entropy_Koncz_imm_epi_dict))
        print("A few items from logKInv_entropy_Koncz_imm_epi_dict: ")
        for key, value in itertools.islice(logKInv_entropy_Koncz_imm_epi_dict.items(), 5):
            print(f"{key}: {value}")
        print("[python] Koncz_imm runtime: ",runtime)

        logKInv_entropy_Koncz_non_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_KNImm_dists_file,load_precomputed_distances,nonimmunogenic_Koncz_fasta_path)], csv_KNImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold, d_NS_cutoff,
            compute_logKinv_and_entropy,
            compute_logCh,
        )
        print("[python] Koncz_non_imm runtime: ",runtime)

        logKInv_entropy_Ours_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_OImm_dists_file,load_precomputed_distances,immunogenic_Ours_fasta_path)], csv_OImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold, d_NS_cutoff,
            compute_logKinv_and_entropy,
            compute_logCh,
        )
        print("[python] Ours_imm runtime: ",runtime)

        logKInv_entropy_Ours_non_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_ONImm_dists_file,load_precomputed_distances,nonimmunogenic_Ours_fasta_path)], csv_ONImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold, d_NS_cutoff,
            compute_logKinv_and_entropy,
            compute_logCh,
        )
        print("[python] Ours_non_imm runtime: ",runtime)

        ##  --------------  COMPUTE rho  ----------------
        use_Ours_contribution = True 
        use_Koncz_contribution = True

        log_rho_dict, runtime = immunogenicity_rust.compute_log_rho_multi_query_py(
            logKInv_entropy_self_dict, 
            logKInv_entropy_Ours_imm_epi_dict,
            logKInv_entropy_Ours_non_imm_epi_dict, 
            use_Ours_contribution,
            logKInv_entropy_Koncz_imm_epi_dict,
            logKInv_entropy_Koncz_non_imm_epi_dict, 
            use_Koncz_contribution,
        )
        print("[python] log_rho_dict runtime: ",runtime)
        # print("len(log_rho_dict): ",len(log_rho_dict))
        # print("One item from log_rho_dict: ")
        # for key, value in itertools.islice(log_rho_dict.items(), 1):
        #     for key2, value2 in itertools.islice(log_rho_dict[key].items(), 1):
        #         print(f"{key}:{key2}: {value2}")

        print("Now saving to .pkl")
        outputs = [logKInv_entropy_self_dict, logCh_dict, logKInv_entropy_Koncz_imm_epi_dict, logKInv_entropy_Koncz_non_imm_epi_dict, 
                   logKInv_entropy_Ours_imm_epi_dict, logKInv_entropy_Ours_non_imm_epi_dict,
                   log_rho_dict]
        
        strg = 'immunogenicity_outputs_'+distance_metric+'_'+str(idx)+'_'
        for elem in query_epi_list:
            strg += elem + '_'
        strg += hla 
        strg += '.pkl'
        with open(os.path.join(args.save_pkl_dir,strg), 'wb') as pk:
            pickle.dump(outputs, pk)

    print("[python] script runtime: ", time.time()-start_time)
