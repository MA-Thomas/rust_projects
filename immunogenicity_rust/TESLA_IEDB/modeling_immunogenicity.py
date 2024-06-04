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
HLAA = ['HLA-A0101', 'HLA-A0201', 'HLA-A0202', 'HLA-A0203', 'HLA-A0205', 'HLA-A0206', 'HLA-A0207', 'HLA-A0211',
        'HLA-A0212', 'HLA-A0216', 'HLA-A0217', 'HLA-A0219', 'HLA-A0250', 'HLA-A0301', 'HLA-A0302', 'HLA-A0319',
        'HLA-A1101', 'HLA-A2301', 'HLA-A2402', 'HLA-A2403', 'HLA-A2501', 'HLA-A2601', 'HLA-A2602', 'HLA-A2603',
        'HLA-A2902', 'HLA-A3001', 'HLA-A3002', 'HLA-A3101', 'HLA-A3201', 'HLA-A3207', 'HLA-A3215', 'HLA-A3301',
        'HLA-A6601', 'HLA-A6801', 'HLA-A6802', 'HLA-A6823', 'HLA-A6901', 'HLA-A8001']

HLAB = ['HLA-B0702', 'HLA-B0801', 'HLA-B0802', 'HLA-B0803', 'HLA-B1401', 'HLA-B1402', 'HLA-B1501', 'HLA-B1502',
        'HLA-B1503', 'HLA-B1509', 'HLA-B1517', 'HLA-B1801', 'HLA-B2705', 'HLA-B2720', 'HLA-B3501', 'HLA-B3503',
        'HLA-B3701', 'HLA-B3801', 'HLA-B3901', 'HLA-B4001', 'HLA-B4002', 'HLA-B4013', 'HLA-B4201', 'HLA-B4402',
        'HLA-B4403', 'HLA-B4501', 'HLA-B4506', 'HLA-B4601', 'HLA-B4801', 'HLA-B5101', 'HLA-B5301', 'HLA-B5401',
        'HLA-B5701', 'HLA-B5703', 'HLA-B5801', 'HLA-B5802', 'HLA-B7301', 'HLA-B8101', 'HLA-B8301']

HLAC = ['HLA-C0303', 'HLA-C0401', 'HLA-C0501', 'HLA-C0602', 'HLA-C0701', 'HLA-C0702', 'HLA-C0802', 'HLA-C1203',
        'HLA-C1402', 'HLA-C1502']

HLA = HLAA + HLAB + HLAC

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

    parser.add_argument("-immunogenic_Tesla_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Tesla_peptides.fasta")
    parser.add_argument("-nonimmunogenic_Tesla_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Tesla_peptides.fasta")

    parser.add_argument("-csv_S_dir", default="/Users/marcus/Work_Data/Self_Epitopes")
    parser.add_argument("-csv_F_dir", default="/Users/marcus/Work_Data/Foreign_Epitopes")

    parser.add_argument("-tesla_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/TESLA")
    parser.add_argument("-iedb_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/IEDB")
    
    parser.add_argument("-load_target_hla_epi_dir", default="/Users/marcus/Work_Data/Foreign_Epitopes")

    parser.add_argument("-allele", default='A2301')

    parser.add_argument("-inclusive_start_ind", default="0")
    parser.add_argument("-inclusive_end_ind", default="0")
    args = parser.parse_args()



    
    '''
    TODO: 
    1. Test new code that restricts target epitope sets based on hla.
    2. Add tesla target epitope sets to rust functions

    '''

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
    distance_metric = args.distance_metric_type #"all_tcr_all_combos_model" 
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

    immunogenic_Tesla_fasta_path = args.immunogenic_Tesla_fasta_path
    nonimmunogenic_Tesla_fasta_path = args.nonimmunogenic_Tesla_fasta_path

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
    csv_TImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Tesla_Imm/epitopes_Kd_values_"+hla+".csv")
    csv_TNImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Tesla_nonImm/epitopes_Kd_values_"+hla+".csv")

    with open(os.path.join(args.load_target_hla_epi_dir,'target_hla_epi_dict.pkl'),'rb') as pot:
        target_hla_epi_dict = pickle.load(pot)


    for idx, (df_index,row) in enumerate(hla_df.iterrows()):
        print("idx = ",idx, flush=True)
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
            csv_TImm_dists_file = os.path.join(args.csv_S_dir, "Distances/QUERYEPISTR_Tesla_imm_distances.csv")
            csv_TNImm_dists_file = os.path.join(args.csv_S_dir, "Distances/QUERYEPISTR_Tesla_nonimm_distances.csv")
        else:
            csv_S_dists_file = ""
            csv_KImm_dists_file = ""
            csv_KNImm_dists_file = ""
            csv_OImm_dists_file = ""
            csv_ONImm_dists_file = "" 
            csv_TImm_dists_file = ""
            csv_TNImm_dists_file = ""         


        
        compute_logKinv_and_entropy = True

        # # PARAMETER SETS FOR ZACH'S METRIC
        # gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 12),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 3),4)) )))
        # gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  )))

        # gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 12),10)) + list(np.round(create_log_spaced_list(1, 5, 15),4)) + [1e-8, 1e-100])))
        # gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  +[1e-8, 1e-100])))


        gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 7),4)) )))
        gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  )))

        gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 10),10)) + list(np.round(create_log_spaced_list(1, 5, 8),4)) + [1e-8, 1e-100])))
        gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 15),4))  +[1e-8, 1e-100])))
        #################################################################################################
        #################################################################################################
        ##                                 Self settings.
        compute_logCh = True
        d_PS_threshold = 70.0 # # distances above this threshold do not contirbute to Kinv_self (models positive selection)
        d_NS_cutoff = 3.14 # # distances below this cutoff do not contribute to Kinv_self (partially models negative selection)
        #################################################################################################
        #################################################################################################

        target_epitopes_at_allele = ['all']
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
            target_epitopes_at_allele,
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
        d_PS_threshold_foreign = 1e10 # # not relevant for K_inv evaluated on foreign target set
        d_NS_cutoff_foreign = 0 # # not relevant for K_inv evaluated on foreign target set

        '''
        When the target set is the self-epitope set, all self epitopes are used to compute KInv.
        When the target set is IEDB+ and IEDB- or TESLA+ or TESLA-, each of these target sets should be hla dependent.
        The fasta files (e.g., immunogenic_Koncz_fasta_path, nonimmunogenic_Koncz_fasta_path, etc.) are not hla dependent.
        We therefore need to additionally pass as an argument the list of target epitopes corresponding to the current hla.
        The target epitopes are not necessarily 9mers. 10 and 11 mers (and maybe longer) are included.
        The Kds for target epitopes are also for the full Nmers.
        This is ok because my distance function returns the minimal 9mer distance.
        '''
        #################################################################################################
        #################################################################################################
        if allele in target_hla_epi_dict['koncz_imm']:
            target_epitopes_at_allele = target_hla_epi_dict['koncz_imm'][allele]
        else:
            target_epitopes_at_allele = []
        logKInv_entropy_Koncz_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_KImm_dists_file,load_precomputed_distances,immunogenic_Koncz_fasta_path)], csv_KImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold_foreign, d_NS_cutoff_foreign,
            compute_logKinv_and_entropy,
            compute_logCh,
            target_epitopes_at_allele,
        )
        print("len(logKInv_entropy_Koncz_imm_epi_dict): ",len(logKInv_entropy_Koncz_imm_epi_dict))
        print("A few items from logKInv_entropy_Koncz_imm_epi_dict: ")
        for key, value in itertools.islice(logKInv_entropy_Koncz_imm_epi_dict.items(), 5):
            print(f"{key}: {value}")
        print("[python] Koncz_imm runtime: ",runtime)

        if allele in target_hla_epi_dict['koncz_nonimm']:
            target_epitopes_at_allele = target_hla_epi_dict['koncz_nonimm'][allele]
        else:
            target_epitopes_at_allele = [] 
        logKInv_entropy_Koncz_non_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_KNImm_dists_file,load_precomputed_distances,nonimmunogenic_Koncz_fasta_path)], csv_KNImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold_foreign, d_NS_cutoff_foreign,
            compute_logKinv_and_entropy,
            compute_logCh,
            target_epitopes_at_allele,
        )
        print("[python] Koncz_non_imm runtime: ",runtime)

        if allele in target_hla_epi_dict['ours_imm']:
            target_epitopes_at_allele = target_hla_epi_dict['ours_imm'][allele]
        else:
            target_epitopes_at_allele = [] 
        logKInv_entropy_Ours_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_OImm_dists_file,load_precomputed_distances,immunogenic_Ours_fasta_path)], csv_OImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold_foreign, d_NS_cutoff_foreign,
            compute_logKinv_and_entropy,
            compute_logCh,
            target_epitopes_at_allele,
        )
        print("[python] Ours_imm runtime: ",runtime)

        if allele in target_hla_epi_dict['ours_nonimm']:
            target_epitopes_at_allele = target_hla_epi_dict['ours_nonimm'][allele]
        else:
            target_epitopes_at_allele = []
        logKInv_entropy_Ours_non_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_ONImm_dists_file,load_precomputed_distances,nonimmunogenic_Ours_fasta_path)], csv_ONImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold_foreign, d_NS_cutoff_foreign,
            compute_logKinv_and_entropy,
            compute_logCh,
            target_epitopes_at_allele,
        )
        print("[python] Ours_non_imm runtime: ",runtime)

        if allele in target_hla_epi_dict['tesla_imm']:
            target_epitopes_at_allele = target_hla_epi_dict['tesla_imm'][allele]
        else:
            target_epitopes_at_allele = []
        logKInv_entropy_Tesla_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_TImm_dists_file,load_precomputed_distances,immunogenic_Tesla_fasta_path)], csv_TImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold_foreign, d_NS_cutoff_foreign,
            compute_logKinv_and_entropy,
            compute_logCh,
            target_epitopes_at_allele,
        )
        print("[python] Tesla_imm runtime: ",runtime)

        if allele in target_hla_epi_dict['tesla_nonimm']:
            target_epitopes_at_allele = target_hla_epi_dict['tesla_nonimm'][allele]
        else:
            target_epitopes_at_allele = []
        logKInv_entropy_Tesla_non_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list, 
            [(csv_TNImm_dists_file,load_precomputed_distances,nonimmunogenic_Tesla_fasta_path)], csv_TNImm_kds_file,
            distance_metric, 
            data_matrix_dir, 
            max_target_num,
            gamma_d_nonself_values,
            gamma_logkd_nonself_values,
            d_PS_threshold_foreign, d_NS_cutoff_foreign,
            compute_logKinv_and_entropy,
            compute_logCh,
            target_epitopes_at_allele,
        )
        print("[python] Tesla_non_imm runtime: ",runtime)
        ##  --------------  COMPUTE rho  ----------------
        use_Ours_contribution = True 
        use_Koncz_contribution = True
        use_Tesla_contribution = True 

        log_rho_dict, runtime = immunogenicity_rust.compute_log_rho_multi_query_py(
            logKInv_entropy_self_dict, 
            logKInv_entropy_Ours_imm_epi_dict,
            logKInv_entropy_Ours_non_imm_epi_dict, 
            use_Ours_contribution,
            logKInv_entropy_Koncz_imm_epi_dict,
            logKInv_entropy_Koncz_non_imm_epi_dict, 
            use_Koncz_contribution,
            logKInv_entropy_Tesla_imm_epi_dict,
            logKInv_entropy_Tesla_non_imm_epi_dict, 
            use_Tesla_contribution,
        )
        print("[python] log_rho_dict runtime: ",runtime)
        # print("len(log_rho_dict): ",len(log_rho_dict))
        # print("One item from log_rho_dict: ")
        # for key, value in itertools.islice(log_rho_dict.items(), 1):
        #     for key2, value2 in itertools.islice(log_rho_dict[key].items(), 1):
        #         print(f"{key}:{key2}: {value2}")

        # immunogenicity_rust.store_model_dicts(
        #     logKInv_entropy_self_dict,
        #     logKInv_entropy_Koncz_imm_epi_dict,
        #     logKInv_entropy_Koncz_non_imm_epi_dict,
        #     logKInv_entropy_Ours_imm_epi_dict,
        #     logKInv_entropy_Ours_non_imm_epi_dict,
        #     logCh_dict,
        #     log_rho_dict,
        # )
        print("Now saving to .pkl")
        outputs = [logKInv_entropy_self_dict, logCh_dict, logKInv_entropy_Koncz_imm_epi_dict, logKInv_entropy_Koncz_non_imm_epi_dict, 
                   logKInv_entropy_Ours_imm_epi_dict, logKInv_entropy_Ours_non_imm_epi_dict,
                   log_rho_dict]
        
        args.save_pkl_dir = args.save_pkl_dir + '/d_PS_threshold_'+str(d_PS_threshold)+'_d_NS_cutoff_'+str(d_NS_cutoff)+'/'+allele
        if not os.path.exists(args.save_pkl_dir):
            os.makedirs(args.save_pkl_dir)

        strg = 'immunogenicity_outputs_'+distance_metric+'_'+str(idx)+'_'
        for elem in query_epi_list:
            strg += elem + '_'
        strg += hla 
        strg += '.pkl'
        with open(os.path.join(args.save_pkl_dir,strg), 'wb') as pk:
            pickle.dump(outputs, pk)

    print("[python] script runtime: ", time.time()-start_time)
