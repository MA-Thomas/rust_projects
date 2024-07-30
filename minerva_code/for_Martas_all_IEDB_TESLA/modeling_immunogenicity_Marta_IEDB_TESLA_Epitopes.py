import immunogenicity_rust

import argparse
import os
import pickle
import copy
import tarfile

import csv

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
def derive_original_peptide(nmer_list):
    if not nmer_list:
        return ""

    # Start with the first nmer
    original_peptide = nmer_list[0]

    # Append the last character of each subsequent nmer to the original_peptide
    for nmer in nmer_list[1:]:
        original_peptide += nmer[-1]

    return original_peptide
def load_neo_hla_tuples_from_csv(filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader, None)
        # Read the rows into a list of tuples
        tuple_list = [tuple(row) for row in reader]

    return tuple_list

def file_already_saved(file_path):
    if os.path.isfile(file_path):
        # Check if the file size is greater than 0 bytes
        if os.path.getsize(file_path) > 0:
            print(f"The file '{file_path}' exists and is not empty.")
            return True
        else:
            print(f"The file '{file_path}' exists but is empty.")
            return False
    else:
        print(f"The file '{file_path}' does not exist.")
        return False



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

'''
Marta gave me a file that contains all of IEDB and all of Tesla (just the 9mers from each, longer peptides excluded).
She wanted me to compute KInv_self for these across parameter sets.

It wouldn't really make sense to compute the full nb for these.
'''
if __name__ == "__main__":

    # # LOCAL
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # # parser.add_argument("-cTEC", action='store_true', default=False, help="whether to use cTEC gene expression")
    # # parser.add_argument("-cTEC_conc", action='store_true', default=False)
    # parser.add_argument("-save_pkl_dir", default="/Users/marcus/Work_Data/rust_outputs_local/immunogenicity_outputs")
    # parser.add_argument("-distance_metric_type", default="all_tcr_all_combos_model")
    #
    # parser.add_argument("-data_matrix_dir", default="/Users/marcus/Work_Data/Conifold_editing/CFIT/cfit/data/matrices/")
    # parser.add_argument("-self_fasta_path", default="/Users/marcus/Work_Data/Self_Epitopes/self_peptides.fasta")
    # parser.add_argument("-immunogenic_Koncz_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Koncz_peptides.fasta")
    # parser.add_argument("-nonimmunogenic_Koncz_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Koncz_peptides.fasta")
    # parser.add_argument("-immunogenic_Ours_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Ours_peptides.fasta")
    # parser.add_argument("-nonimmunogenic_Ours_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Ours_peptides.fasta")
    #
    # parser.add_argument("-immunogenic_Tesla_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Tesla_peptides.fasta")
    # parser.add_argument("-nonimmunogenic_Tesla_fasta_path", default="/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Tesla_peptides.fasta")
    #
    # parser.add_argument("-csv_S_dir", default="/Users/marcus/Work_Data/Self_Epitopes")
    # parser.add_argument("-csv_F_dir", default="/Users/marcus/Work_Data/Foreign_Epitopes")
    #
    # parser.add_argument("-tesla_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/TESLA")
    # parser.add_argument("-iedb_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/IEDB")
    #
    # parser.add_argument("-neo_hla_tuples_csv", default=""
    #
    # parser.add_argument("-load_target_hla_epi_dir", default="/Users/marcus/Work_Data/Foreign_Epitopes")
    #
    # parser.add_argument("-allele", default='A0301')
    #
    # parser.add_argument("-inclusive_start_ind", default="67")
    # parser.add_argument("-inclusive_end_ind", default="2000")
    # args = parser.parse_args()

    # # MINERVA
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-neo_hla_tuples_csv", default="/sc/arion/projects/FLAI/marcus/MARTA_ALL_IEDB_TESLA_ninemers_HLAs/sorted_imm_iedb_tesla_from_Marta.csv")
    parser.add_argument("-save_pkl_dir", default="/sc/arion/projects/FLAI/marcus/MARTA_IEDB_TESLA_Rust_Results/Immunogenic")

    parser.add_argument("-distance_metric_type", default="all_tcr_all_combos_model")

    parser.add_argument("-data_matrix_dir", default="/sc/arion/work/thomam32/WorkingDir/CFIT/cfit/data/matrices/")
    parser.add_argument("-self_fasta_path", default="/sc/arion/projects/FLAI/marcus/Self_Epitopes/self_peptides.fasta")
    parser.add_argument("-immunogenic_Koncz_fasta_path", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes/immunogenic_Koncz_peptides.fasta")
    parser.add_argument("-nonimmunogenic_Koncz_fasta_path", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes/nonimmunogenic_Koncz_peptides.fasta")
    parser.add_argument("-immunogenic_Ours_fasta_path", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes/immunogenic_Ours_peptides.fasta")
    parser.add_argument("-nonimmunogenic_Ours_fasta_path", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes/nonimmunogenic_Ours_peptides.fasta")

    parser.add_argument("-immunogenic_Tesla_fasta_path", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes/immunogenic_Tesla_peptides.fasta")
    parser.add_argument("-nonimmunogenic_Tesla_fasta_path", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes/nonimmunogenic_Tesla_peptides.fasta")

    parser.add_argument("-csv_S_dir", default="/sc/arion/projects/FLAI/marcus/Self_Epitopes")
    parser.add_argument("-csv_F_dir", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes")

    parser.add_argument("-load_target_hla_epi_dir", default="/sc/arion/projects/FLAI/marcus/Foreign_Epitopes")

    parser.add_argument('-d_ub', type=int, default=100, help='(inclusive bound) distances at or below this threshold contirbute to Kinv_self (models positive selection)')
    parser.add_argument('-d_lb', type=int, default=0, help='(exclusive bound) distances above this cutoff contribute to Kinv_self (partially models negative selection)')

    parser.add_argument("-inclusive_start_ind", default="0")
    parser.add_argument("-inclusive_end_ind", default="10")
    args = parser.parse_args()



    ##distance metric types (ninemer)
    ##['all_tcr_all_combos_model',
    ## 'hamming',
    ## 'epidist_blosum62_distance']


    d_ub = args.d_ub
    d_lb = args.d_lb

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


    # with open(os.path.join(args.load_target_hla_epi_dir,'target_hla_epi_dict.pkl'),'rb') as pot:
    #     target_hla_epi_dict = pickle.load(pot)

    sorted_query_neoantigen_hla_tuples = load_neo_hla_tuples_from_csv(args.neo_hla_tuples_csv)

    for idx, (query_epitope,allele,Kd) in enumerate(sorted_query_neoantigen_hla_tuples):


        if idx < inclusive_start_ind:
            continue
        if idx > inclusive_end_ind:
            break
        print("idx, (query_epitope, allele, query_Kd): ", idx, query_epitope, allele, Kd, flush=True)

        query_epi_list = [query_epitope]

        # # Check if file already saved for this query_epitope
        save_pkl_dir = args.save_pkl_dir + '/d_ub_'+str(d_ub)+'_d_lb_'+str(d_lb)+'/'+allele
        if not os.path.exists(save_pkl_dir):
            os.makedirs(save_pkl_dir)

        strg = 'immunogenicity_outputs_'+distance_metric+'_'+str(idx)+'_'
        for elem in query_epi_list:
            strg += elem #+ '_'
        # strg += hla
        strg += '.pkl'
        outfile = os.path.join(save_pkl_dir,strg)
        if file_already_saved(outfile):
            print("File already saved. Continue...  ", outfile)
            continue


        allele = allele.replace("HLA-", "").replace(":", "")
        hla = "HLA-"+allele

        csv_S_kds_file = os.path.join(args.csv_S_dir, "Kds/self_epitopes_Kd_values_"+hla+".csv")
        csv_KImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Koncz_Imm/epitopes_Kd_values_"+hla+".csv")
        csv_KNImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Koncz_nonImm/epitopes_Kd_values_"+hla+".csv")
        csv_OImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Ours_Imm/epitopes_Kd_values_"+hla+".csv")
        csv_ONImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Ours_nonImm/epitopes_Kd_values_"+hla+".csv")
        csv_TImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Tesla_Imm/epitopes_Kd_values_"+hla+".csv")
        csv_TNImm_kds_file = os.path.join(args.csv_F_dir, "Kds/Tesla_nonImm/epitopes_Kd_values_"+hla+".csv")


        print("calling rust code for (query_epitope,allele): ",query_epi_list, allele)
        original_query_epitope = query_epitope #derive_original_peptide(query_epi_list)


        if save_query_distance_files:
            # # The 'QUERYEPISTR' substring will be replaced by the actual query epi in the rust script.
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
        gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 7),4)) )))
        gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 12),4))  )))

        gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(1e-1, 5, 6),4)) + [1e-8, 1e-100])))
        gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 10),4))  +[1e-8, 1e-100])))

        #################################################################################################
        #################################################################################################
        ##                                 Self settings.

        # d_ub = 100 #(inclusive bound) distances at or below this threshold contirbute to Kinv_self (models positive selection)
        # d_lb = 0 # (exclusive bound) distances above this cutoff contribute to Kinv_self (partially models negative selection)
        #################################################################################################
        #################################################################################################

        target_epitopes_at_allele = ['all']
        compute_logCh = True
        calc_second_hm_without_dist_restriction = False #True #logKInv_self and entropy for rho(e,h)
        logKInv_entropy_self_dict, logKInv_entropy_self_for_rho_dict, logCh_dict, runtime = immunogenicity_rust.compute_log_non_rho_terms_multi_query_single_hla_py(
            query_epi_list,
            [(csv_S_dists_file,load_precomputed_distances,self_fasta_path)],csv_S_kds_file,
            distance_metric,
            data_matrix_dir,
            max_target_num,
            gamma_d_self_values,
            gamma_logkd_self_values,
            d_ub, d_lb,
            compute_logKinv_and_entropy,
            compute_logCh,
            target_epitopes_at_allele,
            calc_second_hm_without_dist_restriction,
        )
        if calc_second_hm_without_dist_restriction == False:
                logKInv_entropy_self_for_rho_dict = logKInv_entropy_self_dict

        print("len(logKInv_entropy_self_dict): ",len(logKInv_entropy_self_dict))
        # print("A few items from logKInv_entropy_self_dict: ")
        # for key, value in itertools.islice(logKInv_entropy_self_dict.items(), 3):
        #     print(f"{key}: {value}")
        print("[python] logKInv_entropy_self_dict/logCh_dict runtime: ",runtime)

        print("len(logCh_dict): ",len(logCh_dict))
        # print("A few items from logKIlogCh_dictnv_entropy_self_dict: ")
        # for key, value in itertools.islice(logCh_dict.items(), 5):
        #     print(f"{key}: {value}")

        #################################################################################################
        #################################################################################################
        ##                                 Foreign settings.
        compute_logCh = False
        calc_second_hm_without_dist_restriction = False
        d_ub_foreign = 1e10 # # not relevant for K_inv evaluated on foreign target set
        d_lb_foreign = 0 # # not relevant for K_inv evaluated on foreign target set

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
        # # FOREIGN DICTS RUST CALLERS REMOVED.

        outputs = [logKInv_entropy_self_dict,
                   original_query_epitope, allele]

        # save_pkl_dir = args.save_pkl_dir + '/d_ub_'+str(d_ub)+'_d_lb_'+str(d_lb)+'/'+allele
        # if not os.path.exists(save_pkl_dir):
        #     os.makedirs(save_pkl_dir)
        #
        # strg = 'immunogenicity_outputs_'+distance_metric+'_'+str(idx)+'_'
        # for elem in query_epi_list:
        #     strg += elem + '_'
        # strg += hla
        # strg += '.pkl'

        # with open(os.path.join(save_pkl_dir,strg), 'wb') as pk:
        #     pickle.dump(outputs, pk)

        # Save the .pkl file
        pkl_path = os.path.join(save_pkl_dir, strg)
        with open(pkl_path, 'wb') as pk:
            pickle.dump(outputs, pk)

        # Compress the .pkl file into a .tar.gz archive
        tar_gz_path = os.path.join(save_pkl_dir, strg + '.tar.gz')
        with tarfile.open(tar_gz_path, 'w:gz') as tar:
            tar.add(pkl_path, arcname=strg)

        # Remove the .pkl file
        os.remove(pkl_path)

    print("[python] script runtime: ", time.time()-start_time)
