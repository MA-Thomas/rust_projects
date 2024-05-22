import immunogenicity_rust 
import itertools


data_matrix_dir = "/Users/marcus/Work_Data/Conifold_editing/CFIT/cfit/data/matrices/"

self_fasta_path = "/Users/marcus/Work_Data/Self_Epitopes/self_peptides.fasta"

immunogenic_Koncz_fasta_path = "/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Koncz_peptides.fasta"
nonimmunogenic_Koncz_fasta_path = "/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Koncz_peptides.fasta"

immunogenic_Ours_fasta_path = "/Users/marcus/Work_Data/Foreign_Epitopes/immunogenic_Ours_peptides.fasta"
nonimmunogenic_Ours_fasta_path = "/Users/marcus/Work_Data/Foreign_Epitopes/nonimmunogenic_Ours_peptides.fasta"

#epidist_blosum62_distance #all_tcr_all_combos_model #hamming
distance_metric = "all_tcr_all_combos_model" 
max_target_num = 20000000
query_epi = "QCDEFGHIK"
hla = "HLA-A0201"

csv_S_dists_file = "/Users/marcus/Work_Data/Self_Epitopes/Distances/QUERYEPISTR_self_distances.csv"
csv_S_kds_file = "/Users/marcus/Work_Data/Self_Epitopes/Kds/self_epitopes_Kd_values_"+hla+".csv"

csv_KImm_dists_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Distances/QUERYEPISTR_Koncz_imm_distances.csv"
csv_KImm_kds_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Kds/Koncz_Imm/epitopes_Kd_values_"+hla+".csv"

csv_KNImm_dists_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Distances/QUERYEPISTR_Koncz_nonimm_distances.csv"
csv_KNImm_kds_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Kds/Koncz_nonImm/epitopes_Kd_values_"+hla+".csv"

csv_OImm_dists_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Distances/QUERYEPISTR_Ours_imm_distances.csv"
csv_OImm_kds_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Kds/Ours_Imm/epitopes_Kd_values_"+hla+".csv"

csv_ONImm_dists_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Distances/QUERYEPISTR_Ours_nonimm_distances.csv"
csv_ONImm_kds_file = "/Users/marcus/Work_Data/Foreign_Epitopes/Kds/Ours_nonImm/epitopes_Kd_values_"+hla+".csv"


# epi_dist_dict, runtime = immunogenicity_rust.compute_distances_from_query_py(query_epi, self_fasta_path, distance_metric, data_matrix_dir, max_target_num, csv_S_dists_file)
# print("[python] runtime: ",runtime)

# epi_dist_dict, runtime = immunogenicity_rust.compute_distances_from_query_py(query_epi, immunogenic_Koncz_fasta_path, distance_metric, data_matrix_dir, max_target_num, csv_KImm_dists_file)
# print("[python] runtime: ",runtime)

# epi_dist_dict, runtime = immunogenicity_rust.compute_distances_from_query_py(query_epi, nonimmunogenic_Koncz_fasta_path, distance_metric, data_matrix_dir, max_target_num, csv_KNImm_dists_file)
# print("[python] runtime: ",runtime)

# epi_dist_dict, runtime = immunogenicity_rust.compute_distances_from_query_py(query_epi, immunogenic_Ours_fasta_path, distance_metric, data_matrix_dir, max_target_num, csv_OImm_dists_file)
# print("[python] runtime: ",runtime)

# epi_dist_dict, runtime = immunogenicity_rust.compute_distances_from_query_py(query_epi, nonimmunogenic_Ours_fasta_path, distance_metric, data_matrix_dir, max_target_num, csv_ONImm_dists_file)
# print("[python] runtime: ",runtime)


'''
SELF:
load_dists_from_csv = False => ~180s 
load_dists_from_csv = True  => ~22s
'''

'''
The middle input of each tuple (boolean) is interpreted as:
     True=>Load precomputed distances 
     False=>Compute distances
'''
load_precomputed_distances = False
compute_logKinv_and_entropy = True
param_info = [(1e-5,1.0,3,1e-2,1.0,3)]
d_PS_threshold = 70.0 # # distances above this threshold do not contirbute to Kinv_self (models positive selection)
d_NS_cutoff = 3.14 # # distances below this cutoff do not contribute to Kinv_self (models negative selection)
#################################################################################################
#################################################################################################
##                                 Self settings.
compute_logCh = True
#################################################################################################
#################################################################################################

logKInv_entropy_self_dict, logCh_dict, runtime = immunogenicity_rust.compute_log_non_rho_terms_py(
    query_epi, 
    [(csv_S_dists_file,load_precomputed_distances,self_fasta_path)],csv_S_kds_file,
    distance_metric, 
    data_matrix_dir, 
    max_target_num,
    param_info,
    d_PS_threshold, d_NS_cutoff,
    compute_logKinv_and_entropy,
    compute_logCh,
)
print("len(logKInv_entropy_self_dict): ",len(logKInv_entropy_self_dict))
print("A few items from logKInv_entropy_self_dict: ")
for key, value in itertools.islice(logKInv_entropy_self_dict.items(), 5):
    print(f"{key}: {value}")
print("[python] self runtime: ",runtime)

print("len(logCh_dict): ",len(logCh_dict))
print("A few items from logKIlogCh_dictnv_entropy_self_dict: ")
for key, value in itertools.islice(logCh_dict.items(), 5):
    print(f"{key}: {value}")

#################################################################################################
#################################################################################################
##                                 Foreign settings.
compute_logCh = False
#################################################################################################
#################################################################################################

logKInv_entropy_Koncz_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_py(
    query_epi, 
    [(csv_KImm_dists_file,load_precomputed_distances,immunogenic_Koncz_fasta_path)], csv_KImm_kds_file,
    distance_metric, 
    data_matrix_dir, 
    max_target_num,
    param_info,
    d_PS_threshold, d_NS_cutoff,
    compute_logKinv_and_entropy,
    compute_logCh,
)
print("len(logKInv_entropy_Koncz_imm_epi_dict): ",len(logKInv_entropy_Koncz_imm_epi_dict))
print("A few items from logKInv_entropy_Koncz_imm_epi_dict: ")
for key, value in itertools.islice(logKInv_entropy_Koncz_imm_epi_dict.items(), 5):
    print(f"{key}: {value}")
print("[python] Koncz_imm runtime: ",runtime)

logKInv_entropy_Koncz_non_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_py(
    query_epi, 
    [(csv_KNImm_dists_file,load_precomputed_distances,nonimmunogenic_Koncz_fasta_path)], csv_KNImm_kds_file,
    distance_metric, 
    data_matrix_dir, 
    max_target_num,
    param_info,
    d_PS_threshold, d_NS_cutoff,
    compute_logKinv_and_entropy,
    compute_logCh,
)
print("[python] Koncz_non_imm runtime: ",runtime)

logKInv_entropy_Ours_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_py(
    query_epi, 
    [(csv_OImm_dists_file,load_precomputed_distances,immunogenic_Ours_fasta_path)], csv_OImm_kds_file,
    distance_metric, 
    data_matrix_dir, 
    max_target_num,
    param_info,
    d_PS_threshold, d_NS_cutoff,
    compute_logKinv_and_entropy,
    compute_logCh,
)
print("[python] Ours_imm runtime: ",runtime)

logKInv_entropy_Ours_non_imm_epi_dict, _, runtime = immunogenicity_rust.compute_log_non_rho_terms_py(
    query_epi, 
    [(csv_ONImm_dists_file,load_precomputed_distances,nonimmunogenic_Ours_fasta_path)], csv_ONImm_kds_file,
    distance_metric, 
    data_matrix_dir, 
    max_target_num,
    param_info,
    d_PS_threshold, d_NS_cutoff,
    compute_logKinv_and_entropy,
    compute_logCh,
)
print("[python] Ours_non_imm runtime: ",runtime)

##  --------------  COMPUTE rho  ----------------
use_Ours_contribution = True 
use_Koncz_contribution = True

log_rho_dict, runtime = immunogenicity_rust.compute_log_rho_py(
    logKInv_entropy_self_dict, 
    logKInv_entropy_Ours_imm_epi_dict,
    logKInv_entropy_Ours_non_imm_epi_dict, 
    use_Ours_contribution,
    logKInv_entropy_Koncz_imm_epi_dict,
    logKInv_entropy_Koncz_non_imm_epi_dict, 
    use_Koncz_contribution,
)
print("[python] log_rho_dict runtime: ",runtime)
print("len(log_rho_dict): ",len(log_rho_dict))
print("One item from log_rho_dict: ")
for key, value in itertools.islice(log_rho_dict.items(), 1):
    print(f"{key}: {value}")