use std::collections::{HashMap, HashSet};
use std::fs::{File, remove_file, create_dir_all};
use std::io::BufReader;

use std::hash::{Hash, Hasher};
//use std::fmt;

use ndarray::{Array1, ArrayBase, OwnedRepr, Dim};

use std::fs;
use std::io::{self, BufRead, Write};
use std::io::BufWriter;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyRuntimeError;

use tar::Builder;
use flate2::write::GzEncoder;
//use flate2::Compression;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

//////////////////////////////////////////////////////////////////////////////////////////
////////////   Declare modules (contents are in the associated files)       //////////////
mod lib_io; 
mod lib_rust_function_versions;
mod lib_data_structures_auxiliary_functions;
mod lib_math_functions;

//////////////////////////////////////////////////////////////////////////////////////////
////////////   Bind these function names to their full module paths         //////////////
use lib_io::{parse_fasta, save_epitopes_distances_to_tar_gz, 
    load_epitopes_distances_from_tar_gz, load_epitopes_kds_from_tar_gz, 
    set_json_path, evaluate_context}; 
use lib_rust_function_versions::{compute_gamma_d_coeff_rs, 
    process_distance_info_vec_rs, process_kd_info_vec_rs, 
    compute_logKinv_and_entropy_dict_rs, compute_logCh_dict_rs};
use lib_data_structures_auxiliary_functions::{DistanceMetricContext, DistanceMetricType, TargetEpiDistances, have_same_keys};
use lib_math_functions::{calculate_auc_rs};
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////



// #######################################################################################
#[pyfunction]
fn compute_distances_from_query_py(query_epi: &str, 
                                fasta_path: &str, 
                                dist_metric: &str, 
                                data_matrix_dir: &str, 
                                max_target_num: usize, 
                                save_csv_file: Option<&str>) -> PyResult<(HashMap<String, f64>, u64)> {

    let start_time = std::time::Instant::now();

    let dm_type = match dist_metric {
        "hamming" =>  DistanceMetricType::hamming,
        "epidist_blosum62_distance" => DistanceMetricType::epidist_blosum62_distance,
        "all_tcr_all_combos_model" => DistanceMetricType::all_tcr_all_combos_model,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid distance metric",
            ));
        }
    };

    // Set full JSON file path.
    let json_file_path = set_json_path(data_matrix_dir, &dm_type);

    // Set the context which includes the JSON path.
    let context = DistanceMetricContext {
        metric: dm_type,
        json_path: json_file_path,
    };

    // Evaluate the context (incl loading the JSON data).
    // Handle transmitting any errors that occur to python.
    let distance_context = match evaluate_context(context) {
        Ok(ctx) => ctx,
        Err(err) => return Err(PyErr::new::<PyException, _>(format!("Error: {}", err))),
    };

    // Parse target epitopes from the .fasta file
    // Handle transmitting any errors that occur to python.
    // To change the epitope buffer capacity away from its default, replace None with e.g., Some(20).
    let target_epitopes = match parse_fasta(fasta_path, max_target_num, None) {
        Ok(epitopes) => epitopes,
        Err(err) => return Err(PyErr::new::<PyException, _>(format!("Error: {}", err))),
    };

    // Parallel Version
    let epitopes_distances: Result<Vec<TargetEpiDistances>, PyErr> = target_epitopes.par_iter() // Parallel iterator
    .map(|target_epi| {
        match distance_context.epitope_dist(query_epi, target_epi) {
            Ok((distance, _)) => Ok(TargetEpiDistances {
                epitope: target_epi.clone(),
                distance,
            }),
            Err(err) => Err(PyErr::new::<PyException, _>(format!("Error: {}", err))),
        }
    })
    .collect();

    // Check if there were any errors during computation
    let epitopes_distances = match epitopes_distances {
        Ok(distances) => distances,
        Err(err) => return Err(err), // Return early with the error if any
    };
    // Create HashMap to store epitope distances
    let mut epi_dist_dict: HashMap<String, f64> = HashMap::new();
    // Fill epi_dist_dict using computed distances
    for dist_info in &epitopes_distances {
        // Use entry API to insert if key doesn't exist
        epi_dist_dict.entry(dist_info.epitope.clone()).or_insert(dist_info.distance);
    }


    // Optionally save the distances to a CSV file
    if let Some(csv_distances_file_path) = save_csv_file {
        //save_epitopes_distances_to_tar_gz //save_distances_to_csv
        if let Err(err) = save_epitopes_distances_to_tar_gz(csv_distances_file_path, &epitopes_distances) {
            return Err(PyErr::new::<PyException, _>(format!("Error saving distances to CSV: {}", err)));
        }
    }
    // End timing
    let end_time = start_time.elapsed().as_secs_f64() as u64;
    Ok((epi_dist_dict, end_time)) // Return HashMap instead of vector
}





#[pyfunction]
pub fn compute_log_non_rho_terms_multi_query_single_hla_py(query_epi_list: Vec<&str>, 
    dist_file_info: Vec<(&str, bool, &str)>,
    kd_file_path: &str,
    dist_metric: &str, 
    data_matrix_dir: &str, 
    max_target_num: usize,
    gamma_d_values: Vec<f64>,
    gamma_logkd_values: Vec<f64>,
    d_PS_threshold: f64,
    d_NS_cutoff: f64,
    compute_logKinv_and_entropy: bool, compute_logCh: bool) -> PyResult<(HashMap<String, HashMap<String, Option<(f64, f64)>>>, HashMap<String, Option<f64>>, u64)> {

    let start_time = std::time::Instant::now();

    //////////    PREPROCESSING    /////////
    // Load Kds from Query into HashMap
    let epi_kd_dict = match process_kd_info_vec_rs(&kd_file_path) {
        Ok(epi_kd_dict) => {
            println!("[rust] Kds processing succeeded.");
            println!("[rust] Length of epi_kd_dict: {}", epi_kd_dict.len());
            epi_kd_dict
        },
        Err(err) => return Err(pyo3::exceptions::PyValueError::new_err(format!("Error processing Kds: {}", err))),
    };

    let mut logKinv_entropy_multi_query_dict: HashMap<String, HashMap<String, Option<(f64, f64)>>> = HashMap::new();

    // Compute gamma_d_coeff
    let gamma_d_coeff_result = compute_gamma_d_coeff_rs(dist_metric, data_matrix_dir);
    let gamma_d_coeff: f64;
    match gamma_d_coeff_result {
        Ok(coeff) => {
            gamma_d_coeff = coeff;
            // Use gamma_d_coeff here
            println!("Gamma D coefficient: {}", gamma_d_coeff);
        }
        Err(err) => {
            // Handle the error
            eprintln!("Error: {}", err);
            let py_err = PyRuntimeError::new_err(err); // Create a PyErr from the string error message
            return Err(py_err);
        }
    }

    // Handle Counts and Concs into HashMap (in future, may choose to load these values).
    /*
        For self epitopes, the count is the number of times the 9-mer (or n-mer) epitope appears 
        in the genome - whether the degeneracy is within a gene, across genes, or both.
        For non self epitopes, the count is 1.
        For self epitopes, the conc is the concentration of the 9-mer (or n-mer) and is based on 
        cTEC gene expression in the Thymus during positive selection of T-cells.
        For non self epitopes, the conc is 1.

        For now, assume both are always 1 (log value=0).
        (These are not relevant for nonself epitopes but the value 1 is ok for use in compute_logKinv_and_entropy_dict_rs(). )
    */
    let mut epi_log_count_dict = HashMap::with_capacity(epi_kd_dict.len());
    let mut epi_log_conc_dict = HashMap::with_capacity(epi_kd_dict.len());
    for key in epi_kd_dict.keys() {
        epi_log_count_dict.insert(key.clone(), 0.0);
        epi_log_conc_dict.insert(key.clone(), 0.0);
    }
    let use_counts_concs: bool = false;


    for query_epi in query_epi_list {

        // Load or Generate Distances from Query into HashMap
        let epi_dist_dict = match process_distance_info_vec_rs(&dist_file_info, query_epi, dist_metric, data_matrix_dir, max_target_num) {
            Ok(epi_dist_dict) => {
                println!("[rust] Distance processing succeeded.");
                println!("[rust] Length of epi_dist_dict: {}", epi_dist_dict.len());
                epi_dist_dict
            },
            Err(err) => return Err(pyo3::exceptions::PyValueError::new_err(format!("Error processing distances: {}", err))),
        };

        // Check if both HashMaps have the same keys
        if !have_same_keys(&epi_dist_dict, &epi_kd_dict) {
            return Err(pyo3::exceptions::PyValueError::new_err("The keys of epi_dist_dict and epi_kd_dict do not match."));
        }

        //////////    MODEL COMPUTATIONS    /////////
        let mut logKinv_entropy_dict: HashMap<String, Option<(f64, f64)>> = HashMap::new();

        if compute_logKinv_and_entropy {

            logKinv_entropy_dict = compute_logKinv_and_entropy_dict_rs(
                &epi_dist_dict,
                &epi_kd_dict,
                &epi_log_count_dict,
                &epi_log_conc_dict,
                &gamma_d_values,
                &gamma_logkd_values,
                gamma_d_coeff,
                d_PS_threshold,
                d_NS_cutoff,
                use_counts_concs,
            );
        }
        
        logKinv_entropy_multi_query_dict.insert(query_epi.to_string(), logKinv_entropy_dict);
    }

    let mut logCh_dict: HashMap<String, Option<f64>> = HashMap::new();
    let log_n_wt = 9.2103; // log(10,000) because 10,000 is the number of MHC molecules per cell.
    let log_h_num = 1.7917; // log(6) because there are 6 HLAs per person.
    if compute_logCh {
        logCh_dict = compute_logCh_dict_rs(
            &epi_kd_dict,
            &epi_log_count_dict,
            &epi_log_conc_dict,
            &gamma_logkd_values,
            log_n_wt,
            log_h_num,
            use_counts_concs,
        );
    }

    let end_time = start_time.elapsed().as_secs_f64() as u64;
    Ok((logKinv_entropy_multi_query_dict, logCh_dict, end_time)) 
  
}


#[pyfunction]
fn compute_log_rho_multi_query_py(
    logKInv_entropy_self_dict: &PyDict,
    logKInv_entropy_Ours_imm_epi_dict: &PyDict,
    logKInv_entropy_Ours_non_imm_epi_dict: &PyDict,
    use_Ours_contribution: bool,
    logKInv_entropy_Koncz_imm_epi_dict: &PyDict,
    logKInv_entropy_Koncz_non_imm_epi_dict: &PyDict,
    use_Koncz_contribution: bool) -> PyResult<(HashMap<String, HashMap<String, HashMap<String, Option<f64>> > >, u64)> {

    // Auxiliary function to convert PyDict to HashMap<String, Option<(f64, f64)>>
    fn convert_nested_PyDict_to_HashMap(dict: &PyDict) -> PyResult<HashMap<String, HashMap<String, Option<(f64, f64)>>>> {
        let mut result_hashm: HashMap<String, HashMap<String, Option<(f64, f64)>>> = HashMap::new();
    
        for (key, value) in dict {
            let key_str: String = key.extract()?;
            let nested_dict = value.extract::<&PyDict>()?; // Extract reference to PyDict
    
            let mut nested_hashm: HashMap<String, Option<(f64, f64)>> = HashMap::new();
            for (nested_key, nested_value) in nested_dict.iter() { // Iterate over reference to PyDict
                let nested_key_str: String = nested_key.extract()?;
                if let Ok((v1, v2)) = nested_value.extract::<(f64, f64)>() {
                    nested_hashm.insert(nested_key_str, Some((v1, v2)));
                } else {
                    nested_hashm.insert(nested_key_str, None);
                }
            }
            result_hashm.insert(key_str, nested_hashm);
        }
    
        Ok(result_hashm)
    }


    let start_time = std::time::Instant::now();
    
    // log_rho_multi_query_dict[query_epi][self_params][foreign_params] = rho_value
    let mut log_rho_multi_query_dict: HashMap<String, HashMap<String, HashMap<String, Option<f64>> > > = HashMap::new();

    // Convert all input python dictionaries to rust hashmaps. 
    let query_dict_logKInv_entropy_self = convert_nested_PyDict_to_HashMap(logKInv_entropy_self_dict)?;
    let query_dict_logKInv_entropy_Ours_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Ours_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Ours_non_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Ours_non_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Koncz_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Koncz_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Koncz_non_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Koncz_non_imm_epi_dict)?;

    for query_epi in query_dict_logKInv_entropy_self.keys() { 

        let logKInv_entropy_self = match query_dict_logKInv_entropy_self.get(query_epi) {
            Some(values) => values,
            None => panic!("logKInv_entropy_self Key '{}' not found", query_epi),
        };
        let logKInv_entropy_Ours_imm = match query_dict_logKInv_entropy_Ours_imm.get(query_epi) {
            Some(values) => values,
            None => panic!("logKInv_entropy_Ours_imm Key '{}' not found", query_epi),
        };
        let logKInv_entropy_Ours_non_imm = match query_dict_logKInv_entropy_Ours_non_imm.get(query_epi) {
            Some(values) => values,
            None => panic!("logKInv_entropy_Ours_non_imm Key '{}' not found", query_epi),
        };
        let logKInv_entropy_Koncz_imm = match query_dict_logKInv_entropy_Koncz_imm.get(query_epi) {
            Some(values) => values,
            None => panic!("logKInv_entropy_Koncz_imm Key '{}' not found", query_epi),
        };
        let logKInv_entropy_Koncz_non_imm = match query_dict_logKInv_entropy_Koncz_non_imm.get(query_epi) {
            Some(values) => values,
            None => panic!("logKInv_entropy_Koncz_non_imm Key '{}' not found", query_epi),
        };


        // For the current query_epi: log_rho_dict[self_params][foreign_params] = rho_value
        let mut log_rho_dict: HashMap<String, HashMap<String, Option<f64>> > = HashMap::new();
        let mut self_term = 0.0;
        let mut iedb_imm_term = 0.0;
        let mut iedb_non_imm_term = 0.0;
        let mut log_rho = 0.0;

        // Iterate over self dict and populate log_rho_dict.
        for (self_key, self_value) in logKInv_entropy_self {
            
            self_term = 0.0;

            // Extracting log_K_inv and entropy for self dict.
            let (log_K_inv_self, entropy_self) = match self_value {
                Some((log_K_inv, entropy)) => (*log_K_inv, *entropy),
                None => continue, // Skip if self_value is None
            };
            self_term += (log_K_inv_self - entropy_self).exp();

            
            // Iterate over foreign dicts and extract log_K_inv and entropy for each.
            for (foreign_key, foreign_value) in logKInv_entropy_Ours_imm {
                
                iedb_imm_term = 0.0;
                iedb_non_imm_term = 0.0;
                
                if use_Ours_contribution {
                    let (log_K_inv_Ours_imm, entropy_Ours_imm) = match foreign_value {
                        Some((log_K_inv, entropy)) => (*log_K_inv, *entropy),
                        None => continue, // Skip if self_value is None
                    };
                    iedb_imm_term += (log_K_inv_Ours_imm - entropy_Ours_imm).exp();

                    let foreign_value = logKInv_entropy_Ours_non_imm.get(foreign_key);
                    let (log_K_inv_Ours_non_imm, entropy_Ours_non_imm) = match foreign_value {
                        Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                        _ => continue, // Skip if self_value is None
                    };
                    iedb_non_imm_term += (log_K_inv_Ours_non_imm - entropy_Ours_non_imm).exp();
                } 

                if use_Koncz_contribution {
                    let foreign_value = logKInv_entropy_Koncz_imm.get(foreign_key);
                    let (log_K_inv_Koncz_imm, entropy_Koncz_imm) = match foreign_value {
                        Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                        _ => continue, // Skip if foreign_value is None or inner value is None
                    };
                    iedb_imm_term += (log_K_inv_Koncz_imm - entropy_Koncz_imm).exp();

                    let foreign_value = logKInv_entropy_Koncz_non_imm.get(foreign_key);
                    let (log_K_inv_Koncz_non_imm, entropy_Koncz_non_imm) = match foreign_value {
                        Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                        _ => continue, // Skip if self_value is None
                    };
                    iedb_non_imm_term += (log_K_inv_Koncz_non_imm - entropy_Koncz_non_imm).exp();
                }

                log_rho = -self_term + iedb_imm_term - iedb_non_imm_term;

                // Insert log_rho at current self/foreign params
                log_rho_dict
                .entry(self_key.clone())
                .or_insert_with(HashMap::new)
                .insert(foreign_key.clone(), Some(log_rho));
            }
                
        }

        log_rho_multi_query_dict.insert(query_epi.to_string(), log_rho_dict);
    }

    let end_time = start_time.elapsed().as_secs_f64() as u64;
    Ok((log_rho_multi_query_dict, end_time))
}



/*
AUC section
*/
#[pyfunction]
fn calculate_auc_dict(nb_records: Vec<(String, i32, (f64, f64), (f64, f64), f64)>) -> HashMap<String, HashMap<String, Vec<(f64, f64, f64)>>> {
    /*
    Returns auc_dict.
    auc_dict[FOREIGN_params_numeric[1]][SELF_params_numeric] = [[FOREIGN_params_numeric[0], query_assay_auc]]
        FOREIGN_params_numeric[1] is the gamma_logKd value for KInv_foreign 
        SELF_params_numeric are the gamma_d,gamma_logKd values for KInv_self 
        FOREIGN_params_numeric[0] is the gamma_d value for KInv_foreign
     */

    // Compute AUC values in parallel and collect them into a vector of tuples
    let temp_auc_values: Vec<_> = nb_records
        .par_iter()
        .map(|(full_query_epi, immunogenicity, self_params_numeric, foreign_params_numeric, nb)| {
            let pos_assay = if *immunogenicity == 1 { vec![*nb] } else { Vec::new() };
            let neg_assay = if *immunogenicity == 0 { vec![*nb] } else { Vec::new() };

            let auc_value = calculate_auc_rs(&pos_assay, &neg_assay).unwrap();

            let foreign_key = format!("{:?}", foreign_params_numeric.0);
            let self_key = format!("{:?}", self_params_numeric);

            (foreign_key, self_key, foreign_params_numeric.0, auc_value, 0.0)
        })
        .collect();

    // Populate auc_dict from temp_auc_values
    let mut auc_dict = HashMap::new();
    for (foreign_key, self_key, foreign_numeric, auc_value, _) in temp_auc_values {
        auc_dict
            .entry(foreign_key)
            .or_insert_with(HashMap::new)
            .entry(self_key)
            .or_insert_with(Vec::new)
            .push((foreign_numeric, auc_value, 0.0)); // koncz_auc set to 0 since there's only one assay
    }

    auc_dict
}

// A Python module implemented in Rust.
#[pymodule]
fn immunogenicity_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_distances_from_query_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_non_rho_terms_multi_query_single_hla_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_rho_multi_query_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_auc_dict, m)?)?;
    Ok(())
}