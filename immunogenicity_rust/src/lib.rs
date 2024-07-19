use std::collections::{HashMap, HashSet};
use std::fs::{File, remove_file, create_dir_all};
use std::io::BufReader;

use std::error::Error;

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
use lib_rust_function_versions::{calculate_auc_dict_iteratively_from_pickle_files_rs, 
    compute_gamma_d_coeff_rs, compute_logCh_dict_rs, compute_logKinv_and_entropy_dict_rs, 
    compute_log_non_rho_terms_multi_query_single_hla_rs, compute_log_rho_multi_query_rs, 
    immunogenicity_dict_from_pickle_files_rs, immunogenicity_dict_for_auc_from_pickle_files_rs, process_distance_info_vec_rs, process_kd_info_vec_rs};
use lib_data_structures_auxiliary_functions::{DistanceMetricContext, DistanceMetricType, TargetEpiDistances,CalculationError, 
    have_same_keys, print_keys_diff, convert_nested_PyDict_to_HashMap, tuple_to_string};
// use lib_math_functions::{};
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
pub fn compute_log_non_rho_terms_multi_query_single_hla_py(
    query_epi_list: Vec<&str>,
    dist_file_info: Vec<(&str, bool, &str)>,
    kd_file_path: &str,
    dist_metric: &str,
    data_matrix_dir: &str,
    max_target_num: usize,
    gamma_d_values: Vec<f64>,
    gamma_logkd_values: Vec<f64>,
    d_ub: f64,
    d_lb: f64,
    compute_logKinv_and_entropy: bool,
    compute_logCh: bool,
    target_epis_at_hla: Vec<&str>,
    calc_second_hm_without_dist_restriction: bool) -> PyResult<(
                                                    HashMap<String, HashMap<String, Option<(f64, f64)>>>,
                                                    HashMap<String, HashMap<String, Option<(f64, f64)>>>,
                                                    HashMap<String, Option<f64>>,
                                                    u64)> {
    // Call the Rust function
    match compute_log_non_rho_terms_multi_query_single_hla_rs(
        query_epi_list,
        dist_file_info,
        kd_file_path,
        dist_metric,
        data_matrix_dir,
        max_target_num,
        gamma_d_values,
        gamma_logkd_values,
        d_ub,
        d_lb,
        compute_logKinv_and_entropy,
        compute_logCh,
        target_epis_at_hla,
        calc_second_hm_without_dist_restriction,
    ) {
        Ok(result) => Ok(result),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
    }
}


#[pyfunction]
pub fn compute_log_rho_multi_query_py(
    py: Python,
    logKInv_entropy_self_dict: &PyDict,
    logKInv_entropy_Ours_imm_epi_dict: &PyDict,
    logKInv_entropy_Ours_non_imm_epi_dict: &PyDict,
    use_Ours_contribution: bool,
    logKInv_entropy_Koncz_imm_epi_dict: &PyDict,
    logKInv_entropy_Koncz_non_imm_epi_dict: &PyDict,
    use_Koncz_contribution: bool,
    logKInv_entropy_Tesla_imm_epi_dict: &PyDict,
    logKInv_entropy_Tesla_non_imm_epi_dict: &PyDict,
    use_Tesla_contribution: bool) -> PyResult<(HashMap<String, HashMap<String, HashMap<String, Option<f64>>>>, u64)> {
    
    // Convert PyDict inputs to HashMaps
    let query_dict_logKInv_entropy_self = convert_nested_PyDict_to_HashMap(logKInv_entropy_self_dict)?;
    let query_dict_logKInv_entropy_Ours_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Ours_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Ours_non_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Ours_non_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Koncz_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Koncz_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Koncz_non_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Koncz_non_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Tesla_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Tesla_imm_epi_dict)?;
    let query_dict_logKInv_entropy_Tesla_non_imm = convert_nested_PyDict_to_HashMap(logKInv_entropy_Tesla_non_imm_epi_dict)?;

    // Call the pure Rust function and handle its Result
    let (result, elapsed_time) = match compute_log_rho_multi_query_rs(
        &query_dict_logKInv_entropy_self,
        &query_dict_logKInv_entropy_Ours_imm,
        &query_dict_logKInv_entropy_Ours_non_imm,
        use_Ours_contribution,
        &query_dict_logKInv_entropy_Koncz_imm,
        &query_dict_logKInv_entropy_Koncz_non_imm,
        use_Koncz_contribution,
        &query_dict_logKInv_entropy_Tesla_imm,
        &query_dict_logKInv_entropy_Tesla_non_imm,
        use_Tesla_contribution,
    ) {
        Ok(result) => result,
        Err(err) => return Err(pyo3::exceptions::PyException::new_err(format!("Rust function error: {}", err))),
    };

    Ok((result, elapsed_time))
}



/*
AUC section. Assumes immunogenicity pkl files exist for all query epitope-hla pairs.
First aggregate separate epi-hla files.
*/

#[derive(Debug, Hash, PartialEq, Eq)]
struct Params {
    self_params: String,
    foreign_params: String,
}
#[pyfunction]
fn get_immunogenicity_dict_py(
    file_paths_list1: Vec<&str>,
    query_presentation_tuples1: Option<Vec<(&str, f64)>>,
    self_gamma_d: Option<String>,
    self_gamma_logKd: Option<String>,
 ) -> PyResult<HashMap<(String,String),HashMap<String, HashMap<String, Vec<f64>>>>> {

    /*
    This function takes an optional arguments: 
    1. query_presentation_tuples1
    This provides the query epitope presentation factors (e.g., 1/Kd(query_epi,hla)) a vectors of tuples.
    2. (self_gamma_d, self_gamma_logKd) 
    These provide the self params to filter result on.
     */
    // println!("[rust] file_paths_list1: {:?}", file_paths_list1);

    // Print query_presentation_tuples1 if it exists
    if let Some(query_tuples) = &query_presentation_tuples1 {
        println!("[rust] query_presentation_tuples1:");
        for (epi, factor) in query_tuples {
            println!("  - {}: {}", epi, factor);
        }
    }

    // Print self_gamma_d and self_gamma_logKd if they exist
    if let (Some(gamma_d), Some(gamma_logKd)) = (&self_gamma_d, &self_gamma_logKd) {
        println!("[rust] self_gamma_d: {}", gamma_d);
        println!("[rust] self_gamma_logKd: {}", gamma_logKd);
    }

     // Convert the tuples vectors - if they exist - into Option<HashMap<String, f64>>
    let query_presentation_factor_map1: Option<HashMap<String, f64>> = query_presentation_tuples1.map(|tuples| {
        tuples.into_iter().map(|(epi, factor)| (epi.to_string(), factor)).collect()
    });
    
    let filter_on_key: Option<Vec<&str>> = match (self_gamma_d, self_gamma_logKd) {
        (Some(gamma_d), Some(gamma_logKd)) => {
            let key_string = format!("({}, {})", gamma_d, gamma_logKd);
            // Convert key_string into a static string slice
            Some(vec![Box::leak(key_string.into_boxed_str())])
        },
        _ => None,
    };
    println!("[rust] filter_on_key: {:?}", filter_on_key);
    
    // Call the existing function with the first list of file paths (None argument means don't filter on specific outer_key)
    let dict1 = match immunogenicity_dict_from_pickle_files_rs(&file_paths_list1, filter_on_key.as_deref(), query_presentation_factor_map1.as_ref()) {
        Ok(dict) => dict,
        Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
    };

    // Return the dictionary
    Ok(dict1)
}

#[pyfunction]
fn get_immunogenicity_dicts_for_auc_py(
    file_paths_list1: Vec<&str>,
    file_paths_list2: Vec<&str>,
    query_presentation_tuples1: Option<Vec<(&str, f64)>>,
    query_presentation_tuples2: Option<Vec<(&str, f64)>>,
 ) -> PyResult<(HashMap<String, HashMap<String, Vec<f64>>>, HashMap<String, HashMap<String, Vec<f64>>>)> {

    /*
    This function takes two optional arguments: query_presentation_tuples1 and query_presentation_tuples2.
    These provide the query epitope presentation factors (e.g., 1/Kd(query_epi,hla)) a vectors of tuples.
     */


     // Convert the tuples vectors - if they exist - into Option<HashMap<String, f64>>
    let query_presentation_factor_map1: Option<HashMap<String, f64>> = query_presentation_tuples1.map(|tuples| {
        tuples.into_iter().map(|(epi, factor)| (epi.to_string(), factor)).collect()
    });
    let query_presentation_factor_map2: Option<HashMap<String, f64>> = query_presentation_tuples2.map(|tuples| {
        tuples.into_iter().map(|(epi, factor)| (epi.to_string(), factor)).collect()
    });

        
    // Call the existing function with the first list of file paths (None argument means don't filter on specific outer_key, i.e., self paramset)
    let dict1 = match immunogenicity_dict_for_auc_from_pickle_files_rs(&file_paths_list1, None, query_presentation_factor_map1.as_ref()) {
        Ok(dict) => dict,
        Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
    };

    // Call the existing function with the second list of file paths
    let dict2 = match immunogenicity_dict_for_auc_from_pickle_files_rs(&file_paths_list2, None, query_presentation_factor_map2.as_ref()) {
        Ok(dict) => dict,
        Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
    };

    // Return both dictionaries as a tuple
    Ok((dict1, dict2))
}


#[pyfunction]
pub fn calculate_auc_dict_from_pickle_files_py(
    file_paths_list1: Vec<&str>,
    file_paths_list2: Vec<&str>,
    num_self_params_per_iter: usize,
    auc_type_str: &str,
    query_presentation_tuples1: Option<Vec<(&str, f64)>>,
    query_presentation_tuples2: Option<Vec<(&str, f64)>>,
) -> PyResult<HashMap<String, HashMap<String, f64>>> {

    // Pos assay epi-hla pkl files: file_paths_list1
    // Neg assay epi-hla pkl files: file_paths_list2
    let auc_dict_result = calculate_auc_dict_iteratively_from_pickle_files_rs(file_paths_list1, file_paths_list2,num_self_params_per_iter, auc_type_str, query_presentation_tuples1, query_presentation_tuples2);
    match auc_dict_result {
        Ok(auc_dict) => Ok(auc_dict),
        Err(err) => Err(PyRuntimeError::new_err(err.to_string())), // Convert the error to a PyRuntimeError
    }
}


// TO CHECK IF RUST'S auc FUNCTION RETURNS THE SAME VALUES AS PYTHON'S (IT DOES.)
#[pyfunction]
pub fn calculate_auc_py(_py: Python, pos: Vec<f64>, neg: Vec<f64>) -> PyResult<f64> {
    if pos.is_empty() || neg.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input vectors must not be empty"));
    }

    // Combine positive and negative scores into a single vector of (label, score) pairs
    let mut pairs: Vec<(bool, f64)> = pos.into_iter().map(|score| (true, score)).collect();
    pairs.extend(neg.into_iter().map(|score| (false, score)));

    // Compute ROC AUC score
    if let Some(auc) = classifier_measures::roc_auc_mut(&mut pairs) {
        Ok(auc)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Error in computing ROC AUC score"))
    }
}


// A Python module implemented in Rust.
#[pymodule]
fn immunogenicity_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_distances_from_query_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_non_rho_terms_multi_query_single_hla_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_rho_multi_query_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_auc_dict_from_pickle_files_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_immunogenicity_dict_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_auc_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_immunogenicity_dicts_for_auc_py, m)?)?;
    Ok(())
}