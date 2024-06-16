use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use ndarray::{Array1, ArrayBase, OwnedRepr, Dim};

use std::io::Read;
use serde_pickle::from_reader;
use std::error::Error;
use serde::ser::StdError;
use std::fmt;

/////////////////////////////////////////////////////////////////////////
////////////   Import from other scripts via the main module.   //////////////
use crate::*; // Import from the main module (lib.rs)
use crate::lib_data_structures_auxiliary_functions::{DistanceMetricType, DistanceMetricContext, TargetEpiDistances, tuple_to_string, generate_variants};
use lib_io::{PickleContents, set_json_path, evaluate_context, load_all_pkl_files};
use lib_math_functions::{EntropyError, log_sum, compute_entropy, calculate_auc};
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

pub fn compute_gamma_d_coeff_rs(dist_metric: &str, data_matrix_dir: &str ) -> Result<f64, String> {

    let dm_type = match dist_metric {
        "hamming" =>  DistanceMetricType::hamming,
        "epidist_blosum62_distance" => DistanceMetricType::epidist_blosum62_distance,
        "all_tcr_all_combos_model" => DistanceMetricType::all_tcr_all_combos_model,
        _ => {
            return Err("Invalid distance metric".to_string());
        }
    };

    let json_file_path = set_json_path(data_matrix_dir, &dm_type);

    let context = DistanceMetricContext {
        metric: dm_type,
        json_path: json_file_path,
    };

    let distance_context = match evaluate_context(context) {
        Ok(ctx) => ctx,
        Err(err) => return Err(format!("Error: {}", err)),
    };

    let gamma_d_coeff = match distance_context.gamma_d_coeff() {
        Ok(dist) => dist,
        Err(err) => return Err(format!("Error: {}", err)),
    };

    Ok(gamma_d_coeff)
}


pub fn compute_distances_from_query_rs(query_epi: &str, 
    fasta_path: &str, 
    dist_metric: &str, 
    data_matrix_dir: &str, 
    max_target_num: usize, 
    save_csv_file: Option<&str>) -> Result<(HashMap<String, f64>, u64), String> {

    let start_time = std::time::Instant::now();
    let mut epitopes_distances = Vec::new();
    let mut epi_dist_dict = HashMap::new();

    let dm_type = match dist_metric {
        "hamming" =>  DistanceMetricType::hamming,
        "epidist_blosum62_distance" => DistanceMetricType::epidist_blosum62_distance,
        "all_tcr_all_combos_model" => DistanceMetricType::all_tcr_all_combos_model,
        _ => {
            return Err("Invalid distance metric".to_string());
        }
    };

    let json_file_path = set_json_path(data_matrix_dir, &dm_type);

    let context = DistanceMetricContext {
        metric: dm_type,
        json_path: json_file_path,
    };

    let distance_context = match evaluate_context(context) {
        Ok(ctx) => ctx,
        Err(err) => return Err(format!("Error: {}", err)),
    };

    let target_epitopes = match parse_fasta(fasta_path, max_target_num, None) {
        Ok(epitopes) => epitopes,
        Err(err) => return Err(format!("Error: {}", err)),
    };

    for target_epi in &target_epitopes {
        let (distance, _) = match distance_context.epitope_dist(query_epi, target_epi) {
            Ok((dist, _)) => (dist, ()),
            Err(err) => return Err(format!("Error: {}", err)),
        };

        epitopes_distances.push(TargetEpiDistances {
            epitope: target_epi.clone(),
            distance: distance,
        });
        epi_dist_dict.insert(target_epi.clone(), distance);
    }

    if let Some(csv_distances_file_path) = save_csv_file {
        if let Err(err) = save_epitopes_distances_to_tar_gz(csv_distances_file_path, &epitopes_distances) {
            return Err(format!("Error saving distances to CSV: {}", err));
        }
    }

    let end_time = start_time.elapsed().as_secs_f64() as u64;

    Ok((epi_dist_dict, end_time))
}


pub fn process_distance_info_vec_rs(dist_file_info: &Vec<(&str, bool, &str)>, 
                             query_epi: &str, 
                             dist_metric: &str, 
                             data_matrix_dir: &str, 
                             max_target_num: usize) -> Result<HashMap<String, f64>, String> {
    
    for (csv_distances_file_path, load_dists_from_csv, fasta_path) in dist_file_info.iter() {

        
        let mut csv_distances_file_path = csv_distances_file_path.to_string();
        csv_distances_file_path = csv_distances_file_path.replace("QUERYEPISTR", query_epi);
        let csv_distances_file_path: &str = &csv_distances_file_path;

        if *load_dists_from_csv {
            let mut csv_distances_file_path_opt: Option<String> = Some(csv_distances_file_path.to_string());

            if let Some(path) = csv_distances_file_path_opt.as_mut() {
                if let Some(extension_position) = path.rfind('.') {
                    path.replace_range(extension_position.., ".tar.gz");
                } else {
                    return Err("Invalid file path provided".to_string());
                }
            } else {
                return Err("CSV file path is not provided".to_string());
            }

            let epitopes_distances = match csv_distances_file_path_opt {
                Some(ref path) => load_epitopes_distances_from_tar_gz(path),
                None => {
                    return Err("CSV file path is not provided".to_string());
                }
            };
            match epitopes_distances {
                Ok(elems) => {
                    let target_epi_dist_dict: HashMap<String, f64> = elems
                        .into_iter()
                        .map(|elem| (elem.epitope, elem.distance))
                        .collect();

                    println!("First few entries of target_epi_dist_dict: ");
                    if target_epi_dist_dict.len() > 3 {
                        for (key, value) in target_epi_dist_dict.iter().take(3) {
                            println!("[rust] {:?}: {:?}", key, value);
                        }
                    } else {
                        println!("[rust] {:?}", target_epi_dist_dict);
                    }
                    return Ok(target_epi_dist_dict);
                }
                Err(err) => {
                    return Err(format!("Error: {}", err));
                }
            }
        } else {
            
            let save_file = if csv_distances_file_path.is_empty() {
                None
            } else {
                Some(csv_distances_file_path)
            };

            let output_tuple = compute_distances_from_query_rs(
                query_epi, fasta_path,
                dist_metric, data_matrix_dir,
                max_target_num, save_file);

            match output_tuple {
                Ok((target_epi_dist_dict, time)) => {
                    println!("[rust] time taken for compute_distances_from_query_rs(): {} seconds", time);
                    return Ok(target_epi_dist_dict);
                }
                Err(err) => {
                    return Err(format!("Error: {}", err));
                }
            }
        }
    }
    Err("No distances processed".to_string())
}


pub fn process_kd_info_vec_rs(csv_kds_file_path: &str) -> Result<HashMap<String, f64>, String> {
    

    let mut csv_kds_file_path_opt: Option<String> = Some(csv_kds_file_path.to_string());

    if let Some(path) = csv_kds_file_path_opt.as_mut() {
        if let Some(extension_position) = path.rfind('.') {
            path.replace_range(extension_position.., ".tar.gz");
        } else {
            return Err("Invalid file path provided".to_string());
        }
    } else {
        return Err("CSV file path is not provided".to_string());
    }

    let epitopes_kds = match csv_kds_file_path_opt {
        Some(ref path) => load_epitopes_kds_from_tar_gz(path),
        None => {
            return Err("CSV file path is not provided".to_string());
        }
    };
    match epitopes_kds {
        Ok(elems) => {
            let target_epi_kd_dict: HashMap<String, f64> = elems
                .into_iter()
                .map(|elem| (elem.epitope, elem.Kd))
                .collect();

            println!("First few entries of target_epi_kd_dict: ");
            if target_epi_kd_dict.len() > 3 {
                for (key, value) in target_epi_kd_dict.iter().take(3) {
                    println!("[rust] {:?}: {:?}", key, value);
                }
            } else {
                println!("[rust] {:?}", target_epi_kd_dict);
            }
            return Ok(target_epi_kd_dict);
        }
        Err(err) => {
            return Err(format!("Error: {}", err));
        }
    }
}





pub fn compute_logKinv_and_entropy_dict_rs(
    epi_dist_dict: &HashMap<String, f64>,
    epi_kd_dict: &HashMap<String, f64>,
    epi_log_count_dict: &HashMap<String, f64>,
    epi_log_conc_dict: &HashMap<String, f64>,
    gamma_d_values: &[f64],
    gamma_logkd_values: &[f64],
    gamma_d_coeff: f64,
    d_ub: f64,
    d_lb: f64,
    use_counts_concs: bool) -> HashMap<String, Option<(f64, f64)>> {

    const KD_THRESHOLD: f64 = 1e-10;

    // Convert HashMap values to arrays. Key order is preserved among all four arrays.
    let epi_dist_array = Array1::from(epi_dist_dict.values().copied().collect::<Vec<_>>());
    let epi_kd_array = Array1::from(epi_kd_dict.values().copied().collect::<Vec<_>>());
    let epi_log_count_array = if use_counts_concs {
        Array1::from(epi_log_count_dict.values().copied().collect::<Vec<_>>())
    } else {
        Array1::zeros(epi_dist_array.len())
    };
    let epi_log_conc_array = if use_counts_concs {
        Array1::from(epi_log_conc_dict.values().copied().collect::<Vec<_>>())
    } else {
        Array1::zeros(epi_dist_array.len())
    };


    // Check that all arrays have the same length
    assert_eq!(epi_dist_array.len(), epi_kd_array.len());
    assert_eq!(epi_dist_array.len(), epi_log_count_array.len());
    assert_eq!(epi_dist_array.len(), epi_log_conc_array.len());

    // Apply the natural logarithm to epi_kd_array, replacing zero values with KD_THRESHOLD
    let epi_log_kd_array = epi_kd_array.mapv(|kd| {
        let kd_safe = if kd == 0.0 { KD_THRESHOLD } else { kd };
        kd_safe.ln()
    });

    // Masking based on PS (d_ub) and NS (d_lb)
    let mask = &epi_dist_array.map(|&x| x <= d_ub) & epi_dist_array.map(|&x| x > d_lb);

    // Ensure mask is of the same length as the arrays
    assert_eq!(mask.len(), epi_dist_array.len());

    let epi_dist_array_masked = &epi_dist_array * &mask.map(|&x| if x { 1.0 } else { 0.0 });
    let epi_log_kd_array_masked = &epi_log_kd_array * &mask.map(|&x| if x { 1.0 } else { 0.0 });
    let epi_log_count_array_masked = &epi_log_count_array * &mask.map(|&x| if x { 1.0 } else { 0.0 });
    let epi_log_conc_array_masked = &epi_log_conc_array * &mask.map(|&x| if x { 1.0 } else { 0.0 });

    // Ensure masked arrays are of the same length
    assert_eq!(epi_dist_array_masked.len(), epi_log_kd_array_masked.len());
    assert_eq!(epi_dist_array_masked.len(), epi_log_count_array_masked.len());
    assert_eq!(epi_dist_array_masked.len(), epi_log_conc_array_masked.len());

    // Clone the arrays inside Arc (Ensures that the shared data is safely accessible across parallel threads without unnecessary cloning.)
    let epi_dist_array_masked_arc = Arc::new(epi_dist_array_masked.clone());
    let epi_log_kd_array_masked_arc = Arc::new(epi_log_kd_array_masked.clone());
    let epi_log_count_array_masked_arc = Arc::new(epi_log_count_array_masked.clone());
    let epi_log_conc_array_masked_arc = Arc::new(epi_log_conc_array_masked.clone());

    let log_Kinv_dict: HashMap<String, Option<(f64, f64)>> = gamma_d_values
        .par_iter()
        .flat_map(|&gamma_d_value| {
            /*
            Note: Arc::as_ref() is called next to obtain an immutable reference to the data inside the Arc. 
            This is necessary because the computation inside the .par_iter() closure needs to access the actual arrays, 
            not the Arc wrapper itself.
             */
            let epi_dist_array_masked = Arc::as_ref(&epi_dist_array_masked_arc);
            let epi_log_kd_array_masked = Arc::as_ref(&epi_log_kd_array_masked_arc);
            let epi_log_count_array_masked = Arc::as_ref(&epi_log_count_array_masked_arc);
            let epi_log_conc_array_masked = Arc::as_ref(&epi_log_conc_array_masked_arc);

            gamma_logkd_values
                .par_iter()
                .map(move |&gamma_logKd_value| {
                    // Compute individual components
                    let comp1 = -gamma_d_value * epi_dist_array_masked * gamma_d_coeff;
                    let comp2 = - gamma_logKd_value * epi_log_kd_array_masked;
                    let comp3 = epi_log_count_array_masked;
                    let comp4 = epi_log_conc_array_masked;

                    let values = comp1 + comp2 + comp3 + comp4;
                    let values_slice = values.as_slice().unwrap();
                    let result = match log_sum(values_slice) {
                        Ok(result) => result,
                        Err(err) => {
                            panic!("Error calculating log sum: {:?}", err);
                        }
                    };
                    let entropy = match compute_entropy(values_slice, result) {
                        Ok(entropy) => entropy,
                        Err(error) => {
                            match error {
                                EntropyError::NaN => {
                                    panic!("Error calculating entropy: entropy value is NaN.");
                                }
                                EntropyError::Infinite => {
                                    panic!("Error calculating entropy: entropy value is Infinite.");
                                }
                                EntropyError::InvalidDistribution => {
                                    panic!("Error calculating entropy: invalid probability encountered.");
                                }
                            }
                        }
                    };
                    // let entropy = match compute_entropy(values_slice, result) {
                    //     Some(entropy) => entropy,
                    //     None => {
                    //         panic!("Error calculating entropy: entropy value is None.");
                    //     }
                    // };
                    let key = tuple_to_string(&(gamma_d_value, gamma_logKd_value));
                    (key, Some((result, entropy)))
                })
        })
        .collect();

    log_Kinv_dict
}




pub fn compute_logCh_dict_rs(
    epi_kd_dict: &HashMap<String, f64>,
    epi_log_count_dict: &HashMap<String, f64>,
    epi_log_conc_dict: &HashMap<String, f64>,
    gamma_logkd_values: &[f64],
    log_n_wt: f64,
    log_h_num: f64,
    use_counts_concs: bool) -> HashMap<String, Option<f64>> {

    let epi_kd_dict_arc = Arc::new(epi_kd_dict.clone());

    let mut log_Ch_dict: HashMap<String, Option<f64>> = HashMap::new();
    
    if use_counts_concs { 

        let epi_log_count_dict_arc = Arc::new(epi_log_count_dict.clone());
        let epi_log_conc_dict_arc = Arc::new(epi_log_conc_dict.clone());

        log_Ch_dict = gamma_logkd_values
            .par_iter()
            .map(|&gamma_logKd_value| {

                let epi_kd_dict_ref = &epi_kd_dict_arc;
                let epi_log_count_dict_ref = &epi_log_count_dict_arc;
                let epi_log_conc_dict_ref = &epi_log_conc_dict_arc;

                let mut values = Vec::with_capacity(epi_kd_dict_ref.len());
                for (epi, &kd) in epi_kd_dict_ref.iter() {
                    if let Some(&log_count) = epi_log_count_dict_ref.get(epi) {
                        if let Some(&log_conc) = epi_log_conc_dict_ref.get(epi) {
                            let value = -gamma_logKd_value * kd.ln() + log_count + log_conc;
                            values.push(value);
                        }
                    }
                }

                let result = match log_sum(&values) {
                    Ok(result) => result,
                    Err(err) => {
                        panic!("Error calculating log sum: {:?}", err);
                    }
                };
                let key = gamma_logKd_value.to_string();
                (key, Some(result))

            })
            .collect::<HashMap<_, _>>();

    } else {

        log_Ch_dict = gamma_logkd_values
            .par_iter()
            .map(|&gamma_logKd_value| {
                let epi_kd_dict_ref = &epi_kd_dict_arc;

                let mut values = Vec::with_capacity(epi_kd_dict_ref.len());
                for (epi, &kd) in epi_kd_dict_ref.iter() {
                    let value = -gamma_logKd_value * kd.ln();
                    values.push(value); 
                }

                let result = match log_sum(&values) {
                    Ok(result) => result,
                    Err(err) => {
                        panic!("Error calculating log sum: {:?}", err);
                    }
                };
                let key = gamma_logKd_value.to_string();
                (key, Some(result))

            })
            .collect::<HashMap<_, _>>();        
    }

    log_Ch_dict
}


//TODO: add a rust-python wrapper function to lib.rs that calls this!
pub fn compute_log_non_rho_terms_multi_query_single_hla_rs(
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
    calc_second_hm_without_dist_restriction: bool,
) -> Result<(
    HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    HashMap<String, Option<f64>>,
    u64,
), Box<dyn Error>> {
    let start_time = std::time::Instant::now();

    // Load Kds from Target Set into HashMap
    let mut epi_kd_dict = process_kd_info_vec_rs(kd_file_path)?;

    println!("[rust] Kds processing succeeded.");
    println!("[rust] Length of epi_kd_dict: {}", epi_kd_dict.len());

    // Filter epi_kd_dict based on target_epis_at_hla
    if !(target_epis_at_hla.len() == 1 && target_epis_at_hla[0] == "all") {
        epi_kd_dict.retain(|k, _| target_epis_at_hla.contains(&k.as_str()));
    }

    let mut logKinv_entropy_multi_query_dict: HashMap<String, HashMap<String, Option<(f64, f64)>>> =
        HashMap::new();
    let mut logKinv_entropy_multi_query_no_dist_restriction_dict: HashMap<
        String,
        HashMap<String, Option<(f64, f64)>>,
    > = HashMap::new();

    // Compute gamma_d_coeff
    let gamma_d_coeff = compute_gamma_d_coeff_rs(dist_metric, data_matrix_dir)?;

    println!("Gamma D coefficient: {}", gamma_d_coeff);

    // Initialize epi_log_count_dict and epi_log_conc_dict
    let mut epi_log_count_dict = HashMap::new();
    let mut epi_log_conc_dict = HashMap::new();
    for key in epi_kd_dict.keys() {
        epi_log_count_dict.insert(key.clone(), 0.0);
        epi_log_conc_dict.insert(key.clone(), 0.0);
    }
    let use_counts_concs: bool = false;

    for query_epi in &query_epi_list {
        // Load or Generate Distances from Query into HashMap
        let mut epi_dist_dict =
            process_distance_info_vec_rs(&dist_file_info, query_epi, dist_metric, data_matrix_dir, max_target_num)?;

        println!("[rust] Distance processing succeeded.");
        println!("[rust] Length of epi_dist_dict: {}", epi_dist_dict.len());

        // Filter epi_dist_dict based on target_epis_at_hla
        if !(target_epis_at_hla.len() == 1 && target_epis_at_hla[0] == "all") {
            epi_dist_dict.retain(|k, _| target_epis_at_hla.contains(&k.as_str()));
        }

        // Check if both HashMaps have the same keys
        if !have_same_keys(&epi_dist_dict, &epi_kd_dict) {
            print_keys_diff(&epi_dist_dict, &epi_kd_dict);
            return Err("The keys of epi_dist_dict and epi_kd_dict do not match.".into());
        }

        //////////    MODEL COMPUTATIONS    /////////
        let mut logKinv_entropy_dict: HashMap<String, Option<(f64, f64)>> = HashMap::new();
        let mut logKinv_entropy_no_dist_restriction_dict: HashMap<String, Option<(f64, f64)>> =
            HashMap::new();

        if compute_logKinv_and_entropy {
            if target_epis_at_hla.is_empty() {
                logKinv_entropy_dict.insert("no_target".to_string(), None);
            } else {
                logKinv_entropy_dict = compute_logKinv_and_entropy_dict_rs(
                    &epi_dist_dict,
                    &epi_kd_dict,
                    &epi_log_count_dict,
                    &epi_log_conc_dict,
                    &gamma_d_values,
                    &gamma_logkd_values,
                    gamma_d_coeff,
                    d_ub,
                    d_lb,
                    use_counts_concs,
                );

                if calc_second_hm_without_dist_restriction {
                    logKinv_entropy_no_dist_restriction_dict = compute_logKinv_and_entropy_dict_rs(
                        &epi_dist_dict,
                        &epi_kd_dict,
                        &epi_log_count_dict,
                        &epi_log_conc_dict,
                        &gamma_d_values,
                        &gamma_logkd_values,
                        gamma_d_coeff,
                        1000000000.0,
                        -1.0,
                        use_counts_concs,
                    );
                }
            }
        }

        if compute_logKinv_and_entropy {
            logKinv_entropy_multi_query_dict.insert(query_epi.to_string(), logKinv_entropy_dict);
            if calc_second_hm_without_dist_restriction {
                logKinv_entropy_multi_query_no_dist_restriction_dict.insert(
                    query_epi.to_string(),
                    logKinv_entropy_no_dist_restriction_dict,
                );
            }
        }
    }

    let mut logCh_dict: HashMap<String, Option<f64>> = HashMap::new();
    let log_n_wt = 9.2103;
    let log_h_num = 1.7917;
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
    Ok((
        logKinv_entropy_multi_query_dict,
        logKinv_entropy_multi_query_no_dist_restriction_dict,
        logCh_dict,
        end_time,
    ))
}




pub fn compute_log_rho_multi_query_rs(
    query_dict_logKInv_entropy_self: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    query_dict_logKInv_entropy_Ours_imm: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    query_dict_logKInv_entropy_Ours_non_imm: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    use_Ours_contribution: bool,
    query_dict_logKInv_entropy_Koncz_imm: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    query_dict_logKInv_entropy_Koncz_non_imm: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    use_Koncz_contribution: bool,
    query_dict_logKInv_entropy_Tesla_imm: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    query_dict_logKInv_entropy_Tesla_non_imm: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    use_Tesla_contribution: bool,
) -> Result<(HashMap<String, HashMap<String, HashMap<String, Option<f64>>>>, u64), String> {
    let start_time = std::time::Instant::now();

    let mut log_rho_multi_query_dict: HashMap<String, HashMap<String, HashMap<String, Option<f64>>>> = HashMap::new();
    
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
        let logKInv_entropy_Tesla_imm = match query_dict_logKInv_entropy_Tesla_imm.get(query_epi) {
            Some(values) => values,
            None => panic!("logKInv_entropy_Tesla_imm Key '{}' not found", query_epi),
        };
        let logKInv_entropy_Tesla_non_imm = match query_dict_logKInv_entropy_Tesla_non_imm.get(query_epi) {
            Some(values) => values,
            None => panic!("logKInv_entropy_Tesla_non_imm Key '{}' not found", query_epi),
        };

        // For the current query_epi: log_rho_dict[self_params][foreign_params] = rho_value
        let mut log_rho_dict: HashMap<String, HashMap<String, Option<f64>> > = HashMap::new();
        let mut self_term = 0.0;
        let mut iedb_imm_term = 0.0;
        let mut iedb_non_imm_term = 0.0;
        let mut tesla_non_imm_term = 0.0;
        let mut tesla_imm_term = 0.0;
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

            
            let mut all_keys_set: HashSet<&String> = HashSet::new();
            all_keys_set.extend(logKInv_entropy_Ours_imm.keys());
            all_keys_set.extend(logKInv_entropy_Ours_non_imm.keys());
            all_keys_set.extend(logKInv_entropy_Koncz_imm.keys());
            all_keys_set.extend(logKInv_entropy_Koncz_non_imm.keys());

            // Iterate over foreign dicts and extract log_K_inv and entropy for each.
            for foreign_key in all_keys_set {
                
                iedb_imm_term = 0.0;
                iedb_non_imm_term = 0.0;
                tesla_imm_term = 0.0;
                tesla_non_imm_term = 0.0;

                if use_Ours_contribution {

                    let target_epitopes_available = logKInv_entropy_Ours_imm.keys().filter(|&key| key != "no_target").count() > 1;
                    if target_epitopes_available {
                        let foreign_value = logKInv_entropy_Ours_imm.get(foreign_key);
                        let (log_K_inv_Ours_imm, entropy_Ours_imm) = match foreign_value {
                            Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                            _ => continue, // Skip if self_value is None
                        };
                        iedb_imm_term += (log_K_inv_Ours_imm - entropy_Ours_imm).exp();
                    }

                    let target_epitopes_available = logKInv_entropy_Ours_non_imm.keys().filter(|&key| key != "no_target").count() > 1;
                    if target_epitopes_available {
                        let foreign_value = logKInv_entropy_Ours_non_imm.get(foreign_key);
                        let (log_K_inv_Ours_non_imm, entropy_Ours_non_imm) = match foreign_value {
                            Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                            _ => continue, // Skip if self_value is None
                        };
                        iedb_non_imm_term += (log_K_inv_Ours_non_imm - entropy_Ours_non_imm).exp();
                    }
                } 

                if use_Koncz_contribution {

                    let target_epitopes_available = logKInv_entropy_Koncz_imm.keys().filter(|&key| key != "no_target").count() > 1;
                    if target_epitopes_available {
                        let foreign_value = logKInv_entropy_Koncz_imm.get(foreign_key);
                        let (log_K_inv_Koncz_imm, entropy_Koncz_imm) = match foreign_value {
                            Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                            _ => continue, // Skip if foreign_value is None or inner value is None
                        };
                        iedb_imm_term += (log_K_inv_Koncz_imm - entropy_Koncz_imm).exp();
                    }

                    let target_epitopes_available = logKInv_entropy_Koncz_non_imm.keys().filter(|&key| key != "no_target").count() > 1;
                    if target_epitopes_available {
                        let foreign_value = logKInv_entropy_Koncz_non_imm.get(foreign_key);
                        let (log_K_inv_Koncz_non_imm, entropy_Koncz_non_imm) = match foreign_value {
                            Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                            _ => continue, // Skip if self_value is None
                        };
                        iedb_non_imm_term += (log_K_inv_Koncz_non_imm - entropy_Koncz_non_imm).exp();
                    }
                }

                if use_Tesla_contribution {
                    
                    let target_epitopes_available = logKInv_entropy_Tesla_imm.keys().filter(|&key| key != "no_target").count() > 1;
                    if target_epitopes_available {
                        let foreign_value = logKInv_entropy_Tesla_imm.get(foreign_key);
                        let (log_K_inv_Tesla_imm, entropy_Tesla_imm) = match foreign_value {
                            Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                            _ => continue, // Skip if foreign_value is None or inner value is None
                        };
                        tesla_imm_term += (log_K_inv_Tesla_imm - entropy_Tesla_imm).exp();
                    }

                    let target_epitopes_available = logKInv_entropy_Tesla_non_imm.keys().filter(|&key| key != "no_target").count() > 1;
                    if target_epitopes_available {
                        let foreign_value = logKInv_entropy_Tesla_non_imm.get(foreign_key);
                        let (log_K_inv_Tesla_non_imm, entropy_Tesla_non_imm) = match foreign_value {
                            Some(Some((log_K_inv, entropy))) => (*log_K_inv, *entropy),
                            _ => continue, // Skip if self_value is None
                        };
                        tesla_non_imm_term += (log_K_inv_Tesla_non_imm - entropy_Tesla_non_imm).exp();
                    }
                }
                log_rho = -self_term + iedb_imm_term - iedb_non_imm_term + tesla_imm_term - tesla_non_imm_term;

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

// Helper function to get the nth element from a comma-separated string inside parentheses
pub fn get_nth_element(s: &str, n: usize) -> String {
    // Check if the string starts with '(' and ends with ')'
    if s.starts_with('(') && s.ends_with(')') {
        // Trim the parentheses
        let trimmed = &s[1..s.len()-1];
        // Split the trimmed string by commas
        trimmed.split(',')
               .nth(n)
               .map(|elem| elem.trim().to_string())
               .unwrap_or(String::new())
    } else {
        // Handle the original comma-separated format
        s.split(',')
         .nth(n)
         .map(|elem| elem.trim().to_string())
         .unwrap_or(String::new())
    }
}
#[derive(Debug)]
enum MyError {
    SerdePickleError(serde_pickle::Error),
}

impl Error for MyError {}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MyError::SerdePickleError(err) => write!(f, "Serde Pickle Error: {}", err),
        }
    }
}
pub fn immunogenicity_dict_from_pickle_files_rs(
    file_paths: &[&str],
) -> Result<HashMap<String, HashMap<String, Vec<f64>>>, Box<dyn StdError>> {
    // Load pickle files
    let start_time = std::time::Instant::now();
    let pickle_contents_vec = load_all_pkl_files(file_paths)?;
    let end_time = std::time::Instant::now();
    let elapsed_time = (end_time - start_time).as_secs_f64();
    println!("Elapsed time for load_all_pkl_files(): {:.6} seconds", elapsed_time);

    // Initialize the result dictionary
    let mut immunogenicity_dict: HashMap<String, HashMap<String, Vec<f64>>> = HashMap::new();

    // Parallel processing for each PickleContents
    let result: Vec<HashMap<String, HashMap<String, Vec<f64>>>> = pickle_contents_vec
        .par_iter()
        .map(|pickle_contents| {
            let PickleContents {
                logKInv_entropy_self_dict,
                logCh_dict,
                log_rho_multi_query_dict,
                ..
            } = pickle_contents;

            // Local result dictionary for this thread
            let mut local_immunogenicity_dict: HashMap<String, HashMap<String, Vec<f64>>> = HashMap::new();

            // Cache logCh_dict lookups
            let mut logCh_cache = HashMap::new();

            // Process each entry in log_rho_multi_query_dict
            for (query_epi, log_rho_dict) in log_rho_multi_query_dict {
                for (self_params, foreign_params_dict) in log_rho_dict {
                    let gamma_logKd_str = get_nth_element(&self_params, 1);

                    // Generate variants of the key
                    let variants = generate_variants(&gamma_logKd_str);

                    // Attempt to find a matching entry in logCh_dict, cache the result
                    let log_Ch = logCh_cache
                        .entry(gamma_logKd_str.clone())
                        .or_insert_with(|| {
                            variants
                                .iter()
                                .filter_map(|variant| logCh_dict.get(variant).and_then(|&val| val))
                                .next()
                        });

                    // Handle the result of the lookup
                    if let Some(log_Ch) = log_Ch {
                        if let Some(inner_map) = logKInv_entropy_self_dict.get(query_epi.as_str()) {
                            if let Some(Some((K_Inverse_self, Entropy_self))) =
                                inner_map.get(self_params.as_str())
                            {
                                let log_non_rho = *log_Ch + K_Inverse_self - Entropy_self;

                                for (foreign_params, log_rho_opt) in foreign_params_dict {
                                    if let Some(log_rho) = log_rho_opt {
                                        let nb = (log_non_rho + log_rho).exp();

                                        local_immunogenicity_dict
                                            .entry(foreign_params.clone())
                                            .or_insert_with(HashMap::new)
                                            .entry(self_params.clone())
                                            .or_insert_with(Vec::new)
                                            .push(nb);
                                    }
                                }
                            }
                        }
                    } else {
                        // Create a formatted list of variants for the error message
                        let variant_strings: Vec<&str> =
                            variants.iter().map(|s| s.as_str()).collect();
                        panic!(
                            "None of the key variants {:?} found in logCh_dict",
                            &variant_strings
                        );
                    }
                }
            }

            local_immunogenicity_dict // Return local results
        })
        .collect(); // Collect results from all threads

    // Merge results from all threads into the main immunogenicity_dict
    for thread_result in result {
        for (foreign_params, self_params_map) in thread_result {
            for (self_params, nb_vec) in self_params_map {
                immunogenicity_dict
                    .entry(foreign_params.clone())
                    .or_insert_with(HashMap::new)
                    .entry(self_params.clone())
                    .or_insert_with(Vec::new)
                    .extend(nb_vec);
            }
        }
    }

    Ok(immunogenicity_dict)
}




pub fn calculate_auc_dict_from_pickle_files_rs(
    file_paths_list1: Vec<&str>,
    file_paths_list2: Vec<&str>,
) -> Result<HashMap<String, HashMap<String, f64>>, Box<dyn Error>> {
    /*
    non parallel version
     */
    // Call immunogenicity_dict_from_pickle_files_rs for both lists
    let immunogenicity_dict_1 = immunogenicity_dict_from_pickle_files_rs(&file_paths_list1)?;
    let immunogenicity_dict_2 = immunogenicity_dict_from_pickle_files_rs(&file_paths_list2)?;

    // Initialize the AUC dictionary
    let mut auc_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

    // Iterate over the first dictionary
    for (foreign_params, inner_map_1) in immunogenicity_dict_1.iter() {
        // Check if the second dictionary contains the same foreign_params
        if let Some(inner_map_2) = immunogenicity_dict_2.get(foreign_params) {
            // Initialize the inner map for the AUC values
            let mut computed_values_map: HashMap<String, f64> = HashMap::new();
            
            // Iterate over the inner map of the first dictionary
            for (self_params, vector_1) in inner_map_1.iter() {
                // Check if the second inner map contains the same self_params
                if let Some(vector_2) = inner_map_2.get(self_params) {
                    // Calculate AUC and insert into the inner map
                    if let Ok(auc_value) = calculate_auc(vector_1, vector_2) {
                        computed_values_map.insert(self_params.to_string(), auc_value);
                    }
                }
            }

            // Insert the inner map into the AUC dictionary
            auc_dict.insert(foreign_params.to_string(), computed_values_map);
        }
    }

    Ok(auc_dict)
}
// pub fn calculate_auc_dict_from_pickle_files_rs(
//     file_paths_list1: Vec<&str>,
//     file_paths_list2: Vec<&str>,
// ) -> Result<HashMap<String, HashMap<String, f64>>, Box<dyn Error>> {
//     /*
//     PARALLEL VERSION
//     Compute auc_dict[foreign_params][self_params]
//      */

//     // Call immunogenicity_dict_from_pickle_files_rs for both lists
//     let immunogenicity_dict_1 = immunogenicity_dict_from_pickle_files_rs(&file_paths_list1)?;
//     let immunogenicity_dict_2 = immunogenicity_dict_from_pickle_files_rs(&file_paths_list2)?;

//     // Parallelize the loop using rayon
//     let auc_dict: HashMap<String, HashMap<String, f64>> = immunogenicity_dict_1
//         .par_iter()
//         .filter_map(|(foreign_params, inner_map_1)| {
//             immunogenicity_dict_2.get(foreign_params).map(|inner_map_2| {
//                 let computed_values_map: HashMap<String, f64> = inner_map_1
//                     .iter()
//                     .filter_map(|(self_params, vector_1)| {
//                         inner_map_2.get(self_params).and_then(|vector_2| {
//                             calculate_auc(vector_1, vector_2).ok()
//                         })
//                         .map(|n| (self_params.to_string(), n))
//                     })
//                     .collect();
//                 (foreign_params.to_string(), computed_values_map)
//             })
//         })
//         .collect();

//     Ok(auc_dict)
// }


