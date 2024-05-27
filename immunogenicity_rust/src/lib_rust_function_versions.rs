/////////////////////////////////////////////////////////////////////////
////////////   Import from other scripts via the main module.   //////////////
use crate::*; // Import from the main module (lib.rs)
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



// pub fn log_sum(v: &[f64]) -> Option<f64> {
//     if let Some(&ma) = v.iter().max_by(|&&a, &&b| a.partial_cmp(&b).unwrap()) {
//         if ma == f64::NEG_INFINITY {
//             Some(f64::NEG_INFINITY)
//         } else {
//             let sum_exp = v.iter().map(|&x| (x - ma).exp()).sum::<f64>();
//             Some(sum_exp.ln() + ma)
//         }
//     } else {
//         None
//     }
// }
// Define a custom error type for NaN or infinity values
#[derive(Debug)]
struct InputError(&'static str);

// Implement std::fmt::Display trait for InputError
use std::fmt;
impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
// Implement std::error::Error trait for InputError
use std::error::Error;
impl Error for InputError {}
use std::num::ParseFloatError;
pub fn log_sum(v: &[f64]) -> Result<f64, InputError> {
    if v.is_empty() {
        // If the input slice is empty, return negative infinity
        return Ok(f64::NEG_INFINITY);
    }

    // Check for NaN or infinity values in the input slice
    if v.iter().any(|&x| x.is_nan() || x == f64::INFINITY || x == f64::NEG_INFINITY) {
        // Return an error indicating that the input contains NaN or infinity values
        return Err(InputError("Input contains NaN or INFINITY values"));
    }

    // Find the maximum value among the input values
    let ma = *v.iter().max_by(|&&a, &&b| a.partial_cmp(&b).unwrap()).unwrap();

    if ma == f64::NEG_INFINITY {
        // If the maximum value is negative infinity, return negative infinity
        Ok(f64::NEG_INFINITY)
    } else {
        // Calculate the sum of exponential values relative to the maximum value
        let sum_exp = v.iter().map(|&x| (x - ma).exp()).sum::<f64>();
        Ok(sum_exp.ln() + ma)
    }
}



pub fn compute_entropy(v: &[f64], log_z: f64) -> Option<f64> {
    // Calculate the probabilities by exponentiating each value in the vector
    let probs: Vec<f64> = v.iter().map(|val| E.powf(val - log_z)).collect();

    // Calculate the entropy
    let entropy = probs.iter().filter_map(|&prob| {
        if prob > 0.0 {
            Some(-prob * prob.ln()) // Avoid NaN for 0 probability
        } else {
            None
        }
    }).sum::<f64>();

    // Check if entropy is NaN or infinite
    if entropy.is_nan() || entropy.is_infinite() {
        None
    } else {
        Some(entropy)
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
    d_PS_threshold: f64,
    d_NS_cutoff: f64,
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

    // Masking based on PS and NS thresholds on distances
    let mask = &epi_dist_array.map(|&x| x < d_PS_threshold) & epi_dist_array.map(|&x| x > d_NS_cutoff);

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
                        Some(entropy) => entropy,
                        None => {
                            panic!("Error calculating entropy: entropy value is None.");
                        }
                    };
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