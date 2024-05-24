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

                    if target_epi_dist_dict.len() > 10 {
                        for (key, value) in target_epi_dist_dict.iter().take(10) {
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


pub fn process_kd_info_vec(csv_kds_file_path: &str) -> Result<HashMap<String, f64>, String> {
    

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

            if target_epi_kd_dict.len() > 10 {
                for (key, value) in target_epi_kd_dict.iter().take(10) {
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



pub fn log_sum(v: &[f64]) -> Option<f64> {
    if let Some(&ma) = v.iter().max_by(|&&a, &&b| a.partial_cmp(&b).unwrap()) {
        if ma == f64::NEG_INFINITY {
            Some(f64::NEG_INFINITY)
        } else {
            let sum_exp = v.iter().map(|&x| (x - ma).exp()).sum::<f64>();
            Some(sum_exp.ln() + ma)
        }
    } else {
        None
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

// Function to generate parameter values with either linear or logarithmic spacing
pub fn generate_parameter_values(lower_bound: f64, upper_bound: f64, num_values: usize, use_log_spacing: bool) -> Vec<f64> {
    if use_log_spacing {
        (0..num_values)
            .map(|i| lower_bound * ((upper_bound / lower_bound).powf(i as f64 / (num_values - 1) as f64)))
            .collect()
    } else {
        (0..num_values)
            .map(|i| lower_bound + (upper_bound - lower_bound) * i as f64 / (num_values - 1) as f64)
            .collect()
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

    let mut log_Kinv_dict: HashMap<String, Option<(f64, f64)>> = HashMap::new();

    let epi_dist_dict_arc = Arc::new(epi_dist_dict.clone());
    let epi_kd_dict_arc = Arc::new(epi_kd_dict.clone());

    if use_counts_concs {   
        // The extra loop iterations associated with epi_log_count_dict and epi_log_conc_dict can nearly double the
        // run time of this function. Try to find ways to improve efficiency if they need to be used. 
        let epi_log_count_dict_arc = Arc::new(epi_log_count_dict.clone());
        let epi_log_conc_dict_arc = Arc::new(epi_log_conc_dict.clone());

        let epi_dist_dict_ref = &epi_dist_dict_arc;
        let epi_kd_dict_ref = &epi_kd_dict_arc;
        let epi_log_count_dict_ref = &epi_log_count_dict_arc;
        let epi_log_conc_dict_ref = &epi_log_conc_dict_arc;

        log_Kinv_dict = gamma_d_values
            .par_iter()
            .flat_map(|&gamma_d_value| {
 
                let epi_dist_dict_ref = &epi_dist_dict_ref;
                let epi_kd_dict_ref = &epi_kd_dict_ref;
                let epi_log_count_dict_ref = &epi_log_count_dict_ref;
                let epi_log_conc_dict_ref = &epi_log_conc_dict_ref;

                gamma_logkd_values.par_iter().map(move |&gamma_logKd_value| {
                    let mut values = Vec::with_capacity(epi_dist_dict_ref.len());
                    for (epi, &dist) in epi_dist_dict_ref.iter() {
                        if dist < d_PS_threshold && dist > d_NS_cutoff {
                            if let Some(&kd) = epi_kd_dict_ref.get(epi) {
                                if let Some(&log_count) = epi_log_count_dict_ref.get(epi) {
                                    if let Some(&log_conc) = epi_log_conc_dict_ref.get(epi) {
                                        let value =
                                            -gamma_d_value * dist * gamma_d_coeff - gamma_logKd_value * kd.ln() + log_count + log_conc;
                                        values.push(value);
                                    }
                                }
                            }
                        }
                    }
                    let result = log_sum(&values); // log_Kinv or "logZ"
                    let entropy = if let Some(result) = result {
                        compute_entropy(&values, result)
                    } else {
                        None
                    };
                    let key = tuple_to_string(&(gamma_d_value, gamma_logKd_value));
                    (key, Some((result.unwrap_or(0.0), entropy.unwrap_or(0.0))))
                })
            })
            .collect();

    } else {

        let epi_dist_dict_ref = &epi_dist_dict_arc;
        let epi_kd_dict_ref = &epi_kd_dict_arc;

        log_Kinv_dict = gamma_d_values
            .par_iter()
            .flat_map(|&gamma_d_value| {

                let epi_dist_dict_ref = &epi_dist_dict_ref;
                let epi_kd_dict_ref = &epi_kd_dict_ref;

                gamma_logkd_values.par_iter().map(move |&gamma_logKd_value| {
                    let mut values = Vec::with_capacity(epi_dist_dict_ref.len());
                    for (epi, &dist) in epi_dist_dict_ref.iter() {
                        if dist < d_PS_threshold && dist > d_NS_cutoff {
                            if let Some(&kd) = epi_kd_dict_ref.get(epi) {
                                let value =
                                    -gamma_d_value * dist * gamma_d_coeff - gamma_logKd_value * kd.ln();
                                values.push(value);
                            }
                        }
                    }
                    let result = log_sum(&values); // log_Kinv or "logZ"
                    let entropy = if let Some(result) = result {
                        compute_entropy(&values, result)
                    } else {
                        None
                    };
                    let key = tuple_to_string(&(gamma_d_value, gamma_logKd_value));
                    (key, Some((result.unwrap_or(0.0), entropy.unwrap_or(0.0))))
                }) 
            })
            .collect();
    }

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

                let log_sum_values = log_sum(&values);
                let result = match log_sum_values {
                    Some(sum) => log_n_wt - log_h_num - sum,
                    None => return (gamma_logKd_value.to_string(), None),
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

                let log_sum_values = log_sum(&values);
                let result = match log_sum_values {
                    Some(sum) => log_n_wt - log_h_num - sum,
                    None => return (gamma_logKd_value.to_string(), None),
                };
                let key = gamma_logKd_value.to_string();
                (key, Some(result))

            })
            .collect::<HashMap<_, _>>();        
    }

    log_Ch_dict
}