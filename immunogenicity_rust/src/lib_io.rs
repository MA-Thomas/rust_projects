use std::error::Error;
use std::fs::{File, remove_file, create_dir_all};
use std::io::BufReader;
// use std::fs;
use std::io::{self, BufRead, Write, Read};
use std::io::BufWriter;

use tar::Builder;
use flate2::write::GzEncoder;
use flate2::Compression;

use flate2::read::GzDecoder;
use tar::Archive;
use std::collections::HashMap;
use serde_pickle::from_reader;
use serde::Deserialize;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// This script contains I/O functions that do not belong to an impl block for a custom type.

/////////////////////////////////////////////////////////////////
////////////   Use types defined in lib.rs   ////////////////////
use crate::lib_data_structures_auxiliary_functions::DistanceMetricType;
use crate::lib_data_structures_auxiliary_functions::TargetEpiDistances;
use crate::lib_data_structures_auxiliary_functions::TargetEpiKds;
use crate::lib_data_structures_auxiliary_functions::DistanceMetricContext;
use crate::lib_data_structures_auxiliary_functions::EpitopeDistanceStruct;
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////


pub fn set_json_path(data_matrix_dir: &str, dm_type: &DistanceMetricType) -> String {
    match dm_type {
        DistanceMetricType::all_tcr_all_combos_model => data_matrix_dir.to_owned()  + "all_tcr_all_combos_model.json",
        DistanceMetricType::epidist_blosum62_distance => data_matrix_dir.to_owned()  + "epidist_blosum62_distance.json",
        DistanceMetricType::hamming => "".to_string()
    }
}
pub fn evaluate_context(c: DistanceMetricContext) -> Result<EpitopeDistanceStruct,Box<dyn Error>> {
    match c.metric {
        DistanceMetricType::all_tcr_all_combos_model => EpitopeDistanceStruct::load_from_json(&c.json_path),
        DistanceMetricType::epidist_blosum62_distance => EpitopeDistanceStruct::load_from_json(&c.json_path),
        DistanceMetricType::hamming => EpitopeDistanceStruct::load_hamming(9)
    }
}

pub fn parse_fasta(file_path: &str, max_target_num: usize, buffer_capacity: Option<usize>) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut epitopes = Vec::new();
    let mut current_epitope = match buffer_capacity {
        Some(capacity) => String::with_capacity(capacity),
        None => String::with_capacity(30), // Default capacity
    };

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if !current_epitope.is_empty() {
                epitopes.push(std::mem::take(&mut current_epitope));
            }
        } else {
            current_epitope.push_str(&line);
        }
    }

    if !current_epitope.is_empty() {
        epitopes.push(current_epitope);
    }

    // Sort the vector of epitopes alphabetically.
    epitopes.sort();

    // Only return up to <max_target_num> of the sorted epitopes.
    if epitopes.len() > max_target_num {
        epitopes.truncate(max_target_num);
    }

    Ok(epitopes)
}

pub fn save_epitopes_distances_to_tar_gz(csv_distances_file_path: &str, distances: &[TargetEpiDistances]) -> Result<(), Box<dyn Error>> {
    // Extract directory and file name without extension
    let (directory, file_stem) = match (
        std::path::Path::new(csv_distances_file_path).parent(),
        std::path::Path::new(csv_distances_file_path).file_stem(),
    ) {
        (Some(dir), Some(name)) => (dir.to_string_lossy().to_string(), name.to_string_lossy().to_string()),
        _ => return Err("Invalid file path".into()),
    };

    // Create the directory if it doesn't exist
    create_dir_all(&directory)?;

    // Create a .tar.gz file in the specified directory
    let tar_gz_file = File::create(format!("{}/{}.tar.gz", directory, file_stem))?;
    let tar = GzEncoder::new(tar_gz_file, Compression::default());
    let mut tar_builder = Builder::new(tar);

    // Create the CSV file and write distances to it
    let csv_file_name = format!("{}/{}.csv", directory, file_stem);
    let mut csv_file = BufWriter::new(File::create(&csv_file_name)?);
    for dist in distances {
        writeln!(csv_file, "{},{}", dist.epitope, dist.distance)?;
    }
    csv_file.flush()?;

    // Add the CSV file to the .tar.gz archive
    let csv_file_path = std::path::Path::new(&csv_file_name);
    let csv_file_name_str = csv_file_path.file_name().unwrap().to_str().unwrap();
    tar_builder.append_path_with_name(&csv_file_path, csv_file_name_str)?;

    // Finish writing the archive
    tar_builder.finish()?;

    // Clean up temporary CSV file
    remove_file(&csv_file_name)?;

    Ok(())
}


pub fn load_epitopes_distances_from_tar_gz(csv_distances_file_path: &str) -> Result<Vec<TargetEpiDistances>, Box<dyn Error>> {
    println!("[rust] In load_epitopes_distances_from_tar_gz. csv_distances_file_path: {}", csv_distances_file_path);
    // Open the .tar.gz file for reading
    let tar_gz_file = File::open(csv_distances_file_path)?;
    let tar = GzDecoder::new(tar_gz_file);
    let mut archive = Archive::new(tar);

    // Find the CSV file within the archive
    let mut csv_file = archive.entries()?;
    let mut csv_file_entry = match csv_file.next() {
        Some(Ok(entry)) => entry,
        _ => return Err("No CSV file found in the archive".into()),
    };

    // Read distances and epitopes from the CSV file
    let mut target_epi_distances = Vec::new();
    let mut csv_contents = String::new();
    csv_file_entry.read_to_string(&mut csv_contents)?;

    // Split the contents into lines
    let mut lines = csv_contents.lines();

    // Check if the header line contains only text (no numerical values)
    if let Some(header) = lines.next() {
        if header.split(',').all(|field| field.trim().parse::<f64>().is_err()) {
            println!("[rust] Header line skipped: {}", header);
        } 
    }

    for line in lines {
        let mut fields = line.split(',');
        let epitope = match fields.next() {
            Some(epitope) => epitope,
            None => return Err("Empty line found in CSV".into()),
        };
        let distance_str = match fields.next() {
            Some(distance_str) => distance_str,
            None => return Err("Missing distance in CSV".into()),
        };
        if let Ok(distance) = distance_str.trim().parse::<f64>() {
            target_epi_distances.push(TargetEpiDistances {
                epitope: epitope.to_string(),
                distance: distance,
            });
        } else {
            return Err(format!("Failed to parse distance: {}", distance_str.trim()).into());
        }
    }

    Ok(target_epi_distances)
}


pub fn load_epitopes_kds_from_tar_gz(csv_kds_file_path: &str) -> Result<Vec<TargetEpiKds>, Box<dyn Error>> {
    println!("[rust] In load_epitopes_kds_from_tar_gz. csv_kds_file_path: {}", csv_kds_file_path);
    // Open the .tar.gz file for reading
    let tar_gz_file = File::open(csv_kds_file_path)?;
    let tar = GzDecoder::new(tar_gz_file);
    let mut archive = Archive::new(tar);

    // Find the CSV file within the archive
    let mut csv_file = archive.entries()?;
    let mut csv_file_entry = match csv_file.next() {
        Some(Ok(entry)) => entry,
        _ => return Err("No CSV file found in the archive".into()),
    };

    // Read epitopes and Kds from the CSV file
    let mut target_epi_kds = Vec::new();
    let mut csv_contents = String::new();
    csv_file_entry.read_to_string(&mut csv_contents)?;

    // Split the contents into lines
    let mut lines = csv_contents.lines();

    // Check if the header line contains only text (no numerical values). (There should be no header for distance files)
    if let Some(header) = lines.next() {
        if header.split(',').all(|field| field.trim().parse::<f64>().is_err()) {
            println!("[rust] Header line skipped: {}", header);
        } else {
            return Err("Header line contains numerical values, skipping not possible".into());
        }
    }

    for line in lines {
        let mut fields = line.split(',');
        let epitope = match fields.next() {
            Some(epitope) => epitope,
            None => return Err("Empty line found in CSV".into()),
        };
        let kd_str = match fields.next() {
            Some(kd_str) => kd_str,
            None => return Err("Missing Kd in CSV".into()),
        };
        if let Ok(kd) = kd_str.trim().parse::<f64>() {
            target_epi_kds.push(TargetEpiKds {
                epitope: epitope.to_string(),
                Kd: kd,
            });
        } else {
            return Err(format!("Failed to parse Kd: {}", kd_str.trim()).into());
        }
    }
    Ok(target_epi_kds)
}


#[derive(Deserialize, Clone)]
pub struct PickleContents {
    // keys for logCh_dict: [self_params]
    // keys for log_rho_multi_query_dict: [epitope][self_params][foreign_params]
    // keys for logKInv_entropy_self_dict: [epitope][self_params]
    // keys for IEDB dicts: [epitope][foreign_params]
    pub logKInv_entropy_self_dict: HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    pub logCh_dict: HashMap<String, Option<f64>>,
    pub logKInv_entropy_Koncz_imm_epi_dict: HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    pub logKInv_entropy_Koncz_non_imm_epi_dict: HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    pub logKInv_entropy_Ours_imm_epi_dict: HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    pub logKInv_entropy_Ours_non_imm_epi_dict: HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    pub log_rho_multi_query_dict: HashMap<String, HashMap<String, HashMap<String, Option<f64>> > >,
}


pub fn filter_dict_for_inner_key(
    dict: &HashMap<String, HashMap<String, Option<(f64, f64)>>>,
    keys: &[&str],
) -> HashMap<String, HashMap<String, Option<(f64, f64)>>> {
    dict.iter()
        .filter_map(|(outer_key, inner)| {
            let filtered_inner: HashMap<String, Option<(f64, f64)>> = inner
                .iter()
                .filter_map(|(inner_key, value)| {
                    if keys.contains(&inner_key.as_str()) {
                        Some((inner_key.clone(), value.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            
            if filtered_inner.is_empty() {
                None
            } else {
                Some((outer_key.clone(), filtered_inner))
            }
        })
        .collect()
}
pub fn filter_dict_for_middle_key(
    dict: &HashMap<String, HashMap<String, HashMap<String, Option<f64>>>>,
    keys: &[&str],
) -> HashMap<String, HashMap<String, HashMap<String, Option<f64>>>> {
    dict.iter()
        .map(|(outer_key, middle_map)| {
            let filtered_middle_map: HashMap<String, HashMap<String, Option<f64>>> = middle_map
                .iter()
                .filter_map(|(middle_key, inner_map)| {
                    if keys.contains(&middle_key.as_str()) {
                        Some((middle_key.clone(), inner_map.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            (outer_key.clone(), filtered_middle_map)
        })
        .filter(|(_, filtered_middle_map)| !filtered_middle_map.is_empty())
        .collect()
}

pub fn load_epi_hla_pkl_file(pkl_file_path: &str, filter_on_self_param_keys: Option<&[&str]>) -> Result<PickleContents, Box<dyn Error>> {
    /*
    If filter_on_self_param_keys is &str, only filter the hashmap data in the .pkl corresponding to the provided outer key.
    (Except for logCh_dict which is only defined at inner_keys (i.e., only at self parameter sets))
     */
    // Open the file
    let mut file = File::open(pkl_file_path)?;
    
    // Read the contents of the file into a buffer
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Deserialize the data
    // let outputs: PickleContents = from_reader(&buffer[..])?;
    let mut outputs: PickleContents = from_reader(&buffer[..], Default::default())?;

    if let Some(self_param_keys) = filter_on_self_param_keys {

        // Filter the dictionaries by the specified key (self param set)
        outputs.logKInv_entropy_self_dict = filter_dict_for_inner_key(&outputs.logKInv_entropy_self_dict, self_param_keys);
        // outputs.logKInv_entropy_Koncz_imm_epi_dict = filter_dict_for_inner_key(&outputs.logKInv_entropy_Koncz_imm_epi_dict, -);
        // outputs.logKInv_entropy_Koncz_non_imm_epi_dict = filter_dict_for_inner_key(&outputs.logKInv_entropy_Koncz_non_imm_epi_dict, -);
        // outputs.logKInv_entropy_Ours_imm_epi_dict = filter_dict_for_inner_key(&outputs.logKInv_entropy_Ours_imm_epi_dict, -);
        // outputs.logKInv_entropy_Ours_non_imm_epi_dict = filter_dict_for_inner_key(&outputs.logKInv_entropy_Ours_non_imm_epi_dict, -);
        outputs.log_rho_multi_query_dict = filter_dict_for_middle_key(&outputs.log_rho_multi_query_dict, self_param_keys);
    }

    Ok(outputs)
}

pub fn load_all_pkl_files(file_paths: &[&str], filter_on_self_param_keys: Option<&[&str]>) -> Result<Vec<PickleContents>, Box<dyn Error>> {
    let pickle_contents_vec: Arc<Mutex<Vec<PickleContents>>> = Arc::new(Mutex::new(Vec::new()));

    file_paths.par_iter().for_each(|&file_path| {
        match load_epi_hla_pkl_file(file_path, filter_on_self_param_keys) {
            Ok(contents) => {
                let mut vec = pickle_contents_vec.lock().unwrap();
                vec.push(contents);
            },
            Err(e) => eprintln!("Failed to load and deserialize file {:?}: {}", file_path, e),
        }
    });

    let mut result_vec = Vec::new();
    {
        let vec = pickle_contents_vec.lock().unwrap();
        result_vec.extend_from_slice(&*vec);
    }
    Ok(result_vec)
}
// pub fn load_all_pkl_files(file_paths: &[&str]) -> Result<Vec<PickleContents>, Box<dyn Error>> {
//     let mut pickle_contents_vec = Vec::new();

//     for &file_path in file_paths {
//         match load_epi_hla_pkl_file(file_path) {
//             Ok(contents) => pickle_contents_vec.push(contents),
//             Err(e) => eprintln!("Failed to load and deserialize file {:?}: {}", file_path, e),
//         }
//     }

//     Ok(pickle_contents_vec)
// }
