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

// This script contains I/O functions that do not belong to an impl block for a custom type.

/////////////////////////////////////////////////////////////////
////////////   Use types defined in lib.rs   ////////////////////
use crate::DistanceMetricType;
use crate::TargetEpiDistances;
use crate::TargetEpiKds;
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

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