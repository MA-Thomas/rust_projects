use std::collections::{HashMap, HashSet};
use std::fs::{File, remove_file, create_dir_all};
use std::io::BufReader;
use std::error::Error;

use pyo3::exceptions::PyException;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyRuntimeError;

use std::str::FromStr;
use std::fmt;

#[derive(Debug, serde::Deserialize)]
pub struct JSONData {
    // If fields may or may not appear in the json (e.g., epidist_blosum62_distance vs all_tcr_all_combos_model), use the Option<> type.
    // The struct names must match those used in the json file.
    d_i: Vec<f64>,
    euclid_coords: Option<HashMap<char, [f64; 2]>>,
    M_ab: HashMap<String, f64>,
    blosum62_reg: Option<f64>,
}

pub struct EpitopeDistanceStruct {
    amino_acid_dict: HashMap<char, usize>,
    di: Vec<f64>,
    mab: Vec<Vec<f64>>,
}

impl EpitopeDistanceStruct {
    pub fn new(amino_acids: &str, di: Vec<f64>, mab: Vec<Vec<f64>>) -> Self {
        let mut amino_acid_dict = HashMap::new();
        for (i, aa) in amino_acids.chars().enumerate() {
            amino_acid_dict.insert(aa, i);
        }

        EpitopeDistanceStruct {
            amino_acid_dict,
            di,
            mab,
        }
    }

    pub fn load_from_json(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let json_data: JSONData = serde_json::from_reader(reader)?;

        let amino_acids = "ACDEFGHIKLMNPQRSTVWY";
        let mut mab = vec![vec![0.0; amino_acids.len()]; amino_acids.len()];

        for (i, aa_a) in amino_acids.chars().enumerate() {
            for (j, aa_b) in amino_acids.chars().enumerate() {
                let akey = format!("{}->{}", aa_a, aa_b);
                let bkey = format!("{}->{}", aa_b, aa_a);
                let val = *json_data.M_ab.get(&akey).unwrap_or_else(|| json_data.M_ab.get(&bkey).unwrap_or(&0.0));
                mab[i][j] = val;
            }
        }

        Ok(EpitopeDistanceStruct::new(
            amino_acids,
            json_data.d_i,
            mab,
        ))
    }

    pub fn load_hamming(epitope_length: usize) -> Result<Self, Box<dyn Error>> {
        let amino_acids = "ACDEFGHIKLMNPQRSTVWY";
        
        let di = vec![1.0; epitope_length];

        let mut mab = vec![vec![0.0; amino_acids.len()]; amino_acids.len()];
        for (i, aa_a) in amino_acids.chars().enumerate() {
            for (j, aa_b) in amino_acids.chars().enumerate() {
                if i == j {
                   let val = 0.0;
                   mab[i][j] = val;
                } else{
                    let val = 1.0;
                    mab[i][j] = val;
                }

                
            }
        }  
        Ok(EpitopeDistanceStruct::new(
            amino_acids,
            di,
            mab,
        ))      
    }

    pub fn gamma_d_coeff<'a>(&'a self) -> Result<f64, Box<dyn Error>> {
        let mut melems = Vec::new();
        
        for (i, row) in self.mab.iter().enumerate() {
            let row2: Vec<_> = row.iter().enumerate()
                .filter(|&(j, _)| j != i) // Exclude diagonal element
                .map(|(_, &val)| val)
                .collect();
            melems.extend_from_slice(&row2);
        }

        let sum: f64 = melems.iter().sum();
        let mean = sum / (melems.len() as f64);

        if mean.is_nan() {
            Err("Mean is NaN".into())
        } else {
            Ok(1.0 / mean)
        }
    }

    pub fn epitope_dist<'a>(&'a self, epi_a: &'a str, epi_b: &'a str) -> Result<(f64, Vec<&'a str>), Box<dyn Error>> {
        /*
        Compute the minimum 9-mer distance between epi_a and epi_b. Both epitopes must be at least 9 aa long.

        We introduce a named lifetime parameter ('a) to tie the lifetimes of the input references and 
        the references stored in the invalid_epitopes vector that is returned. 
        This ensures that all references have compatible lifetimes and prevents any lifetime-related errors.
        Rust ensures that references returned from a function cannot outlive the data they refer to, and we need to 
        ensure that all references have compatible lifetimes to satisfy this constraint.
        */
        if epi_a.len() < 9 || epi_b.len() < 9 {
            return Err("Epitope lengths must be at least 9 characters".into());
        }
    
        let mut invalid_epitopes = Vec::new();
        let mut min_distance = f64::INFINITY;
    
        for i in 0..=epi_a.len() - 9 {
            for j in 0..=epi_b.len() - 9 {
                let sub_epi_a = &epi_a[i..i+9];
                let sub_epi_b = &epi_b[j..j+9];
                let mut sum = 0.0;
    
                for (a, b) in sub_epi_a.chars().zip(sub_epi_b.chars()) {
                    let idx_a = match self.amino_acid_dict.get(&a) {
                        Some(idx) => *idx,
                        None => {
                            invalid_epitopes.push(sub_epi_a);
                            break;
                        }
                    };
    
                    let idx_b = match self.amino_acid_dict.get(&b) {
                        Some(idx) => *idx,
                        None => {
                            invalid_epitopes.push(sub_epi_b);
                            break;
                        }
                    };
    
                    sum += self.mab[idx_a][idx_b] * self.di[i];
                }
    
                if sum < min_distance {
                    min_distance = sum;
                }
            }
        }
    
        Ok((min_distance, invalid_epitopes))
    }
}        


pub enum DistanceMetricType {
    hamming,
    epidist_blosum62_distance,
    all_tcr_all_combos_model,
}
pub struct DistanceMetricContext {
    pub metric: DistanceMetricType,
    pub json_path: String,
}


#[derive(Debug)]
#[derive(Clone)]
pub struct TargetEpiDistances {
    pub epitope: String,
    pub distance: f64,
}

pub struct TargetEpiKds {
    pub epitope: String,
    pub Kd: f64,
}


pub enum ImmunogenicityValue {
    Scalar(f64),
    Vector(Vec<f64>),
}
// Implement AsMut for ImmunogenicityValue to allow conversion to &mut Vec<f64>
impl AsMut<Vec<f64>> for ImmunogenicityValue {
    fn as_mut(&mut self) -> &mut Vec<f64> {
        match self {
            ImmunogenicityValue::Scalar(_) => {
                *self = ImmunogenicityValue::Vector(Vec::new()); // Convert Scalar to Vector if needed
                if let ImmunogenicityValue::Vector(ref mut vec) = self {
                    vec
                } else {
                    unreachable!(); // This should never happen due to the above match arm
                }
            }
            ImmunogenicityValue::Vector(ref mut vec) => vec,
        }
    }
}



pub fn tuple_to_string(tuple: &(f64, f64)) -> String {
    format!("{:?}", tuple)
}

// Helper function to check if two HashMaps have the same keys
pub fn have_same_keys(map1: &HashMap<String, f64>, map2: &HashMap<String, f64>) -> bool {
    let keys1: HashSet<_> = map1.keys().collect();
    let keys2: HashSet<_> = map2.keys().collect();
    keys1 == keys2
}

// Define a function to generate variants of the key.
/*
This is useful when looking for a hashmap value associated with a key such as '1.0'
when the true key is '1'. The two forms, '1.0' and '1' are considered here as variants 
of the same key. 
*/
pub fn generate_variants(key: &str) -> Vec<String> {
    let mut variants = vec![key.to_string()];

    // Add additional variants as needed
    if let Ok(float_value) = key.parse::<f64>() {
        variants.push(format!("{:.1}", float_value)); // Adjust precision as needed
        variants.push(format!("{:.0}", float_value)); // Integer format
    }

    variants
}
pub fn print_keys_diff<K, V>(map1: &HashMap<K, V>, map2: &HashMap<K, V>)
where
    K: std::hash::Hash + Eq + std::fmt::Debug,
{
    let keys1: HashSet<_> = map1.keys().collect();
    let keys2: HashSet<_> = map2.keys().collect();

    let only_in_map1: HashSet<_> = keys1.difference(&keys2).collect();
    let only_in_map2: HashSet<_> = keys2.difference(&keys1).collect();

    println!("Keys found only in map1 ({}):", only_in_map1.len());
    // for key in &only_in_map1 {
    //     println!("{:?}", key);
    // }

    println!("Keys found only in map2 ({}):", only_in_map2.len());
    // for key in &only_in_map2 {
    //     println!("{:?}", key);
    // }
}

// Auxiliary function to convert PyDict to HashMap<String, Option<(f64, f64)>>
pub fn convert_nested_PyDict_to_HashMap(dict: &PyDict) -> PyResult<HashMap<String, HashMap<String, Option<(f64, f64)>>>> {
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

#[derive(Debug)]
pub enum CalculationError {
    InvalidInput,
    RocCurveError,
    PrCurveError,
    InvalidAucType,
}
// Implement the Error trait for CalculationError
impl fmt::Display for CalculationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CalculationError::InvalidInput => write!(f, "Invalid input error occurred"),
            CalculationError::RocCurveError => write!(f, "ROC curve calculation error"),
            CalculationError::PrCurveError => write!(f, "Precision-recall curve calculation error"),
            CalculationError::InvalidAucType => write!(f, "Invalid AUC type specified"),
        }
    }
}
// Implementing the Error trait allows CalculationError to be used as a Box<dyn Error>
impl Error for CalculationError {}


pub enum AucType {
    ROC,
    PR,
}
impl FromStr for AucType {
    type Err = CalculationError;

    fn from_str(input: &str) -> Result<AucType, Self::Err> {
        match input.to_uppercase().as_str() {
            "ROC" => Ok(AucType::ROC),
            "PR" => Ok(AucType::PR),
            _ => Err(CalculationError::InvalidAucType),
        }
    }
}