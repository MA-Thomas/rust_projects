use std::collections::{HashMap, HashSet};
use std::fs::{File, remove_file, create_dir_all};
use std::io::BufReader;
use std::error::Error;

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

pub fn tuple_to_string(tuple: &(f64, f64)) -> String {
    format!("{:?}", tuple)
}

// Helper function to check if two HashMaps have the same keys
pub fn have_same_keys(map1: &HashMap<String, f64>, map2: &HashMap<String, f64>) -> bool {
    let keys1: HashSet<_> = map1.keys().collect();
    let keys2: HashSet<_> = map2.keys().collect();
    keys1 == keys2
}