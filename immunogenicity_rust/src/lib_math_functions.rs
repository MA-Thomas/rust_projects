use std::f64::consts::E;

use crate::*; // Import from the main module (lib.rs)
use crate::lib_data_structures_auxiliary_functions::{CalculationError, AucType};

// pub fn calculate_auc(pos: &[f64], neg: &[f64]) -> Result<f64, CalculationError> {
//     // Check if input slices are empty
//     if pos.is_empty() || neg.is_empty() {
//         return Err(CalculationError::InvalidInput);
//     }

//     // Combine positive and negative scores into a single vector of (label, score) pairs
//     let mut pairs: Vec<(bool, f64)> = pos.iter().map(|&score| (true, score)).collect();
//     pairs.extend(neg.iter().map(|&score| (false, score)));

//     // Compute ROC AUC score
//     if let Some(auc) = classifier_measures::roc_auc_mut(&mut pairs) {
//         Ok(auc)
//     } else {
//         Err(CalculationError::RocCurveError)
//     }
// }

pub fn calculate_auc(pos: &[f64], neg: &[f64], auc_type: &AucType) -> Result<f64, CalculationError> {
    // Check if input slices are empty
    if pos.is_empty() || neg.is_empty() {
        return Err(CalculationError::InvalidInput);
    }

    // Combine positive and negative scores into a single vector of (label, score) pairs
    let mut pairs: Vec<(bool, f64)> = pos.iter().map(|&score| (true, score)).collect();
    pairs.extend(neg.iter().map(|&score| (false, score)));

    match auc_type {
        AucType::ROC => {
            // Compute ROC AUC score using the same function as in the original code
            if let Some(auc) = classifier_measures::roc_auc_mut(&mut pairs) {
                Ok(auc)
            } else {
                Err(CalculationError::RocCurveError)
            }
        }
        AucType::PR => {
            // Compute PR AUC score
            if let Some(auc) = classifier_measures::pr_auc_mut(&mut pairs) {
                Ok(auc)
            } else {
                Err(CalculationError::PrCurveError)
            }
        }
    }
}

impl From<CalculationError> for pyo3::PyErr {
    fn from(error: CalculationError) -> Self {
        // Convert CalculationError to PyErr here
        // Example:
        pyo3::exceptions::PyValueError::new_err(format!("CalculationError: {:?}", error))
    }
}

// Define a custom error type for NaN or infinity values
#[derive(Debug)]
pub struct InputError(&'static str);

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


#[derive(Debug)]
pub enum EntropyError {
    NaN,
    Infinite,
    InvalidDistribution,
}
pub fn compute_entropy(v: &[f64], log_z: f64) -> Result<f64, EntropyError> {
    // Calculate the probabilities by exponentiating each value in the vector
    let probs: Vec<f64> = v.iter().map(|val| E.powf(val - log_z)).collect();

    // Check if any probability is less than or equal to 0
    if probs.iter().any(|&prob| prob <= 0.0) {
        return Err(EntropyError::InvalidDistribution);
    }

    // Calculate the entropy
    let entropy = probs.iter().map(|&prob| -prob * prob.ln()).sum::<f64>();

    // Check if entropy is NaN or infinite
    if entropy.is_nan() {
        Err(EntropyError::NaN)
    } else if entropy.is_infinite() {
        Err(EntropyError::Infinite)
    } else {
        Ok(entropy)
    }
}
