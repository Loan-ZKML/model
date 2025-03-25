use anyhow::Result;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write;

fn main() -> Result<()> {
    println!("Generate synthetic data");

    fs::create_dir_all("proof_generation")?;

    println!("Generating synthetic data with test addresses...");
    let data = generate_synthetic_data(1000)?;
    save_data_as_json(&data, "proof_generation/credit_data.json")?;

    Ok(())
}

pub fn generate_synthetic_data(num_samples: usize) -> Result<CreditData> {
    let mut rng = thread_rng();

    // Define distributions with their normalizers in a single array for better organization
    let distributions = [
        (Uniform::new(0.0f32, 5000.0f32), 5000.0f32),
        (Uniform::new(0.0f32, 365.0f32 * 5.0f32), 365.0f32 * 5.0f32),
        (Uniform::new(0.0f32, 10000.0f32), 10000.0f32),
        (Uniform::new(0.0f32, 1.0f32), 1.0f32),
    ];

    // Generate features using iterators in a more compact way
    let features: Vec<Vec<f32>> = (0..num_samples)
        .map(|_| {
            distributions
                .iter()
                .map(|(dist, normalizer)| {
                    let raw_value = dist.sample(&mut rng);
                    // For the last feature (repayment history), round to 0 or 1
                    if normalizer == &1.0f32 {
                        // Just check the normalizer for repayment history
                        f32::round(raw_value)
                    } else {
                        raw_value / normalizer
                    }
                })
                .collect()
        })
        .collect();

    // Generate synthetic credit scores as a function of features with some noise
    let noise_dist = Normal::new(0.0f32, 0.05f32)?;

    // Calculate scores using the new pure functions
    let scores: Vec<f32> = features
        .iter()
        .map(|feature| {
            let noise = noise_dist.sample(&mut rng);
            calculate_score(feature, noise)
        })
        .collect();

    let feature_names = vec![
        "tx_count".to_string(),
        "wallet_age".to_string(),
        "avg_balance".to_string(),
        "repayment_history".to_string(),
    ];

    Ok(CreditData {
        features,
        scores,
        feature_names,
        address_mapping: None,
    })
}

/// Calculate a credit score based on feature values using a composition of pure functions
fn calculate_score(features: &[f32], noise: f32) -> f32 {
    // Use a pipeline of pure functions with Option type for proper error handling
    let result = validate_features(features)
        .and_then(|valid_features| calculate_weighted_score(&valid_features))
        .and_then(|score| process_noise(noise).map(|n| score + n))
        .and_then(|combined| sigmoid(10.0f32 * combined - 5.0f32));

    // Default score for invalid inputs
    result.unwrap_or(0.5f32)
}

/// Sigmoid activation function that maps any real number to a value between 0 and 1
///
/// Transforms inputs using the formula: f(x) = 1 / (1 + e^(-x))
/// The function has a natural S-curve shape and smoothly maps all real inputs to [0, 1]
/// Handles extreme values to prevent floating-point issues.
fn sigmoid(x: f32) -> Option<f32> {
    // Handle extreme values to avoid floating-point issues
    // Using a smaller threshold (10.0) to ensure more precise bounds for extreme values
    const EPSILON: f32 = 1e-6f32;

    if !x.is_finite() {
        // Handle NaN and infinity
        None
    } else {
        let result = if x > 10.0f32 {
            // For large positive inputs, return a value very close to 1.0 but not exactly 1.0
            1.0f32 - EPSILON
        } else if x < -10.0f32 {
            // For large negative inputs, return a value very close to 0.0 but not exactly 0.0
            EPSILON
        } else {
            // Normal calculation for reasonable range inputs
            1.0f32 / (1.0f32 + (-x).exp())
        };
        Some(result)
    }
}

/// Processes noise to ensure it's within reasonable bounds
fn process_noise(noise: f32) -> Option<f32> {
    if noise.is_finite() {
        Some(noise.clamp(-5.0f32, 5.0f32))
    } else {
        None
    }
}

/// Calculates a weighted score from validated features
/// Returns None if there aren't exactly 4 features
fn calculate_weighted_score(features: &[f32]) -> Option<f32> {
    if features.len() != 4 {
        return None;
    }

    let weights = [0.3f32, 0.2f32, 0.2f32, 0.3f32];
    Some(
        features
            .iter()
            .zip(weights.iter())
            .map(|(&f, &w)| f * w)
            .sum(),
    )
}

/// Validates all features in a slice, ensuring each is valid
/// Returns None if any feature is invalid
fn validate_features(features: &[f32]) -> Option<Vec<f32>> {
    features.iter().map(|&x| validate_feature(x)).collect()
}

pub fn save_data_as_json(data: &CreditData, path: &str) -> Result<()> {
    let json_data = serde_json::to_string_pretty(data)?;
    let mut file = File::create(path)?;
    file.write_all(json_data.as_bytes())?;
    Ok(())
}

/// Validates a single feature value, ensuring it's finite and within the [0,1] range
fn validate_feature(x: f32) -> Option<f32> {
    if x.is_finite() {
        Some(x.clamp(0.0f32, 1.0f32))
    } else {
        None
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreditData {
    pub features: Vec<Vec<f32>>,
    pub scores: Vec<f32>,
    pub feature_names: Vec<String>,
    // Optional mapping from Ethereum addresses to indices in the features/scores arrays
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address_mapping: Option<HashMap<String, usize>>,
}
