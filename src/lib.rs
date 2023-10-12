//! Error estimation and correction for single and multigas sensing systems
//!
//! This library provides the following functionality
//! - Generates polynomial calibration models for single gas sensors.
//! - Given a calibration dataset predicts gas concentration with an associated margin of error
//! - Generates polynomial calibration models for multi-gas sensors with crosstalk.
//! - Given a calibration dataset can correct for errors in multigas sensing, predicting the true
//! concentration with associated margin of error.
//!


#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

/// Methods to create calibration curves from raw data
pub mod calibration;
/// Probability distributions, used to calculate the variance and expectation value of the
/// observable
pub(crate) mod distributions;
/// Computation of the margin of error in the concentration
pub(crate) mod margin;
/// Helper methods for commonly used mathematical operations
pub mod math;
/// Low-level numerical methods for error correctin in a multigas system
pub(crate) mod minimisation;
/// Reconstruction for multi-gas sensors
pub(crate) mod multi;
/// Fitting methods for polynomial regression problems
pub(crate) mod polyfit;
/// Methods for reconstruction of the concentration from measurements.
pub(crate) mod single;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;
