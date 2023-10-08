use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::{collections::HashMap, marker::PhantomData};

use ndarray::ScalarOperand;
use ndarray_linalg::{Lapack, Scalar};
use num_traits::real::Real;
use num_traits::Float;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::polyfit::{polyfit, FitResult};
use crate::Result;

pub struct Config<E> {
    pub polynomial_fit_degree: usize,
    pub operating_frequency: E,
}

/// Build a system of sensors from a file path and config
///
/// # Errors
/// Returns an error if the file system has an incorrect structure, files have incorrect structure
/// or if the polynomial fit fails.
pub fn build<E: Float + Lapack + Scalar + ScalarOperand + Real>(
    working_directory: &Path,
    config: &Config<E>,
) -> Result<Vec<Sensor<E>>> {
    let mut sensors = vec![];
    for subdir in fs::read_dir(working_directory.join("calibration"))? {
        println!("working in {subdir:?}");
        let subdir = subdir?;
        let path = subdir.path();
        let sensor_calibration_data: SensorBuilder<E, Set> = process(&path, config)?;
        sensors.push(sensor_calibration_data);
    }

    // Build the sensors
    let sensors = sensors
        .into_iter()
        .map(SensorBuilder::build)
        .collect::<Result<Vec<Sensor<E>>>>()?;

    Ok(sensors)
}

#[derive(Deserialize, Serialize)]
pub struct SensorData<E> {
    pub noise_equivalent_power: E,
}

fn process<E: Scalar>(path: &PathBuf, config: &Config<E>) -> Result<SensorBuilder<E, Set>> {
    let target = Gas(path.file_stem().unwrap().to_string_lossy().into_owned()); // The target gas is the directory name
    println!("Working on target {target:?}");
    let sensor_data_file = path.join("sensor.toml");
    let sensor_data = fs::read_to_string(&sensor_data_file)?;
    let sensor_data: SensorData<E> = toml::from_str(&sensor_data)?;
    println!("Successfully read sensor file");

    // Paths to crosstalk
    let csv_file_paths = fs::read_dir(path)?
        .filter(::std::result::Result::is_ok)
        .map(::std::result::Result::unwrap)
        // Map the directory entries to paths
        .map(|dir_entry| dir_entry.path())
        // Filter out all paths with extensions other than `csv`
        .filter_map(|path| {
            if path.extension().map_or(false, |ext| ext == "csv") {
                Some(path)
            } else {
                None
            }
        })
        .filter(|path| {
            path.file_stem()
                .and_then(OsStr::to_str)
                .map_or(false, |stem| stem != target.0)
        });

    let mut builder: SensorBuilder<E, Unset> = SensorBuilder::new(
        target.clone(),
        sensor_data.noise_equivalent_power,
        config.operating_frequency,
        config.polynomial_fit_degree,
    );

    for csv_file_path in csv_file_paths {
        println!("reading {csv_file_path:?}");
        let crosstalk_data = CalibrationData::from_file(&csv_file_path)?;
        builder = builder.with_crosstalk(crosstalk_data);
    }

    let path_to_calibration_csv = path.join(format!("{}.csv", target.0));
    let calibration_data = CalibrationData::from_file(&path_to_calibration_csv)?;
    let builder = builder.with_calibration(calibration_data);

    Ok(builder)
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Gas(pub String);

pub struct Sensor<E: Scalar> {
    /// The molecule the sensor is designed to detect
    target: Gas,
    /// Frequency of operation
    operation_frequency: E,
    /// The noise equivalent power of the photodetector in nano-Watt
    noise_equivalent_power: E,
    /// Calibration Curve
    calibration: FitResult<E>,
    /// Crosstalk curves, may or may not be provided
    crosstalk: HashMap<Gas, FitResult<E>>,
}

impl<E: Scalar + std::fmt::Debug> Sensor<E> {
    pub const fn calibration(&self) -> &FitResult<E> {
        &self.calibration
    }

    pub const fn target(&self) -> &Gas {
        &self.target
    }

    pub const fn crosstalk(&self) -> &HashMap<Gas, FitResult<E>> {
        &self.crosstalk
    }
}

enum Set {}
enum Unset {}

struct SensorBuilder<E: Scalar, N> {
    target: Gas,
    operation_frequency: E,
    noise_equivalent_power: E,
    polynomial_degree: usize,
    raw_calibration_data: Vec<CalibrationData<E>>,
    phantom_data: PhantomData<N>,
}

impl<E: Scalar, N> SensorBuilder<E, N> {
    fn new(
        target: Gas,
        noise_equivalent_power: E,
        operation_frequency: E,
        polynomial_degree: usize,
    ) -> Self {
        Self {
            target,
            noise_equivalent_power,
            operation_frequency,
            polynomial_degree,
            raw_calibration_data: vec![],
            phantom_data: PhantomData,
        }
    }

    fn with_crosstalk(mut self, calibration_data: CalibrationData<E>) -> Self {
        self.raw_calibration_data.push(calibration_data);
        self
    }
}

impl<E: Scalar> SensorBuilder<E, Unset> {
    fn with_calibration(mut self, calibration_data: CalibrationData<E>) -> SensorBuilder<E, Set> {
        self.raw_calibration_data.push(calibration_data);
        SensorBuilder {
            target: self.target,
            noise_equivalent_power: self.noise_equivalent_power,
            operation_frequency: self.operation_frequency,
            polynomial_degree: self.polynomial_degree,
            raw_calibration_data: self.raw_calibration_data,
            phantom_data: PhantomData,
        }
    }
}

impl<E: Float + Lapack + Real + Scalar + ScalarOperand> SensorBuilder<E, Set> {
    fn build(self) -> Result<Sensor<E>> {
        let mut crosstalk = HashMap::new();
        let mut calibration = None;
        for calibration_data in self.raw_calibration_data {
            let fit = generate_fit(
                &calibration_data,
                self.noise_equivalent_power,
                self.operation_frequency,
                self.polynomial_degree,
            )?;

            if calibration_data.gas == self.target {
                calibration = Some(fit);
            } else {
                crosstalk.insert(calibration_data.gas.clone(), fit);
            }
        }

        Ok(Sensor {
            target: self.target,
            noise_equivalent_power: self.noise_equivalent_power,
            operation_frequency: self.operation_frequency,
            calibration: calibration.unwrap(),
            crosstalk,
        })
    }
}

fn generate_fit<E: Float + Lapack + Real + Scalar + ScalarOperand>(
    calibration_data: &CalibrationData<E>,
    noise_equivalent_power: E,
    operating_frequency: E,
    degree: usize,
) -> Result<FitResult<E>> {
    let data = calibration_data.generate_fitting_data(noise_equivalent_power, operating_frequency);
    let fit = polyfit(
        &data.x,
        &data.y,
        degree,
        Some(&data.w),
        crate::polyfit::Scaling::Unscaled,
    )?;
    Ok(fit)
}

struct CalibrationData<E: Scalar> {
    gas: Gas,
    concentration: Vec<E>,
    raw_measurements: Vec<Measurement<E>>,
}

#[derive(Deserialize)]
struct Row<E>(E, E, E, E, E);

impl<E: Scalar + Copy + DeserializeOwned> CalibrationData<E> {
    /// Create a `CalibrationData` from an on-disk representation
    fn from_file(filepath: &PathBuf) -> Result<Self> {
        if !filepath.exists() {
            return Err("requested file not found".into());
        }

        let gas = filepath
            .file_stem()
            .expect("filestem missing")
            .to_str()
            .expect("failed to convert stem to string")
            .to_owned();

        let file = fs::read(filepath)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(&file[..]);

        let mut concentration = vec![];
        let mut raw_measurements = vec![];

        for result in rdr.deserialize() {
            let record: Row<E> = result?;
            concentration.push(record.0);
            let measurement = Measurement {
                raw_signal: record.1,
                raw_reference: record.2,
                emergent_signal: record.3,
                emergent_reference: record.4,
            };
            raw_measurements.push(measurement);
        }

        Ok(Self {
            gas: Gas(gas),
            concentration,
            raw_measurements,
        })
    }
}

pub struct Measurement<E: Scalar> {
    raw_signal: E,
    raw_reference: E,
    emergent_signal: E,
    emergent_reference: E,
}

impl<E: Real + Scalar> Measurement<E> {
    fn scaled(&self) -> E {
        (self.raw_signal / self.emergent_signal * self.emergent_reference / self.emergent_signal)
            .log10()
    }

    // The weights are the inverse of the variance of the measurement
    fn weight(&self, noise_equivalent_power: E, operation_frequency: E) -> E {
        let standard_deviation = noise_equivalent_power
            * Scalar::sqrt(operation_frequency)
            * (self.raw_reference + self.raw_reference)
            / Scalar::powi(self.raw_signal, 2);

        E::one() / Scalar::powi(standard_deviation, 2)
    }
}

struct PolyFitInput<E> {
    x: Vec<E>,
    y: Vec<E>,
    w: Vec<E>,
}

impl<E: Real + Scalar> CalibrationData<E> {
    fn generate_fitting_data(
        &self,
        noise_equivalent_power: E,
        operation_frequency: E,
    ) -> PolyFitInput<E> {
        PolyFitInput {
            x: self.concentration.clone(),
            y: self
                .raw_measurements
                .iter()
                .map(Measurement::scaled)
                .collect(),
            w: self
                .raw_measurements
                .iter()
                .map(|measurement| measurement.weight(noise_equivalent_power, operation_frequency))
                .collect(),
        }
    }
}
