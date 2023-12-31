use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::{collections::HashMap, marker::PhantomData};

use ndarray::{Array1, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use num_traits::real::Real;
use num_traits::Float;
use rand_isaac::Isaac64Rng;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::polyfit::{polyfit, FitResult};
use crate::Result;

/// Configuratin for a device
pub struct Config<E> {
    /// Degree of polynomial fit
    ///
    /// The degree is the highest power used in the fit. Currently we use the same degree of
    /// polynomial fit for ALL fits for all sensors.
    pub polynomial_fit_degree: usize,
    /// The operating frequency (or sampling rate) of the sensor in Hz
    pub operating_frequency: E,
    /// The number of samples to generate when fitting.
    ///
    /// To estimate the error on the polynomial coefficients the fit must be carried out multiple
    /// times. This is because the predominant error is on the x-axis variable, which the
    /// polynomial fitting algorithm assumes to be error free.
    ///
    /// To overcome this we carry out `number_of_polyfit_samples` for each fit, sampling the `x`
    /// valuse from a normal distribution. The fit coefficients and variances are then constructed
    /// from the generated distribution.
    pub number_of_polyfit_samples: usize,
}

/// Build a system of sensors from a file path and configuration file.
///
/// The `working_directory` is assumed to contain one sub-folder for each sensor in the device,
/// with name equal to the gas the sensor detects.
///
/// Each sub-folder is expected to contain at minimum a `.toml` configuration file describing the sensor noise
/// characteristic, and a `.csv` file containing the raw calibration data. In a multigas
/// sensing problem each sub-folder will contain additional `.csv` files with the crosstalk data.
///
/// The calibration data is expected to be in a 5-column csv file with format [`CalibrationCsvRow`]
/// and the crosstalk data in an 9-column csv file with format [`CrosstalkCsvRow`]. The
/// `sensor.toml` must contain all fields in [`SensorData`].
///
/// Note that at present it is assumed elsewhere that in a multigas system there is a single sensor
/// for each absorbing species, and that crosstalk data is present at every sensor for every gas.
/// This function TODO does not currently verify this is the case, but if it is not undefined
/// behaviour will occur elsewhere.
pub fn build<E>(working_directory: &Path, config: &Config<E>) -> Result<Vec<Sensor<E>>>
where
    E: Float + Lapack + Scalar + ScalarOperand + Real,
    StandardNormal: Distribution<E>,
{
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
/// Minimal data to describe a photodetector
pub struct SensorData<E> {
    /// Noise equivalent power in nano-Watts
    pub noise_equivalent_power: E,
}

/// Process a directory `path` corresponding to a single [`Sensor`]
///
/// This function builds a single sensor from `.toml` and `.csv` files present at the root level in
/// `path`. The `path` is expected to contain a `.toml` configuration file describing the sensor noise
/// characteristic, and a `.csv` file containing the raw calibration data. In a multigas
/// sensing problem it should also contain `.csv` files with the crosstalk data.
///
/// The calibration data is expected to be in a 5-column csv file with format [`CalibrationCsvRow`]
/// and the crosstalk data in an 9-column csv file with format [`CrosstalkCsvRow`]. The
/// `sensor.toml` must contain all fields in [`SensorData`].
fn process<E: Scalar>(path: &PathBuf, config: &Config<E>) -> Result<SensorBuilder<E, Set>> {
    let target = Gas(path.file_stem().unwrap().to_string_lossy().into_owned()); // The target gas is the directory name
    println!("Working on target {target:?}");
    let sensor_data_file = path.join("sensor.toml");
    let sensor_data = fs::read_to_string(&sensor_data_file)?;
    let sensor_data: SensorData<E> = toml::from_str(&sensor_data)?;
    println!("Successfully read sensor file");

    // Paths to crosstalk
    let crosstalk_file_paths = fs::read_dir(path)?
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
        config.number_of_polyfit_samples,
    );

    for csv_file_path in crosstalk_file_paths {
        println!("reading crosstalk from {csv_file_path:?}");
        let crosstalk_data = CrosstalkData::from_file(&csv_file_path, target.clone())?;
        builder = builder.with_crosstalk(crosstalk_data);
    }

    println!("reading calibration curve for {}", target.0);
    let path_to_calibration_csv = path.join(format!("{}.csv", target.0));
    let calibration_data = CalibrationData::from_file(&path_to_calibration_csv)?;
    let builder = builder.with_calibration(calibration_data);

    Ok(builder)
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Gas(pub String);

/// Representation of one sensor
pub struct Sensor<E: Scalar> {
    /// The molecule the sensor is designed to detect
    target: Gas,
    /// Frequency of operation
    operation_frequency: E,
    /// The noise equivalent power of the photodetector in nano-Watt
    noise_equivalent_power: E,
    /// Calibration Curve
    calibration: FitResult<E>,
    /// Crosstalk curves
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

pub(crate) enum Set {}
pub(crate) enum Unset {}

pub(crate) struct SensorBuilder<E: Scalar, N> {
    target: Gas,
    operation_frequency: E,
    noise_equivalent_power: E,
    number_of_polyfit_samples: usize,
    polynomial_degree: usize,
    raw_calibration_data: Option<CalibrationData<E>>,
    raw_crosstalk_data: Vec<CrosstalkData<E>>,
    phantom_data: PhantomData<N>,
}

impl<E: Scalar, N> SensorBuilder<E, N> {
    pub(crate) fn new(
        target: Gas,
        noise_equivalent_power: E,
        operation_frequency: E,
        polynomial_degree: usize,
        number_of_polyfit_samples: usize,
    ) -> Self {
        Self {
            target,
            noise_equivalent_power,
            operation_frequency,
            number_of_polyfit_samples,
            polynomial_degree,
            raw_calibration_data: None,
            raw_crosstalk_data: vec![],
            phantom_data: PhantomData,
        }
    }

    pub(crate) fn with_crosstalk(mut self, calibration_data: CrosstalkData<E>) -> Self {
        self.raw_crosstalk_data.push(calibration_data);
        self
    }
}

impl<E: Scalar> SensorBuilder<E, Unset> {
    pub(crate) fn with_calibration(
        mut self,
        calibration_data: CalibrationData<E>,
    ) -> SensorBuilder<E, Set> {
        self.raw_calibration_data.replace(calibration_data);
        SensorBuilder {
            target: self.target,
            noise_equivalent_power: self.noise_equivalent_power,
            operation_frequency: self.operation_frequency,
            polynomial_degree: self.polynomial_degree,
            number_of_polyfit_samples: self.number_of_polyfit_samples,
            raw_calibration_data: self.raw_calibration_data,
            raw_crosstalk_data: self.raw_crosstalk_data,
            phantom_data: PhantomData,
        }
    }
}

impl<E> SensorBuilder<E, Set>
where
    E: Float + Lapack + Real + Scalar + ScalarOperand,
    StandardNormal: Distribution<E>,
{
    pub(crate) fn build(self) -> Result<Sensor<E>> {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let mut crosstalk = HashMap::new();

        let calibration_data = self.raw_calibration_data.unwrap(); // Safe as we cannot built without
                                                                   // this being `Set`. If not set we cannot constuct the type
        let calibration = generate_fit(
            &calibration_data,
            self.noise_equivalent_power,
            self.operation_frequency,
            self.number_of_polyfit_samples,
            self.polynomial_degree,
            &mut rng,
        )?;

        for crosstalk_data in self.raw_crosstalk_data {
            let fit = generate_crosstalk_fit(
                &crosstalk_data,
                self.noise_equivalent_power,
                self.operation_frequency,
                self.number_of_polyfit_samples,
                self.polynomial_degree,
                &mut rng,
            )?;

            crosstalk.insert(crosstalk_data.target_gas.clone(), fit);
        }

        Ok(Sensor {
            target: self.target,
            noise_equivalent_power: self.noise_equivalent_power,
            operation_frequency: self.operation_frequency,
            calibration,
            crosstalk,
        })
    }
}

/// Generate a calibration curve from [`CalibrationData`]
///
/// In calibration data the predominant source of error is on the x-axis variables, which are the
/// signal recorded from the detector. As the underlying polynomial regression algorithm cannot
/// account for this `generate_fit` runs `number_of_samples` calculations of the polynomial
/// coefficients.
///
/// For each sample new x-data is generated from the known distributions of the x-axis variables.
/// Finally the coefficients of the polynomial are calculated from the mean of the result, along
/// with their variance.
fn generate_fit<E>(
    calibration_data: &CalibrationData<E>,
    noise_equivalent_power: E,
    operating_frequency: E,
    number_of_samples: usize,
    degree: usize,
    rng: &mut impl Rng,
) -> Result<FitResult<E>>
where
    E: Float + Lapack + Real + Scalar + ScalarOperand,
    StandardNormal: Distribution<E>,
{
    let mut fits = Vec::new();
    for _ in 0..number_of_samples {
        let data = calibration_data.generate_fitting_data(
            noise_equivalent_power,
            operating_frequency,
            rng,
        )?;
        let fit = polyfit(
            &data.x,
            &data.y,
            degree,
            data.w.as_ref().map(|x| &x[..]),
        )?;
        fits.push(fit);
    }

    let means = fits
        .iter()
        .map(crate::polyfit::FitResult::solution)
        .fold(Array1::zeros(degree + 1), |a, b| a + b)
        .mapv(|summed: E| summed / E::from(fits.len()).unwrap());
    let variance = fits
        .iter()
        .map(crate::polyfit::FitResult::solution)
        .fold(Array1::zeros(degree + 1), |a, b| {
            a + (b - &means).mapv(|x| Scalar::powi(x, 2))
        })
        .mapv(|summed: E| summed / E::from(fits.len() - 1).unwrap());

    let mut fit = fits.pop().unwrap();
    fit.set_solution(means);
    fit.set_variance(variance);

    Ok(fit)
}

/// Generate a crosstalk curve from [`CrosstalkData`]
///
/// In crosstalk data the error on x-axis and y-axis variables are the same order of magnitude:
/// they both arise from measurement error of the signal.
///
/// Currently or each sample new x-data is generated from the known distributions of the x-axis variables.
/// Finally the coefficients of the polynomial are calculated from the mean of the result, along
/// with their variance.
///
/// TODO: We might not have to do this, maybe we can just take the single-pass results and get an
/// estimate. Alternatively we might need to combine the two error sources. I do not have time to
/// do this. Currently errors from `y` are being thrown away.
fn generate_crosstalk_fit<E>(
    crosstalk_data: &CrosstalkData<E>,
    noise_equivalent_power: E,
    operating_frequency: E,
    number_of_samples: usize,
    degree: usize,
    rng: &mut impl Rng,
) -> Result<FitResult<E>>
where
    E: Float + Lapack + Real + Scalar + ScalarOperand,
    StandardNormal: Distribution<E>,
{
    let mut fits = Vec::new();
    for _ in 0..number_of_samples {
        let data = crosstalk_data.generate_fitting_data(
            noise_equivalent_power,
            operating_frequency,
            rng,
        )?;
        let fit = polyfit(
            &data.x,
            &data.y,
            degree,
            data.w.as_ref().map(|x| &x[..]),
        )?;
        fits.push(fit);
    }

    let means = fits
        .iter()
        .map(crate::polyfit::FitResult::solution)
        .fold(Array1::zeros(degree + 1), |a, b| a + b)
        .mapv(|summed: E| summed / E::from(fits.len()).unwrap());

    let variance = fits
        .iter()
        .map(crate::polyfit::FitResult::solution)
        .fold(Array1::zeros(degree + 1), |a, b| {
            a + (b - &means).mapv(|x| Scalar::powi(x, 2))
        })
        .mapv(|summed: E| summed / E::from(fits.len() - 1).unwrap());

    let mut fit = fits.pop().unwrap();
    fit.set_solution(means);
    fit.set_variance(variance);

    Ok(fit)
}

/// Calibration data for target gas in a sensor
pub(crate) struct CalibrationData<E: Scalar> {
    /// The gas the sensor is designed to measure
    pub(crate) gas: Gas,
    /// The known concentration of gas
    pub(crate) concentration: Vec<E>,
    /// The raw measurements corresponding to `concentration`
    pub(crate) raw_measurements: Vec<Measurement<E>>,
}

pub(crate) struct CrosstalkData<E: Scalar> {
    /// The target gas is the one providing the x-data. So for the crosstalk resulting from the
    /// presence of gas x in channel y `target_gas` corresponds to the concentration of x
    pub(crate) target_gas: Gas,
    /// The other gas is the one where the crosstalk is being observed.
    pub(crate) crosstalk_gas: Gas,
    /// x-values: the signal we would see if only `target_gas` existed
    pub(crate) signal: Vec<Measurement<E>>,
    /// y-values: the signal observed in the `crosstalk_gas` channel
    pub(crate) crosstalk: Vec<Measurement<E>>,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Deserialize, Serialize)]
/// A single row in the expected `.csv` file for calibration data
pub struct CalibrationCsvRow<E> {
    pub concentration: E,
    pub raw_signal: E,
    pub raw_reference: E,
    pub emergent_signal: E,
    pub emergent_reference: E,
}

#[derive(Deserialize, Serialize)]
/// A single row in the expected `.csv` file for crosstalk data
pub struct CrosstalkCsvRow<E> {
    pub raw_signal_target: E,
    pub raw_reference_target: E,
    pub emergent_signal_target: E,
    pub emergent_reference_target: E,
    pub raw_signal_crosstalk: E,
    pub raw_reference_crosstalk: E,
    pub emergent_signal_crosstalk: E,
    pub emergent_reference_crosstalk: E,
}

impl<E: Scalar + Copy + DeserializeOwned> CalibrationData<E> {
    /// Create a [`CalibrationData`] from an on-disk representation
    ///
    /// Reads a `.csv` located at `filepath` into a [`CalibrationData`]. It is assumed that the
    /// `.csv` has one header row. The format of the `.csv` file must contain 5-columns, matching
    /// [`CalibrationCsvRow`]
    ///
    /// # Panics
    /// - If the `filepath` does not have a `file_stem()`, or that stem cannot be converted to a
    /// non-null `str`
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
            let record: CalibrationCsvRow<E> = result?;
            concentration.push(record.concentration);
            let measurement = Measurement {
                raw_signal: record.raw_signal,
                raw_reference: record.raw_reference,
                emergent_signal: record.emergent_signal,
                emergent_reference: record.emergent_reference,
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

impl<E: Scalar + Copy + DeserializeOwned> CrosstalkData<E> {
    /// Create a [`CrosstalkData`] from an on-disk representation
    ///
    /// Reads a `.csv` located at `filepath` into a [`CrosstalkData`]. It is assumed that the
    /// `.csv` has one header row. The format of the `.csv` file must contain 8-columns, matching
    /// [`CrosstalkCsvRow`]
    ///
    /// # Panics
    /// - If the `filepath` does not have a `file_stem()`, or that stem cannot be converted to a
    /// non-null `str`
    fn from_file(filepath: &PathBuf, crosstalk_gas: Gas) -> Result<Self> {
        if !filepath.exists() {
            return Err("requested file not found".into());
        }

        // The gas defining the x-coordinate of the signal
        let target_gas = filepath
            .file_stem()
            .expect("filestem missing")
            .to_str()
            .expect("failed to convert stem to string")
            .to_owned();

        let file = fs::read(filepath)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(&file[..]);

        let mut signal = vec![];
        let mut crosstalk = vec![];

        for result in rdr.deserialize() {
            let record: CrosstalkCsvRow<E> = result?;
            let s = Measurement {
                raw_signal: record.raw_signal_target,
                raw_reference: record.raw_reference_target,
                emergent_signal: record.emergent_signal_target,
                emergent_reference: record.emergent_reference_target,
            };

            let c = Measurement {
                raw_signal: record.raw_signal_crosstalk,
                raw_reference: record.raw_reference_crosstalk,
                emergent_signal: record.emergent_signal_crosstalk,
                emergent_reference: record.emergent_reference_crosstalk,
            };
            signal.push(s);
            crosstalk.push(c);
        }

        Ok(Self {
            target_gas: Gas(target_gas),
            crosstalk_gas,
            signal,
            crosstalk,
        })
    }
}

/// A raw measurement
pub struct Measurement<E: Scalar> {
    /// Signal recorded in the channel
    pub(crate) raw_signal: E,
    /// Signal recorded in the reference
    pub(crate) raw_reference: E,
    /// Signal emitted in the channel
    pub(crate) emergent_signal: E,
    /// Signal emitted in the reference
    pub(crate) emergent_reference: E,
}

impl<E: Real + Scalar> Measurement<E> {
    /// Returns the scaled signal
    ///
    /// To resolve large variations in concentration using a polynomial fit we form a composite
    /// figure of merit, which is the natural logarithm of the ratio of recorded to emitted signals in the channel, divided
    /// by that in the reference.
    pub(crate) fn scaled(&self) -> E {
        Scalar::ln(
            self.raw_signal / self.emergent_signal * self.emergent_reference / self.emergent_signal,
        )
    }

    /// The weights are the inverse of the variance of the measurement
    fn weight(&self, noise_equivalent_power: E, operation_frequency: E) -> E {
        let standard_deviation = noise_equivalent_power
            * Scalar::sqrt(operation_frequency)
            * (self.raw_signal + self.raw_reference)
            / Scalar::powi(self.raw_signal, 2);

        E::one() / Scalar::powi(standard_deviation, 2)
    }
}

impl<E> Measurement<E>
where
    E: Scalar + Float,
    StandardNormal: Distribution<E>,
{
    /// Sample from the known distribution for a measurement
    ///
    /// Above we compute the central values and weights (inverse variance) for a measurement. For a
    /// given `operation_frequency` and `noise_equivalent_power` this method takes these
    /// distributional quantities, forms a normal distribution and returns a single sample.
    fn sample(
        &self,
        rng: &mut impl Rng,
        noise_equivalent_power: E,
        operation_frequency: E,
    ) -> Result<E> {
        let mean = self.scaled();
        let weight = self.weight(noise_equivalent_power, operation_frequency);
        let std_dev = Scalar::sqrt(E::one() / weight);
        let dist = Normal::new(mean, std_dev)?;
        Ok(dist.sample(rng))
    }
}

#[derive(Debug)]
struct PolyFitInput<E> {
    x: Vec<E>,
    y: Vec<E>,
    w: Option<Vec<E>>,
}

impl<E> CalibrationData<E>
where
    E: Float + Scalar,
    StandardNormal: Distribution<E>,
{
    fn generate_fitting_data(
        &self,
        noise_equivalent_power: E,
        operation_frequency: E,
        rng: &mut impl Rng,
    ) -> Result<PolyFitInput<E>> {
        Ok(PolyFitInput {
            x: self
                .raw_measurements
                .iter()
                .map(|raw| raw.sample(rng, noise_equivalent_power, operation_frequency))
                .collect::<Result<_>>()?,
            y: self.concentration.clone(),
            w: None,
        })
    }
}

impl<E> CrosstalkData<E>
where
    E: Float + Scalar,
    StandardNormal: Distribution<E>,
{
    fn generate_fitting_data(
        &self,
        noise_equivalent_power: E,
        operation_frequency: E,
        rng: &mut impl Rng,
    ) -> Result<PolyFitInput<E>> {
        Ok(PolyFitInput {
            x: self
                .signal
                .iter()
                .map(|raw| raw.sample(rng, noise_equivalent_power, operation_frequency))
                .collect::<Result<_>>()?,
            y: self
                .crosstalk
                .iter()
                .map(|raw| raw.sample(rng, noise_equivalent_power, operation_frequency))
                .collect::<Result<_>>()?,
            w: Some(
                self.crosstalk
                    .iter()
                    .map(|raw| raw.weight(noise_equivalent_power, operation_frequency))
                    .collect(),
            ),
        })
    }
}
