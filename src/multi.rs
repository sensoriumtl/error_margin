use std::collections::HashMap;

use ndarray::{Array1, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num_traits::Float;
use rand_isaac::Isaac64Rng;

use crate::calibration::Gas;
use crate::minimisation::Problem;
use crate::polyfit::Polynomial;
use crate::Result;
use crate::{calibration::Sensor, margin::Measurement};

/// Convert a solution vector to a [`HashMap`]
///
/// The coefficients are computed as an ordered `Vec`. Using a list of `measurement_targets` with
/// the same ordering as `solution` this method constructs a [`HashMap`] linking the elements of
/// `measurement_targets` to those of `solution`.
fn into_map<E>(solution: Array1<E>, measurement_targets: &[Gas]) -> HashMap<Gas, E> {
    assert_eq!(solution.len(), measurement_targets.len());
    let mut map = HashMap::new();
    for (target, element) in measurement_targets.iter().zip(solution) {
        map.insert(target.clone(), element);
    }
    map
}

/// Sampling strategy for error correction.
pub enum Strategy {
    /// Just compute the central value, ignoring error
    CentralValue,
    /// Compute `n` samples to get an estimate of the variance.
    SampleDistribution(usize),
}

/// Carry out error correction
///
/// Given `raw_measurements` for `sensors` computes the corrected measurements, eliminating the
/// effects of crosstalk.
///
/// # Panics
/// - If the keys of 'raw_measurements` do not match those of `sensors`
/// - If any sensor is missing full crosstalk data. All sensors must have a crosstalk curve for
/// every gas in `raw_measurements`
pub fn correct<
    E: Lapack
        + PartialOrd
        + Scalar
        + ScalarOperand
        + std::convert::Into<f64>
        + std::convert::From<f64>
        + Float,
>(
    raw_measurements: &HashMap<Gas, Measurement<E>>,
    sensors: &HashMap<Gas, Sensor<E>>,
    initial_parameters: Option<Array1<E>>,
    // whether to compute the central value given the measurements, or to compute the
    // distributional properties
    strategy: &Strategy,
) -> Result<HashMap<Gas, Measurement<E>>>
where
    StandardNormal: Distribution<E>,
{
    let results = match strategy {
        Strategy::CentralValue => {
            compute_central_value(raw_measurements, sensors, initial_parameters)?
        }
        Strategy::SampleDistribution(num_samples) => build_and_evaluate_distribution(
            raw_measurements,
            sensors,
            initial_parameters,
            *num_samples,
        )?,
    };
    Ok(results)
}

fn compute_central_value<
    E: Lapack
        + PartialOrd
        + Scalar
        + ScalarOperand
        + std::convert::Into<f64>
        + std::convert::From<f64>,
>(
    raw_measurements: &HashMap<Gas, Measurement<E>>,
    sensors: &HashMap<Gas, Sensor<E>>,
    initial_parameters: Option<Array1<E>>,
) -> Result<HashMap<Gas, Measurement<E>>> {
    let measurement_targets = raw_measurements.keys().cloned().collect::<Vec<_>>();
    let problem = Problem::build(&measurement_targets, raw_measurements, sensors).into_f64();

    // TODO: casting to f64 because `solve` does not work with generics
    let initial_parameters: Array1<f64> = initial_parameters.map_or_else(
        || Array1::zeros(measurement_targets.len()),
        |initial_parameters| initial_parameters.mapv(std::convert::Into::into),
    );

    let solution: Array1<E> = problem
        .solve(initial_parameters)?
        .mapv(std::convert::Into::into);
    let solution = into_map(solution, &measurement_targets)
        .into_iter()
        .map(|(target, value)| {
            (
                target,
                Measurement {
                    value,
                    uncertainty: E::zero(), // Uncertainty cannot be determined from 1 sample
                },
            )
        })
        .collect::<HashMap<_, _>>();

    Ok(solution)
}

/// Converts a vec of n-dimensional samples to distributional quantities, computing the mean
/// and standard deviation of the n-dimensional dataset
fn form_measurements<E: Scalar>(samples: &[Array1<E>]) -> Array1<Measurement<E>> {
    let means = samples
        .iter()
        .fold(Array1::zeros(samples[0].len()), |a, b| a + b)
        .mapv(|summed: E| summed / E::from(samples.len()).unwrap());

    let variance = samples
        .iter()
        .fold(Array1::zeros(samples[0].len()), |a, b| {
            a + (b - &means).mapv(|x| x.powi(2))
        })
        .mapv(|summed: E| summed / E::from(samples.len() - 1).unwrap());

    means
        .into_iter()
        .zip(variance)
        .map(|(mean, variance)| Measurement {
            value: mean,
            uncertainty: variance.sqrt(),
        })
        .collect()
}

fn build_and_evaluate_distribution<
    E: Lapack
        + PartialOrd
        + Scalar
        + ScalarOperand
        + std::convert::Into<f64>
        + std::convert::From<f64>
        + Float,
>(
    raw_measurements: &HashMap<Gas, Measurement<E>>,
    sensors: &HashMap<Gas, Sensor<E>>,
    initial_parameters: Option<Array1<E>>,
    num_samples: usize,
) -> Result<HashMap<Gas, Measurement<E>>>
where
    StandardNormal: Distribution<E>,
{
    let state = 40;
    let mut rng = Isaac64Rng::seed_from_u64(state);

    let measurement_targets = raw_measurements.keys().cloned().collect::<Vec<_>>();

    // TODO: casting to f64 because `solve` does not work with generics
    let initial_parameters: Array1<f64> = initial_parameters.map_or_else(
        || Array1::zeros(measurement_targets.len()),
        |initial_parameters| initial_parameters.mapv(std::convert::Into::into),
    );

    let mut solutions = vec![];
    for _ in 0..num_samples {
        let problem = Problem::build_with_sampling(
            &measurement_targets,
            raw_measurements,
            sensors,
            &mut rng,
        )?
        .into_f64();

        // If solve fails for some parameter we just skip
        // TODO: a better strategy to avoid missing problematic parameter sets
        match problem.solve(initial_parameters.clone()) {
            Ok(arr) => {
                let solution = arr.mapv(std::convert::Into::into);
                solutions.push(solution);
            }
            Err(e) => eprintln!("Error in solve step {e:?}"),
        }
    }

    let solution = form_measurements(&solutions);

    let solution = into_map(solution, &measurement_targets);

    Ok(solution)
}

pub fn reconstruct<E: Lapack + PartialOrd + Scalar + ScalarOperand>(
    true_measurement: &Measurement<E>, // Measurement after error correction with corresponding
    // uncertainty
    sensor: &Sensor<E>,
) -> Measurement<E> {
    let calibration_curve = sensor.calibration();
    assert!(
        calibration_curve.window_contains(&true_measurement.value),
        "measurement point {} must be in the range of calibration {:?}",
        true_measurement.value,
        calibration_curve.window()
    );

    let polynomial = Polynomial::from(calibration_curve.clone());

    true_measurement.compute_unknown(&polynomial)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, ops::Range};

    use ndarray::{Array, Array1};
    use ndarray_rand::{
        rand::{Rng, SeedableRng},
        rand_distr::{Alphanumeric, Normal},
        RandomExt,
    };
    use rand_isaac::Isaac64Rng;

    use crate::{
        calibration::{CalibrationData, CrosstalkData, Gas, Measurement, SensorBuilder},
        multi::{correct, Strategy},
        Result,
    };

    use super::form_measurements;

    struct GeneratedPolynomial<const N: usize> {
        x: Vec<f64>,
        y: Vec<f64>,
        coeffs: [f64; N],
    }

    #[allow(clippy::cast_precision_loss)]
    fn generate_polynomial<const N: usize>(
        rng: &mut impl Rng,
        num_samples: usize,
        window: Range<f64>,
    ) -> GeneratedPolynomial<N> {
        let x = (0..num_samples)
            .map(|n| window.start + (window.end - window.start) * n as f64 / num_samples as f64)
            .collect::<Vec<_>>();

        let mut coeffs = [0f64; N];
        for coeff in &mut coeffs {
            *coeff = rng.gen();
        }
        coeffs[0] = 0f64;

        let y = x
            .iter()
            .map(|x| {
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(ii, ci)| ci * x.powi(i32::try_from(ii).unwrap()))
                    .sum()
            })
            .collect::<Vec<f64>>();

        let y_max = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let y = y.iter().map(|y| y / y_max).collect();

        for coeff in &mut coeffs {
            *coeff /= y_max;
        }

        GeneratedPolynomial { x, y, coeffs }
    }

    // TODO: Split out the arrangement phase of this function, remove the `allow`
    #[allow(clippy::too_many_lines)]
    #[test]
    fn errors_are_corrected_within_tolerance_for_two_sensor_system() -> Result<()> {
        const DEGREE: usize = 3;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_sensors = 2;
        let num_samples = rng.gen_range(10..255);
        let start = 0.0;
        let end = 1.0;
        let window = Range { start, end };
        let targets = (0..num_sensors)
            .map(|_| {
                Gas((&mut rng)
                    .sample_iter(Alphanumeric)
                    .take(3)
                    .map(char::from)
                    .collect::<String>())
            })
            .collect::<Vec<_>>();
        let mut sensors = HashMap::new();
        let mut measurements = HashMap::new();
        let num_fit_samples = 1;

        let polynomial: GeneratedPolynomial<DEGREE> =
            generate_polynomial(&mut rng, num_samples, window.clone());
        let sample_idxs = targets
            .iter()
            .map(|target| (target, rng.gen_range(0..num_samples)))
            .collect::<HashMap<_, _>>();
        // The input signal in the channel, ie: what we would observe without crosstalk
        let input_signal_in_channel = sample_idxs
            .iter()
            .map(|(target, idx)| (*target, polynomial.x[*idx]))
            .collect::<HashMap<_, _>>();

        for target in &targets {
            let mut measurement = 0.0;
            let polynomial: GeneratedPolynomial<DEGREE> =
                generate_polynomial(&mut rng, num_samples, window.clone());

            // A default calibration for the sensor calibration curve. In this case we are not using
            // the calibration curve, only the crosstalk. However to `build` a `Sensor` the curve is
            // required.
            let calibration = CalibrationData {
                gas: target.clone(),
                concentration: polynomial.y.clone(),
                raw_measurements: polynomial
                    .x
                    .clone()
                    .into_iter()
                    .map(|ln_signal| Measurement {
                        raw_signal: ln_signal.exp(),
                        raw_reference: 1.0,
                        emergent_signal: 1.0,
                        emergent_reference: 1.0,
                    })
                    .collect(),
            };

            let mut sensor = SensorBuilder::new(
                target.clone(),
                1e-10,
                rng.gen(),
                DEGREE - 1,
                num_fit_samples,
            )
            .with_calibration(calibration);

            // The measurement is the sum of the `input_signal_in_channel`, which is the true value
            // of the signal
            measurement += input_signal_in_channel.get(&target).unwrap();

            for cross_target in targets.iter().filter(|gas| *gas != target) {
                let polynomial: GeneratedPolynomial<DEGREE> =
                    generate_polynomial(&mut rng, num_samples, window.clone());
                let crosstalk = CrosstalkData {
                    target_gas: cross_target.clone(),
                    crosstalk_gas: target.clone(),
                    signal: polynomial
                        .x
                        .clone()
                        .into_iter()
                        .map(|ln_signal| Measurement {
                            raw_signal: ln_signal.exp(),
                            raw_reference: 1.0,
                            emergent_signal: 1.0,
                            emergent_reference: 1.0,
                        })
                        .collect(),
                    crosstalk: polynomial
                        .y
                        .clone()
                        .into_iter()
                        .map(|ln_signal| Measurement {
                            raw_signal: ln_signal.exp(),
                            raw_reference: 1.0,
                            emergent_signal: 1.0,
                            emergent_reference: 1.0,
                        })
                        .collect(),
                };

                sensor = sensor.with_crosstalk(crosstalk);

                // The x-axis for this crosstalk in the sensor `target` is the signal in the
                // sensor `cross_target`.
                let sample_idx = sample_idxs.get(cross_target).unwrap().to_owned();
                // The measurement in the channel is the true `input_signal` plus the sum of all
                // crosstalk contributions
                measurement += polynomial.y[sample_idx];
            }

            sensors.insert(target.clone(), sensor.build()?);
            measurements.insert(
                target.clone(),
                crate::margin::Measurement {
                    value: measurement,
                    uncertainty: 0.0,
                },
            );
        }

        // Act
        let initial_guess = None;
        let corrected = correct(
            &measurements,
            &sensors,
            initial_guess,
            &Strategy::CentralValue,
        )?;

        // Assert
        for (target, calculated) in corrected {
            let expected_value = input_signal_in_channel
                .get(&target)
                .expect("value missing from expectations map");
            approx::assert_relative_eq!(calculated.value, expected_value, max_relative = 1e-10);
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    #[test]
    fn reasonable_bounds_are_found_for_two_sensor_system_with_sampling() -> Result<()> {
        const DEGREE: usize = 3;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_sensors = 2;
        let num_samples = rng.gen_range(10..255);
        let start = rng.gen();
        let end = rng.gen_range(2.0..10.0) * start;
        let window = Range { start, end };
        let fractional_spread = rng.gen_range(1e-4..1e-2);

        let targets = (0..num_sensors)
            .map(|_| {
                Gas((&mut rng)
                    .sample_iter(Alphanumeric)
                    .take(3)
                    .map(char::from)
                    .collect::<String>())
            })
            .collect::<Vec<_>>();

        let mut sensors = HashMap::new();
        let mut measurements = HashMap::new();
        let num_fit_samples = 500;

        let polynomial: GeneratedPolynomial<DEGREE> =
            generate_polynomial(&mut rng, num_samples, window.clone());
        let sample_idxs = targets
            .iter()
            .map(|target| (target, rng.gen_range(0..num_samples)))
            .collect::<HashMap<_, _>>();
        let input_signal_in_channel = sample_idxs
            .iter()
            .map(|(target, idx)| (*target, polynomial.x[*idx]))
            .collect::<HashMap<_, _>>();

        for target in &targets {
            let mut measurement = 0.0;
            let polynomial: GeneratedPolynomial<DEGREE> =
                generate_polynomial(&mut rng, num_samples, window.clone());

            // A default calibration for the sensor calibration curve. In this case we are not using
            // the calibration curve, only the crosstalk. However to `build` a `Sensor` the curve is
            // required.
            let calibration = CalibrationData {
                gas: target.clone(),
                concentration: polynomial.y.clone(),
                raw_measurements: polynomial
                    .x
                    .clone()
                    .into_iter()
                    .map(|ln_signal| Measurement {
                        raw_signal: ln_signal.exp(),
                        raw_reference: 1.0,
                        emergent_signal: 1.0,
                        emergent_reference: 1.0,
                    })
                    .collect(),
            };

            let mut sensor =
                SensorBuilder::new(target.clone(), 1e-5, rng.gen(), DEGREE - 1, num_fit_samples)
                    .with_calibration(calibration);

            // The measurement is the sum of the `input_signal_in_channel`, which is the true value
            // of the signal
            measurement += input_signal_in_channel.get(&target).unwrap();

            for cross_target in targets.iter().filter(|gas| *gas != target) {
                let polynomial: GeneratedPolynomial<DEGREE> =
                    generate_polynomial(&mut rng, num_samples, window.clone());
                let crosstalk = CrosstalkData {
                    target_gas: cross_target.clone(),
                    crosstalk_gas: target.clone(),
                    signal: polynomial
                        .x
                        .clone()
                        .into_iter()
                        .map(|ln_signal| Measurement {
                            raw_signal: ln_signal.exp(),
                            raw_reference: 1.0,
                            emergent_signal: 1.0,
                            emergent_reference: 1.0,
                        })
                        .collect(),
                    crosstalk: polynomial
                        .y
                        .clone()
                        .into_iter()
                        .map(|ln_signal| Measurement {
                            raw_signal: ln_signal.exp(),
                            raw_reference: 1.0,
                            emergent_signal: 1.0,
                            emergent_reference: 1.0,
                        })
                        .collect(),
                };

                sensor = sensor.with_crosstalk(crosstalk);
                // The x-axis for this crosstalk in the sensor `target` is the signal in the
                // sensor `cross_target`.
                let sample_idx = sample_idxs.get(cross_target).unwrap().to_owned();
                // The measurement in the channel is the true `input_signal` plus the sum of all
                // crosstalk contributions
                measurement += polynomial.y[sample_idx];
            }
            sensors.insert(target.clone(), sensor.build()?);
            measurements.insert(
                target.clone(),
                crate::margin::Measurement {
                    value: measurement,
                    uncertainty: measurement * fractional_spread,
                },
            );
        }

        // Act
        let initial_guess = None;
        let corrected = correct(
            &measurements,
            &sensors,
            initial_guess,
            &Strategy::SampleDistribution(100),
        )?;

        // Assert
        for (target, calculated) in corrected {
            let expected_value = input_signal_in_channel
                .get(&target)
                .expect("value missing from expectations map");
            assert!((calculated.value - expected_value).abs() < calculated.uncertainty);
        }

        Ok(())
    }

    #[test]
    fn means_and_standard_deviations_are_successfully_reconstructed_from_samples() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let num_samples = rng.gen::<u16>();
        let num_points = 2;
        let mean = rng.gen();
        let standard_deviation: f64 = mean / 100.;
        let samples: Vec<Array1<f64>> = (0..num_samples)
            .map(|_| Array::random(num_points, Normal::new(mean, standard_deviation).unwrap()))
            .collect();

        let measurements = form_measurements(&samples);

        for measurement in measurements {
            approx::assert_relative_eq!(measurement.value, mean, max_relative = 1e-2);
            approx::assert_relative_eq!(
                measurement.uncertainty,
                standard_deviation,
                max_relative = 1e-2
            );
        }
    }
}
