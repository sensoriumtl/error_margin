use std::collections::HashMap;
use std::f64::EPSILON;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::solver::gaussnewton::GaussNewtonLS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{Array1, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};

use crate::calibration::Gas;
use crate::minimisation::Problem;
use crate::polyfit::Polynomial;
use crate::Result;
use crate::{calibration::Sensor, margin::Measurement};

fn into_map<E>(solution: Array1<E>, measurement_targets: &[Gas]) -> HashMap<Gas, E> {
    assert_eq!(solution.len(), measurement_targets.len());
    let mut map = HashMap::new();
    for (target, element) in measurement_targets.iter().zip(solution) {
        map.insert(target.clone(), element);
    }
    map
}

pub fn correct<
    E: Lapack
        + PartialOrd
        + Scalar
        + ScalarOperand
        + std::convert::Into<f64>
        + std::convert::From<f64>,
>(
    raw_measurements: HashMap<Gas, Measurement<E>>,
    sensors: HashMap<Gas, Sensor<E>>,
    initial_parameters: Option<Array1<E>>,
) -> Result<HashMap<Gas, Measurement<E>>> {
    let measurement_targets = raw_measurements.keys().cloned().collect::<Vec<_>>();
    let problem = Problem::build(&measurement_targets, &raw_measurements, &sensors).to_f64();

    // TODO: casting to f64 because `solve` does not work with generics
    let initial_parameters: Array1<f64> = initial_parameters.map_or_else(
        || Array1::zeros(measurement_targets.len()),
        |initial_parameters| initial_parameters.mapv(|x| x.into()),
    );

    let solution: Array1<E> = problem.solve(initial_parameters)?.mapv(|x| x.into());
    let solution = into_map(solution, &measurement_targets)
        .into_iter()
        .map(|(target, value)| {
            (
                target,
                Measurement {
                    value,
                    uncertainty: E::zero(),
                },
            )
        })
        .collect::<HashMap<_, _>>();

    Ok(solution)
    // println!("{measurements:?}");
    // // Check all the sensors in the list have an associated measurement
    // // TODO: Handle assertion failures gracefully with error handling.
    // assert_eq!(measurement_targets.len(), sensors.len());
    // for target in measurement_targets.iter() {
    //     assert!(sensors.get(target).is_some());
    // }
    // //
    // // Unwrap is safe because we panic above if any targets are missing from sensors
    // // The above panic will (eventually) be handled so we can unwrap here safely
    // let degree = sensors.get(&measurement_targets[0]).unwrap().calibration().solution().len();
    //
    //
    // // The matrix we want is len(measurements) x ((len(measurements) - 1) * degree)
    // let mut mat: Array2<E> = Array2::zeros((measurement_targets.len(), (measurement_targets.len()) * (degree - 1)));
    // let mut measure: Array1<E> = Array1::zeros(measurement_targets.len());
    //
    // let mut exp_vec: Array1<f64> = Array1::zeros(measurement_targets.len());
    // for (ii, target) in measurement_targets.iter().enumerate() {
    //     println!("{ii} {target:?}");
    //     let sensor = sensors.get(target).unwrap();
    //
    //     // The linear element is the diagonal one
    //     mat[[ii, (ii*(degree - 1))]] = E::one();
    //
    //     measure[ii] = measurements.get(target).unwrap().value;
    //     println!("{ii} {:?}", measure[ii]);
    //
    //     exp_vec[ii] = expected.get(target).unwrap().to_owned();
    //
    //
    //     // Crosstalk elements are
    //
    //     // Get the zero-point offset
    //     let sum_of_zero_order = sensor.crosstalk()
    //         .values()
    //         .map(|coeffs| coeffs.solution()[0])
    //         .sum();
    //
    //     measure[ii] -= sum_of_zero_order;
    //
    //     for (gas, crosstalk_coeffs) in sensor.crosstalk() {
    //         println!("{target:?}, {gas:?}, {:?}", crosstalk_coeffs.solution());
    //         let index_of_gas = measurement_targets.iter().position(|target| *target == gas).expect("gas present in crosstalk but absent from measurements");
    //         mat.slice_mut(s![ii, (index_of_gas*(degree-1))..((index_of_gas+1)*(degree-1))])
    //             .assign(&crosstalk_coeffs.solution().slice(s![1..]));
    //     }
    // }
    //
    // dbg!(&mat);
    // dbg!(&measure);
    //
    // let matrix: Array2<f64> = mat.mapv(|ele| ele.into());
    // let vector: Array1<f64> = measure.mapv(|ele| ele.into());
    // let cost = Problem { matrix: matrix.clone(), vector, degree: degree - 1};

    // let res = solve(cost.clone())?
    //     .mapv(|ele| E::from(ele).unwrap());
    //
    // let mut results = HashMap::new();
    // for (ii, target) in measurement_targets.into_iter().enumerate() {
    //     results.insert(target.clone(), Measurement { value: res[ii], uncertainty: E::zero() });
    // }
    //
    // let sanity_check = matrix;
    // let vec = cost.promote(&res.mapv(|ele| ele.into()));
    // let prod = sanity_check.dot(&vec);
    // println!("prod: {prod:?}");
    // println!("measure: {measure:?}");
    // println!("cost: {}", (prod - measure.mapv(|ele| ele.into())).norm());
    //
    // let exp_vec = cost.promote(&exp_vec);
    // println!("exp vec: {exp_vec:?}");
    // let prod = sanity_check.dot(&exp_vec);
    // println!("prod atc: {prod:?}");
    // println!("cost: {}", (prod - measure.mapv(|ele| ele.into())).norm());
    //
    //
    // Ok(results)
}

pub fn reconstruct<E: Lapack + PartialOrd + Scalar + ScalarOperand>(
    true_measurement: &Measurement<E>, // Measurement after error correction with corresponding
    // uncertainty
    sensor: &Sensor<E>,
) -> Result<Measurement<E>> {
    let calibration_curve = sensor.calibration();
    assert!(
        calibration_curve.window_contains(&true_measurement.value),
        "measurement point {} must be in the range of calibration {:?}",
        true_measurement.value,
        calibration_curve.window()
    );

    let polynomial = Polynomial::from(calibration_curve.clone());

    let value = true_measurement.compute_unknown(&polynomial)?;

    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, ops::Range};

    use ndarray_rand::{
        rand::{Rng, SeedableRng},
        rand_distr::Alphanumeric,
    };
    use rand_isaac::Isaac64Rng;

    use crate::{
        calibration::{CalibrationData, Gas, Measurement, SensorBuilder},
        multi::correct,
        Result,
    };

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

    #[test]
    fn errors_are_corrected_within_tolerance_for_two_sensor_system() -> Result<()> {
        const DEGREE: usize = 3;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let num_sensors = 2;

        let num_samples = rng.gen_range(10..255);

        let start = rng.gen();
        let end = rng.gen_range(2.0..10.0) * start;
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

        // Generate a dummy polynomial, we do this because in this test all polynomials have the
        // same x-axis. This allows us to generate `signal_in_channel` which
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

        for target in targets.iter() {
            let mut measurement = 0.0;
            let polynomial: GeneratedPolynomial<DEGREE> =
                generate_polynomial(&mut rng, num_samples, window.clone());

            // A default calibration for the sensor calibration curve. In this case we are not using
            // the calibration curve, only the crosstalk. However to `build` a `Sensor` the curve is
            // required.
            let calibration = CalibrationData {
                gas: target.clone(),
                concentration: polynomial.x.clone(),
                raw_measurements: polynomial
                    .y
                    .clone()
                    .into_iter()
                    .map(|log_signal| Measurement {
                        raw_signal: 10f64.powf(log_signal),
                        raw_reference: 1.0,
                        emergent_signal: 1.0,
                        emergent_reference: 1.0,
                    })
                    .collect(),
            };

            let mut sensor = SensorBuilder::new(target.clone(), rng.gen(), rng.gen(), DEGREE - 1)
                .with_calibration(calibration);

            // The measurement is the sum of the `input_signal_in_channel`, which is the true value
            // of the signal
            measurement += input_signal_in_channel.get(&target).unwrap();

            for cross_target in targets.iter().filter(|gas| *gas != target) {
                let polynomial: GeneratedPolynomial<DEGREE> =
                    generate_polynomial(&mut rng, num_samples, window.clone());
                println!(
                    "generating {target:?}, {cross_target:?}, {:?}",
                    polynomial.coeffs
                );
                let crosstalk = CalibrationData {
                    gas: cross_target.clone(),
                    concentration: polynomial.x.clone(),
                    raw_measurements: polynomial
                        .y
                        .clone()
                        .into_iter()
                        .map(|log_signal| Measurement {
                            raw_signal: 10f64.powf(log_signal),
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
        let corrected = correct(measurements, sensors, initial_guess)?;

        // Assert
        for (target, calculated) in corrected {
            let expected_value = input_signal_in_channel
                .get(&target)
                .expect("value missing from expectations map");
            approx::assert_relative_eq!(calculated.value, expected_value, max_relative = 1e-10);
        }

        Ok(())
    }
}
