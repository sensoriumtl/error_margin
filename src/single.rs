use ndarray::ScalarOperand;
use ndarray_linalg::{Lapack, Scalar};

use crate::polyfit::Polynomial;
use crate::{calibration::Sensor, margin::Measurement};

pub fn reconstruct<E: Lapack + PartialOrd + Scalar + ScalarOperand>(
    measurement: &Measurement<E>,
    sensor: &Sensor<E>,
) -> Measurement<E> {
    // When we have a single measurement we can just use the data from the fit, and the uncertainty
    // or the measurement. This gives

    assert!(sensor.crosstalk().is_empty(), "for a single measurement there is only one absorbing species, so crosstalk is not supported");
    let calibration_curve = sensor.calibration();
    assert!(
        calibration_curve.window_contains(&measurement.value),
        "measurement point {} must be in the range of calibration {:?}",
        measurement.value,
        calibration_curve.window()
    );

    let polynomial = Polynomial::from(calibration_curve.clone());

    measurement.compute_unknown(&polynomial)
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use ndarray_rand::{
        rand::{Rng, SeedableRng},
        rand_distr::Alphanumeric,
    };
    use rand_isaac::Isaac64Rng;

    use crate::{
        calibration::{CalibrationData, Gas, Measurement, SensorBuilder},
        single::reconstruct,
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
    fn values_are_reconstructed_with_zero_measurement_error() -> Result<()> {
        const DEGREE: usize = 3;

        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let _num_samples = rng.gen_range(10..255);
        let num_samples = 20;
        let start = rng.gen();
        let end = rng.gen_range(2.0..10.0) * start;
        let window = Range { start, end };
        let polynomial: GeneratedPolynomial<DEGREE> =
            generate_polynomial(&mut rng, num_samples, window);

        let concentration = polynomial.y.clone();
        let raw_measurements: Vec<Measurement<f64>> = polynomial.x.clone()
                .into_iter()
                .map(|ln_signal| Measurement {
                    raw_signal: std::f64::consts::E.powf(ln_signal),
                    raw_reference: 1.0,
                    emergent_signal: 1.0,
                    emergent_reference: 1.0,
                })
                .collect();

        let x = raw_measurements.iter().map(Measurement::scaled).collect::<Vec<_>>();

        let target = Gas((&mut rng)
            .sample_iter(Alphanumeric)
            .take(3)
            .map(char::from)
            .collect::<String>());

        let calibration = CalibrationData {
            gas: target.clone(),
            concentration: concentration.clone(),
            raw_measurements,
        };

        let num_fit_samples = 300;

        let sensor = SensorBuilder::new(target, 1e-10, rng.gen(), DEGREE - 1, num_fit_samples)
            .with_calibration(calibration)
            .build()?;

        for (value, actual) in x
            .into_iter()
            .zip(concentration)
            .skip(1)
            .take_while(|(x, _)| sensor.calibration().window_contains(x))
        {
            let measurement = crate::margin::Measurement {
                value,
                uncertainty: value * 1e-10,
            };
            let reconstruction = reconstruct(&measurement, &sensor);
            approx::assert_relative_eq!(actual, reconstruction.value, max_relative = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn values_are_reconstructed_with_finite_measurement_error() -> Result<()> {
        const DEGREE: usize = 5;

        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let num_samples = rng.gen_range(10..255);
        let start = rng.gen();
        let end = rng.gen_range(2.0..10.0) * start;
        let window = Range { start, end };
        let polynomial: GeneratedPolynomial<DEGREE> =
            generate_polynomial(&mut rng, num_samples, window);

        let concentration = polynomial.y.clone();
        let raw_measurements: Vec<Measurement<f64>> = polynomial.x.clone()
                .into_iter()
                .map(|ln_signal| Measurement {
                    raw_signal: std::f64::consts::E.powf(ln_signal),
                    raw_reference: 1.0,
                    emergent_signal: 1.0,
                    emergent_reference: 1.0,
                })
                .collect();

        let target = Gas((&mut rng)
            .sample_iter(Alphanumeric)
            .take(3)
            .map(char::from)
            .collect::<String>());

        let x = raw_measurements.iter().map(Measurement::scaled).collect::<Vec<_>>();

        let calibration = CalibrationData {
            gas: target.clone(),
            concentration: concentration.clone(),
            raw_measurements,
        };


        let num_fit_samples = 300;

        // TODO: Sensor has a different degree, to that in the generated polynomial
        let sensor = SensorBuilder::new(target, 0.1, rng.gen(), DEGREE - 1, num_fit_samples)
            .with_calibration(calibration)
            .build()?;


        for (value, actual) in x
            .into_iter()
            .zip(concentration)
            .skip(1)
            .take_while(|(x, _)| sensor.calibration().window_contains(x))
        {
            let measurement = crate::margin::Measurement {
                value,
                uncertainty: value * 1e-3,
            };
            let reconstruction = reconstruct(&measurement, &sensor);
            assert!((actual - reconstruction.value) < reconstruction.uncertainty);
        }

        Ok(())
    }
}
