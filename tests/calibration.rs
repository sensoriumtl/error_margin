use std::collections::HashMap;

use ndarray_rand::rand::distributions::Alphanumeric;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
use serde::Serialize;
use tempdir::TempDir;

use error_margin::calibration::Config;
use error_margin::calibration::Gas;
use error_margin::calibration::SensorData;
use error_margin::Result;

fn create_calibration_dir(test_name: &str) -> Result<TempDir> {
    let tmp_dir = TempDir::new(test_name).unwrap();
    let calibration_path = tmp_dir.path().join("calibration");
    std::fs::create_dir(calibration_path).unwrap();
    Ok(tmp_dir)
}

fn generate_gases<R: Rng>(num_gases: usize, rng: &mut R) -> Vec<Gas> {
    let mut gases = Vec::new();
    for _ in 0..num_gases {
        let gas = Gas(rng
            .sample_iter(Alphanumeric)
            .take(3)
            .map(char::from)
            .collect::<String>()
            .to_owned());
        gases.push(gas);
    }
    gases
}

fn create_sensor_dir<R: Rng>(target_gas: &Gas, working_dir: &TempDir, rng: &mut R) -> Result<()> {
    let sensor_dir = working_dir.path().join("calibration").join(&target_gas.0);
    std::fs::create_dir(&sensor_dir).unwrap();

    // Write the generic information for the sensor
    let sensor_data = SensorData {
        noise_equivalent_power: rng.gen::<f64>(),
    };
    std::fs::write(
        sensor_dir.join("sensor.toml"),
        toml::to_string(&sensor_data).unwrap(),
    )
    .unwrap();

    // Generate the polynomials

    Ok(())
}

#[derive(Serialize)]
struct Row {
    concentration: f64,
    raw_signal: f64,
    raw_reference: f64,
    emergent_signal: f64,
    emergent_reference: f64,
}

fn generate_calibration_curves<R: Rng>(
    target_gas: &Gas,
    gases: &[Gas],
    working_dir: &TempDir,
    rng: &mut R,
    polynomial_degree: usize,
    num_samples: usize,
) -> Result<HashMap<Gas, Vec<f64>>> {
    let sensor_dir = working_dir.path().join("calibration").join(&target_gas.0);
    let mut coeffs_at_sensor = HashMap::new();

    for gas in gases {
        // The polynomial coefficients
        let coeffs = (0..=polynomial_degree)
            .map(|_| rng.gen::<f64>())
            .collect::<Vec<_>>();

        // Generate the data
        let concentrations = (0..num_samples)
            .map(|n| n as f64 / num_samples as f64)
            .collect::<Vec<_>>();

        // We fix the signals in both emergent channels and the reference signal to unity, here we
        // are just calculating the raw signal at the active channel given by the coeffs above
        let raw_signal = concentrations
            .iter()
            .map(|x| {
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(ii, c)| c * x.powi(ii as i32))
                    .fold(0., |a, b| a + b)
            })
            .map(|x| 10f64.powf(x)) // Fit signals are the base10 log of raw signals, so we invert
            // here to get the data into the expected format
            .collect::<Vec<_>>();

        let mut wtr = csv::Writer::from_path(sensor_dir.join(format!("{}.csv", gas.0))).unwrap();
        for (c, v) in concentrations.iter().zip(raw_signal.iter()) {
            let row = Row {
                concentration: *c,
                raw_signal: *v,
                raw_reference: 1.0,
                emergent_signal: 1.0,
                emergent_reference: 1.0,
            };
            wtr.serialize(&row).unwrap();
        }

        coeffs_at_sensor.insert(gas.clone(), coeffs);
    }

    Ok(coeffs_at_sensor)
}

#[test]
fn single_gas_sensor_fit_matches_input_coefficients() -> Result<()> {
    let seed = 40;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    // Arrange
    let tmp_dir = create_calibration_dir("single_gas_sensor_fit_matches_input_coefficients")?;
    let gases = generate_gases(1, &mut rng);

    let polynomial_degree = rng.gen_range(2..6);
    let num_samples = rng.gen_range(10..100);

    let mut input_polynomial_coefficients = HashMap::new();

    for gas in gases.iter() {
        create_sensor_dir(gas, &tmp_dir, &mut rng)?;
        input_polynomial_coefficients.insert(
            gas,
            generate_calibration_curves(
                gas,
                &gases,
                &tmp_dir,
                &mut rng,
                polynomial_degree,
                num_samples,
            )?,
        );
    }

    let config = Config {
        polynomial_fit_degree: polynomial_degree,
        operating_frequency: rng.gen::<u8>() as f64,
    };

    let sensors = error_margin::calibration::build::<f64>(&tmp_dir.into_path(), &config)?;

    assert_eq!(sensors.len(), 1);

    let sensor = &sensors[0];
    let calculated_calibration = sensor.calibration();
    let calculated_coefficients = calculated_calibration.solution();

    let expected_coefficients = input_polynomial_coefficients
        .get(&gases[0])
        .expect("gas missing from map")
        .get(&gases[0])
        .expect("gas missing from map");

    for (expected, calculated) in expected_coefficients
        .iter()
        .zip(calculated_coefficients.iter())
    {
        approx::assert_relative_eq!(expected, calculated, max_relative = 1e-4,);
    }

    Ok(())
}

#[test]
fn multi_gas_sensor_fit_matches_input_coefficients() -> Result<()> {
    let seed = 40;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    // Arrange
    let tmp_dir = create_calibration_dir("multi_gas_sensor_fit_matches_input_coefficients")?;
    let num_gases = rng.gen_range(2..10);
    let gases = generate_gases(num_gases, &mut rng);

    let polynomial_degree = rng.gen_range(2..6);
    let num_samples = rng.gen_range(10..100);

    let mut input_polynomial_coefficients = HashMap::new();

    for gas in gases.iter() {
        create_sensor_dir(gas, &tmp_dir, &mut rng)?;
        input_polynomial_coefficients.insert(
            gas,
            generate_calibration_curves(
                gas,
                &gases,
                &tmp_dir,
                &mut rng,
                polynomial_degree,
                num_samples,
            )?,
        );
    }

    let config = Config {
        polynomial_fit_degree: polynomial_degree,
        operating_frequency: rng.gen::<u8>() as f64,
    };

    let sensors = error_margin::calibration::build::<f64>(&tmp_dir.into_path(), &config)?;

    assert_eq!(sensors.len(), num_gases);

    for sensor in sensors.iter() {
        let target_gas = sensor.target();
        let expected_coefficients_for_sensor = input_polynomial_coefficients
            .get(&target_gas)
            .expect("target gas present in output but missing from input");

        // Check the calibration
        let calculated_calibration = sensor.calibration().solution();
        let expected_calibration = expected_coefficients_for_sensor
            .get(target_gas)
            .expect("sensor input missing calibration curve");
        for (expected, calculated) in expected_calibration
            .iter()
            .zip(calculated_calibration.iter())
        {
            approx::assert_relative_eq!(expected, calculated, max_relative = 1e-4,);
        }

        // Check the crosstalk
        for (gas, crosstalk) in sensor.crosstalk() {
            let calculated_crosstalk = crosstalk.solution();
            let expected_crosstalk = expected_coefficients_for_sensor
                .get(gas)
                .expect("sensor input missing crosstalk curve");
            for (expected, calculated) in expected_crosstalk.iter().zip(calculated_crosstalk.iter())
            {
                approx::assert_relative_eq!(expected, calculated, max_relative = 1e-4,);
            }
        }
    }

    Ok(())
}
