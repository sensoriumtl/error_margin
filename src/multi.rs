use std::collections::HashMap;
use std::marker::PhantomData;

use argmin::core::observers::{SlogLogger, ObserverMode};
use argmin::solver::gaussnewton::GaussNewtonLS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{ScalarOperand, Array2, s, Array1, Array};
use ndarray_linalg::{Lapack, Scalar, LeastSquaresSvd};

use crate::calibration::Gas;
use crate::polyfit::Polynomial;
use crate::Result;
use crate::{calibration::Sensor, margin::Measurement};

use argmin::core::{Operator, Jacobian, Executor};


struct Problem<E> {
    /// The matrix is the coefficient matrix, holding all the polynomial coefficients
    /// and unity for the current sensor
    ///
    /// TODO explain better
    matrix: Array2<E>,
    /// The vector is the observed signal, minus any zero order polynomial crosstalk
    /// coefficients (although these should be zero)
    vector: Array1<E>,
    /// Polynomial degree. We assume implicitly at this stage that all the sensors are described by
    /// polynomials of the same order. This might relax later??
    degree: usize,
}

impl<E: Scalar> Problem<E> {
    /// Calculates the polynomial vector of passed parameters
    ///
    /// The parameters are passed as [x_1, x_2, x_3, ..., x_n]
    /// while the promoted vector is [x_1, x_1^2,...,x_n^degree]
    /// where degree is the degree of the underlying polynomial used in the reconstruction
    fn promote(&self, params: &Array1<E>) -> Array1<E> {
        let mut promoted = Array1::zeros(params.len() * self.degree);
        for (ii, p) in params.iter().enumerate() {
            for jj in 0..self.degree {
                promoted[ii * self.degree + jj] = p.powi(i32::try_from(jj + 1).unwrap());
            }
        }
        promoted
    }

    fn compute(&self, params: &Array1<E>) -> Array1<E> {
        self.matrix.dot(&self.promote(params))

    }

    fn jacobian_column(&self, params: &Array1<E>, ii: usize) -> Array1<E> {
        let mut promoted = Array1::zeros(params.len() * self.degree);
        for (jj, p) in params.iter().enumerate() {
            for kk in 0..self.degree {
                if jj == ii {
                    // If we are at jj then we take the derivative
                    promoted[jj * self.degree + kk] = E::from(kk + 1).unwrap() * p.powi(i32::try_from(kk).unwrap());
                } else {
                    // If not the derivative is zero
                    promoted[jj * self.degree + kk] = E::zero();
                }
            }
        }
        self.matrix.dot(&promoted)
    }
}


impl<E: Scalar> Operator for Problem<E> {
    type Param = Array1<E>;
    type Output = Array1<E>;

    fn apply(&self, p: &Self::Param) -> ::std::result::Result<Self::Output, argmin::core::Error> {
        // Promote the `p` vector
        Ok(self.compute(p) - &self.vector)
    }
}

impl<E: Scalar> Jacobian for Problem<E> {
    type Param = Array1<E>;
    type Jacobian = Array2<E>;

    fn jacobian(&self, p: &Self::Param) -> ::std::result::Result<Self::Jacobian, argmin::core::Error> {
        let mut jacobian = Array2::zeros((p.len(), p.len()));
        for jj in 0..p.len() {
            let col = self.jacobian_column(p, jj);
            jacobian.slice_mut(s![.., jj])
                .assign(&col);
        }
        Ok(jacobian)
    }
}

pub fn correct<E: Lapack + PartialOrd + Scalar + ScalarOperand + std::convert::Into<f64>>(
    measurements: HashMap<Gas, Measurement<E>>, // Measurement after error correction with corresponding
    // uncertainty
    sensors: HashMap<Gas, Sensor<E>>,
) -> Result<HashMap<Gas, Measurement<E>>>
{
    // Check all the sensors in the list have an associated measurement
    let measurement_targets = measurements.keys().collect::<Vec<_>>();
    assert_eq!(measurement_targets.len(), sensors.len());
    for target in measurement_targets.iter() {
        assert!(sensors.get(target).is_some());
    }
    // We need to reconstruct and get the signals back. It's a simple matrix inversion
    //
    // First we get the Vandermode matrix for the fit
    // Unwrap is safe because we panic above if any targets are missing from sensors
    let degree = sensors.get(&measurement_targets[0]).unwrap().calibration().solution().len();

    // The matrix we want is len(measurements) x ((len(measurements) - 1) * degree)
    let mut mat: Array2<E> = Array2::zeros((measurement_targets.len(), (measurement_targets.len()) * (degree - 1)));
    let mut measure: Array1<E> = Array1::zeros(measurement_targets.len());
    for (ii, target) in measurement_targets.iter().enumerate() {
        println!("{ii} {target:?}");
        let sensor = sensors.get(target).unwrap();

        // The linear element is the diagonal one
        mat[[ii, (ii*(degree - 1))]] = E::one();

        measure[ii] = measurements.get(target).unwrap().value;
        println!("{ii} {:?}", measure[ii]);

        // Crosstalk elements are
        for (gas, crosstalk_coeffs) in sensor.crosstalk() {
            let index_of_gas = measurement_targets.iter().position(|target| *target == gas).expect("gas present in crosstalk but absent from measurements");
            mat.slice_mut(s![ii, (index_of_gas*(degree-1))..((index_of_gas+1)*(degree-1))])
                .assign(&crosstalk_coeffs.solution().slice(s![1..]));
            measure[ii] -= crosstalk_coeffs.solution()[0];
        }
    }

    let mat: Array2<f64> = mat.mapv(|ele| ele.into());
    let measure: Array1<f64> = measure.mapv(|ele| ele.into());
    let cost = Problem { matrix: mat, vector: measure, degree: degree - 1};

    let res = solve(cost)?
        .mapv(|ele| E::from(ele));

    todo!()
}

fn solve(cost: Problem<f64>) -> Result<Array1<f64>> {

    let linesearch = MoreThuenteLineSearch::new().with_bounds(0.0, 1.0)?;

    // Define initial parameter vector
    let init_param: Array1<f64> = Array1::from(vec![0.9, 0.2]);

    // Set up solver
    let solver = GaussNewtonLS::new(linesearch);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(10))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    todo!()

    // The vec of unknowns is ordered as (x_1^0, x_1, x_1^2, ..., x_m^degree)
    // so we need to fill with the crosstalk and calibration coefficients
    // then we need to invert `mat` to get the vector of unknowns
    // from which we can take element 1 + i * degree
//     //
//     // We can check the errors by seeing that x_i^0 = 1, x_i^n = x_1^n
//     // If this is the case the calculation is self consistent. If not we can probably assess the
//     // error by looking at the spread x_i, sqrt(x_i^2) etc and seeing how far away we are.
//     todo!()
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
    use std::{ops::Range, collections::HashMap};

    use ndarray::{Array1, Array2};
    use ndarray_rand::{rand::{Rng, SeedableRng}, rand_distr::Alphanumeric};
    use rand_isaac::Isaac64Rng;

    use crate::{Result, calibration::{Gas, CalibrationData, Measurement, SensorBuilder}, multi::correct};

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
    fn errors_are_corrected_within_tolerance() -> Result<()> {
        const DEGREE: usize = 3;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let num_sensors = 2;

        let num_samples = rng.gen_range(10..255);

        let start = rng.gen();
        let end = rng.gen_range(2.0..10.0) * start;
        let window = Range { start, end };

        let targets = (0..num_sensors)
            .map(|_| Gas((&mut rng)
                .sample_iter(Alphanumeric)
                .take(3)
                .map(char::from)
                .collect::<String>())
            )
            .collect::<Vec<_>>();

        let mut sensors = HashMap::new();
        let mut measurements = HashMap::new();
        for target in targets.iter() {
            let mut measurement = 0.0;
            let polynomial: GeneratedPolynomial<DEGREE> =
                generate_polynomial(&mut rng, num_samples, window.clone());
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

            measurement += polynomial.x[num_samples/2];
            println!("m: {measurement}");

            let mut sensor = SensorBuilder::new(target.clone(), 0.1, rng.gen(), DEGREE - 1)
                .with_calibration(calibration);

            for cross_target in targets.iter().filter(|gas| *gas != target) {
                let polynomial: GeneratedPolynomial<DEGREE> =
                    generate_polynomial(&mut rng, num_samples, window.clone());
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
                println!("{target:?} ({cross_target:?}): {:?}", polynomial.coeffs);
                measurement += polynomial.y[num_samples / 2];
            }

            sensors.insert(target.clone(), sensor.build()?);
            measurements.insert(target.clone(), crate::margin::Measurement { value: measurement, uncertainty: 0.0 });
        }

        correct(measurements, sensors);


        Ok(())




    }

    use super::Problem;

    #[test]
    fn parameter_vectors_are_correctly_promoted_to_polynomials_form() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let num_params = rng.gen_range(5..20);
        let params: Array1<f64> = Array1::from_iter((0..num_params).map(|_| rng.gen()));
        let degree = rng.gen_range(3..10);

        let problem = Problem { matrix: Array2::zeros((1, 1)), vector: Array1::zeros(1), degree };

        let promoted = problem.promote(&params);

        for (ii, ele) in promoted.into_iter().enumerate() {
            let element_index = ii / degree;
            let expected_power = ii % degree;

            approx::assert_relative_eq!(ele, params[element_index].powi(i32::try_from(expected_power + 1).unwrap()));

        }
    }

    use ndarray_rand::{RandomExt, rand_distr::Uniform};

    #[test]
    fn jacobian_columns_matches_finite_difference() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // let num_params = rng.gen_range(5..20);
        let num_params = 5;
        let params: Array1<f64> = Array1::from_iter((0..num_params).map(|_| rng.gen()));
        // let degree = rng.gen_range(3..10);
        let degree = 3;

        let matrix = Array2::random((num_params, num_params * degree), Uniform::new(0., 10.));

        let problem = Problem { matrix, vector: Array1::zeros(1), degree };

        let delta_rel = 1e-6;

        for jj in 0..num_params {
            let computed_jacobian_col = problem.jacobian_column(&params, jj);
            let mut modified_params_plus = params.clone();
            let delta = modified_params_plus[jj] * delta_rel;
            modified_params_plus[jj] += delta;
            let mut modified_params_minus = params.clone();
            modified_params_minus[jj] -= delta;

            let computed_at_plus= problem.compute(&modified_params_plus);
            let computed_at_minus= problem.compute(&modified_params_minus);
            let numerical_col = (computed_at_plus - computed_at_minus) / (2. * delta);

            for (comp, num) in computed_jacobian_col.into_iter().zip(numerical_col) {
                approx::assert_relative_eq!(comp, num, max_relative=1e-4);
            }

        }
    }
}
