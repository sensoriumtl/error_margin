use std::collections::HashMap;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{Executor, Jacobian, Operator};
use argmin::solver::gaussnewton::GaussNewtonLS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{s, Array1, Array2};
use ndarray_linalg::Scalar;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num_traits::Float;

use crate::calibration::{Gas, Sensor};
use crate::margin::Measurement;
use crate::Result;

#[derive(Clone)]
pub struct Problem<E> {
    /// The matrix is the coefficient matrix
    ///
    /// The coefficients matrix is a rectangular matrix of dimension (num_sensors x (degree - 1) *
    /// num_sensors). Each row corresponds to a single sensor, with elements describing the true
    /// signal and crosstalk coefficients.
    ///
    /// The matrix left-multiplies the parameter vector. Which is comprised of the signals in each
    /// individual channel, raised to powers from 1->degree
    /// $$
    ///     ``v = \left[x_1, x_1^2, \dots, x_n^{degree}\right]
    /// $$
    ///
    /// The non-zero elements in a row of `matrix` are as follows: the linear term corresponding to signal
    /// in the sensor of that row is unity. Other elements for the sensor of that row are nil.
    /// Elements corresponding to other sensors are filled with the crosstalk polynomial fitting
    /// coefficients for the relevent power of the detector signal in that channel.
    matrix: Array2<E>,
    /// The vector is the observed signal (without correction), minus any zero order polynomial crosstalk
    /// coefficients (although these should be zero)
    /// $$
    ///     v = \left[A_1, A_2, \dots, A_n\right] - \sum_j \left[c_{01j}, c_{02j}, \dots,
    ///     c_{0nj}\right]
    /// $$
    /// where $A_i$ is the *measured* signal in channel *i* and $c_{0ij}$ is the zero-order
    /// polynomial fit term for the crosstalk from channel $j$, observed in channel $i$.
    lhs: Array1<E>,
    /// Polynomial degree. We assume implicitly at this stage that all the sensors are described by
    /// polynomials of the same order. This might relax later??
    degree: usize,
}

impl<E: Scalar> Problem<E> {
    /// This function DOES NOT sanity check what is passed to it.
    ///
    /// It is assumed that the caller has checked the contents of the passed maps, to ensure they
    /// are complete and consistent. If this is not the case the function will panic.
    /// Particularly note that all elements of `measurement_targets` must be present in `sensors`.
    /// Additionally each element of `sensors` must form a set, wherein the sensor target and all
    /// the gases in the `crosstalk` field coalesce with the set `measurement_targets`.
    ///
    /// TODO: handle this error, and express this requirement in the type system so we cannot have
    /// invalid inputs at this point.
    ///
    /// Note: We pass `measurement_targets` seperately so we have an ordered quantity around which
    /// to construct our matrix. The rows of the matrix correspond to sensors targeting the items
    /// in `measurement_targets` in that order. TODO: Get rid of this needless additional argument.
    pub(crate) fn build(
        measurement_targets: &[Gas],
        raw_measurements: &HashMap<Gas, Measurement<E>>,
        sensors: &HashMap<Gas, Sensor<E>>,
    ) -> Self {
        // Check all the sensors in the list have an associated measurement
        // TODO: Handle assertion failures gracefully with error handling.
        assert_eq!(measurement_targets.len(), sensors.len());
        for target in measurement_targets {
            assert!(sensors.get(target).is_some());
        }

        // Unwrap is safe because we panic above if any targets are missing from sensors
        // The above panic will (eventually) be handled so we can unwrap here safely
        let degree = sensors
            .get(&measurement_targets[0])
            .unwrap()
            .calibration()
            .solution()
            .len()
            - 1;

        let mut matrix: Array2<E> = Array2::zeros((
            measurement_targets.len(),
            (measurement_targets.len()) * degree,
        ));
        let mut lhs: Array1<E> = Array1::zeros(measurement_targets.len());

        for (ii, target) in measurement_targets.iter().enumerate() {
            // This unwrap is safe because we already panicked above if any targets are missing
            // from sensors
            let sensor = sensors.get(target).unwrap();

            // The element of the matrix corresponding to the linear signal for the `ii`th sensor
            // is unity.
            matrix[[ii, ii * degree]] = E::one();

            // The lhs vector is the `raw_measurements` minus any zero order contributions from the
            // crosstalk polynomials (ie: bits not proportional to any signal)
            lhs[ii] = raw_measurements.get(target).unwrap().value
                - sensor
                    .crosstalk()
                    .values()
                    .map(|coeffs| coeffs.solution()[0])
                    .sum();

            for (gas, crosstalk_coeffs) in sensor.crosstalk() {
                // This unwrap is safe because we panic at the function head if any sensor does not
                // have crosstalk spanning `measurement_targets`
                // TODO: This is not true, check at the function head.
                //
                // This is the row index of `gas` in `matrix`
                let index_of_gas = measurement_targets
                    .iter()
                    .position(|target| target == gas)
                    .unwrap();

                // Assign crosstalk coefficients for sensor `ii` due to gas with sensor at
                // `index_of_gas`
                matrix
                    .slice_mut(s![
                        ii,
                        (index_of_gas * degree)..((index_of_gas + 1) * degree)
                    ])
                    .assign(&crosstalk_coeffs.solution().slice(s![1..]));
            }
        }

        Self {
            matrix,
            lhs,
            degree,
        }
    }
}

impl<E> Problem<E>
where
    E: Copy + Float + Scalar,
    StandardNormal: Distribution<E>,
{
    pub(crate) fn build_with_sampling(
        measurement_targets: &[Gas],
        raw_measurements: &HashMap<Gas, Measurement<E>>,
        sensors: &HashMap<Gas, Sensor<E>>,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        // Check all the sensors in the list have an associated measurement
        // TODO: Handle assertion failures gracefully with error handling.
        assert_eq!(measurement_targets.len(), sensors.len());
        for target in measurement_targets {
            assert!(sensors.get(target).is_some());
        }

        // Unwrap is safe because we panic above if any targets are missing from sensors
        // The above panic will (eventually) be handled so we can unwrap here safely
        let degree = sensors
            .get(&measurement_targets[0])
            .unwrap()
            .calibration()
            .solution()
            .len()
            - 1;

        let mut matrix: Array2<E> = Array2::zeros((
            measurement_targets.len(),
            (measurement_targets.len()) * degree,
        ));
        let mut lhs: Array1<E> = Array1::zeros(measurement_targets.len());

        for (ii, target) in measurement_targets.iter().enumerate() {
            // This unwrap is safe because we already panicked above if any targets are missing
            // from sensors
            let sensor = sensors.get(target).unwrap();

            // The element of the matrix corresponding to the linear signal for the `ii`th sensor
            // is unity.
            matrix[[ii, ii * degree]] = E::one();

            // The lhs vector is the `raw_measurements` minus any zero order contributions from the
            // crosstalk polynomials (ie: bits not proportional to any signal)
            lhs[ii] = raw_measurements.get(target).unwrap().sample(rng)?
                - sensor
                    .crosstalk()
                    .values()
                    .map(|coeffs| coeffs.sample_zero_order_coeff(rng))
                    .sum::<Result<E>>()?;

            for (gas, crosstalk_coeffs) in sensor.crosstalk() {
                // This unwrap is safe because we panic at the function head if any sensor does not
                // have crosstalk spanning `measurement_targets`
                // TODO: This is not true, check at the function head.
                //
                // This is the row index of `gas` in `matrix`
                let index_of_gas = measurement_targets
                    .iter()
                    .position(|target| target == gas)
                    .unwrap();

                // Assign crosstalk coefficients for sensor `ii` due to gas with sensor at
                // `index_of_gas`
                matrix
                    .slice_mut(s![
                        ii,
                        (index_of_gas * degree)..((index_of_gas + 1) * degree)
                    ])
                    .assign(&crosstalk_coeffs.sample_higher_order_coeffs(rng)?);
            }
        }

        Ok(Self {
            matrix,
            lhs,
            degree,
        })
    }
}

impl Problem<f64> {
    /// Run the optimisation
    ///
    /// TODO: Currently the solver, linesearch algorithm and parameters are fixed. In future we
    /// probably want to allow these to be changed by configuration file or at runtime.
    ///
    /// TODO: Can't work out the trait bounds for this to be generic, in future re-write as
    /// generic.
    pub(crate) fn solve(self, initial_parameters: Array1<f64>) -> Result<Array1<f64>> {
        // TODO: This should be an error, not a panic
        assert_eq!(initial_parameters.len(), self.lhs.len());
        let linesearch = MoreThuenteLineSearch::new().with_bounds(0.0, 1.0)?;

        // Define initial parameter vector

        // Set up solver
        let solver = GaussNewtonLS::new(linesearch).with_tolerance(std::f64::EPSILON.sqrt())?;

        // Run solver
        let res = Executor::new(self, solver)
            .configure(|state| state.param(initial_parameters).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        let mut state = res.state().clone();
        let param = state.take_param();
        Ok(param.unwrap())
    }
}

impl<E: Scalar + std::convert::Into<f64>> Problem<E> {
    /// At the moment I don't have time to work out the trait bounds for `argmin` so we need to
    /// cast the generic problem to a concrete type to run any optimisations.
    ///
    /// TODO: Work out the trait bounds and delete this method
    pub(crate) fn into_f64(self) -> Problem<f64> {
        Problem {
            matrix: self.matrix.mapv(std::convert::Into::into),
            lhs: self.lhs.mapv(std::convert::Into::into),
            degree: self.degree,
        }
    }
}

impl<E: Scalar> Problem<E> {
    /// Calculates the polynomial vector of passed parameters
    ///
    /// Accepts a vector of unique parameters of length `n`.
    ///
    /// Outputs a vector of length `n * m` where m is the degree of the underlying polynomial used
    /// for fitting. The elements of the output are those of parameter raised to 1->degree. All
    /// powers of the first element of `params` are listed before those of the second.
    fn promote(&self, params: &Array1<E>) -> Array1<E> {
        let mut promoted = Array1::zeros(params.len() * self.degree);
        for (ii, p) in params.iter().enumerate() {
            for jj in 0..self.degree {
                promoted[ii * self.degree + jj] = p.powi(i32::try_from(jj + 1).unwrap());
            }
        }
        promoted
    }

    /// Compute a vector of length `params`.
    ///
    /// This method promotes `params` (length n) to the full vector of polynomial coeffs (length n
    /// * degree) and left-multiplies with the coefficient matrix
    fn compute(&self, params: &Array1<E>) -> Array1<E> {
        self.matrix.dot(&self.promote(params))
    }

    /// Compute the `ii`th column of the problem Jacobian
    fn jacobian_column(&self, params: &Array1<E>, ii: usize) -> Array1<E> {
        let mut promoted = Array1::zeros(params.len() * self.degree);
        for (jj, p) in params.iter().enumerate() {
            for kk in 0..self.degree {
                if jj == ii {
                    // If we are at jj then we take the derivative
                    promoted[jj * self.degree + kk] =
                        E::from(kk + 1).unwrap() * p.powi(i32::try_from(kk).unwrap());
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
        Ok(self.compute(p) - &self.lhs)
    }
}

impl<E: Scalar> Jacobian for Problem<E> {
    type Param = Array1<E>;
    type Jacobian = Array2<E>;

    fn jacobian(
        &self,
        p: &Self::Param,
    ) -> ::std::result::Result<Self::Jacobian, argmin::core::Error> {
        let mut jacobian = Array2::zeros((p.len(), p.len()));
        for jj in 0..p.len() {
            let col = self.jacobian_column(p, jj);
            jacobian.slice_mut(s![.., jj]).assign(&col);
        }
        Ok(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use super::Problem;

    #[test]
    fn parameter_vectors_are_correctly_promoted_to_polynomial_form() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let num_params = rng.gen_range(5..20);
        let params: Array1<f64> = Array1::from_iter((0..num_params).map(|_| rng.gen()));
        let degree = rng.gen_range(3..10);

        let problem = Problem {
            matrix: Array2::zeros((1, 1)),
            lhs: Array1::zeros(1),
            degree,
        };

        let promoted = problem.promote(&params);

        for (ii, ele) in promoted.into_iter().enumerate() {
            let element_index = ii / degree;
            let expected_power = ii % degree;

            approx::assert_relative_eq!(
                ele,
                params[element_index].powi(i32::try_from(expected_power + 1).unwrap())
            );
        }
    }

    use ndarray_rand::{rand_distr::Uniform, RandomExt};

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

        let problem = Problem {
            matrix,
            lhs: Array1::zeros(1),
            degree,
        };

        let delta_rel = 1e-6;

        for jj in 0..num_params {
            let computed_jacobian_col = problem.jacobian_column(&params, jj);
            let mut modified_params_plus = params.clone();
            let delta = modified_params_plus[jj] * delta_rel;
            modified_params_plus[jj] += delta;
            let mut modified_params_minus = params.clone();
            modified_params_minus[jj] -= delta;

            let computed_at_plus = problem.compute(&modified_params_plus);
            let computed_at_minus = problem.compute(&modified_params_minus);
            let numerical_col = (computed_at_plus - computed_at_minus) / (2. * delta);

            for (comp, num) in computed_jacobian_col.into_iter().zip(numerical_col) {
                approx::assert_relative_eq!(comp, num, max_relative = 1e-4);
            }
        }
    }
}
