use ndarray::{s, Array, Array0, Array1, Array2, Axis, ScalarOperand};
use ndarray_linalg::{Inverse, Lapack, LeastSquaresSvd, Scalar};
use num_traits::Float;

use std::ops::{MulAssign, Range};

use crate::margin::Measurement;
use crate::math::{outer_product, vandermonde};
use crate::Result;

pub struct Polynomial<E> {
    /// Polynomial coefficients stored in order of ascending power
    coefficients: Vec<E>,
    /// Each coefficient can be associated with a standard deviation
    ///
    /// It is assured in the constructer that, if present, the length of `standard_deviation`
    /// is equal to the length of `coefficients`
    standard_deviation: Option<Vec<E>>,
    /// If the polynomial was generated by a fit, a window can be attached which
    /// contains the range of the underlying data
    window: Option<Range<E>>,
}

impl<E: Scalar> From<FitResult<E>> for Polynomial<E> {
    fn from(value: FitResult<E>) -> Self {
        Self {
            coefficients: value.solution.into_raw_vec(),
            standard_deviation: Some(
                value
                    .covariance
                    .diag()
                    .mapv(ndarray_linalg::Scalar::sqrt)
                    .into_raw_vec(),
            ),
            window: Some(value.window),
        }
    }
}

impl<E> Polynomial<E> {
    pub(crate) fn degree(&self) -> usize {
        self.coefficients.len()
    }
}

impl<E: Scalar> Polynomial<E> {
    pub(crate) fn evaluate_at(&self, value: E) -> E {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(ii, ci)| *ci * value.powi(i32::try_from(ii).unwrap()))
            .sum()
    }

    pub(crate) fn to_values(&self) -> Vec<Measurement<E>> {
        self.standard_deviation.as_ref().map_or_else(
            || {
                self.coefficients
                    .iter()
                    .map(|&value| Measurement {
                        value,
                        uncertainty: E::zero(),
                    })
                    .collect()
            },
            |standard_deviation| {
                self.coefficients
                    .iter()
                    .zip(standard_deviation.iter())
                    .map(|(&value, &uncertainty)| Measurement { value, uncertainty })
                    .collect()
            },
        )
    }
}

#[derive(Clone)]
pub struct FitResult<E: Scalar> {
    solution: Array1<E>,
    covariance: Array2<E>,
    singular_values: Array1<E::Real>,
    rank: i32,
    residual_sum_of_squares: Option<Array0<E::Real>>,
    window: Range<E>,
}

impl<E: Scalar> FitResult<E> {
    pub const fn solution(&self) -> &Array1<E> {
        &self.solution
    }

    pub const fn window(&self) -> &Range<E> {
        &self.window
    }
}

impl<E: PartialOrd + Scalar> FitResult<E> {
    pub fn window_contains(&self, value: &E) -> bool {
        self.window.contains(value)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Scaling {
    Scaled,
    Unscaled,
}

pub fn polyfit<E: Copy + Float + Lapack + MulAssign + PartialOrd + Scalar + ScalarOperand>(
    x: &[E],
    y: &[E],
    degree: usize,
    maybe_weights: Option<&[E]>,
    covariance: Scaling,
) -> Result<FitResult<E>> {
    if x.iter().any(|&ele| !ele.is_finite()) {
        return Err("x-elements contain infinite or NaN values".into());
    }
    if y.iter().any(|&ele| !ele.is_finite()) {
        return Err("y-elements contain infinite or NaN values".into());
    }
    if x.len() != y.len() {
        return Err("x-elements and y-elements must be of equal length".into());
    }

    let vander = vandermonde(x, degree)?;
    let mut lhs: Array2<E> = vander.to_owned();
    let mut rhs: Array1<E> = Array::from_iter(y.iter().copied()).into_shape(x.len())?;
    if let Some(weights) = maybe_weights {
        if weights.iter().any(|&ele| !ele.is_finite()) {
            return Err("weights-elements contain infinite or NaN values".into());
        }
        if x.len() != weights.len() {
            return Err("x-elements and weights must be of equal length".into());
        }

        let weights: Array1<E> = Array::from_iter(weights.iter().copied()).into_shape(x.len())?;
        rhs *= &weights;

        for (ii, weight) in weights.iter().enumerate() {
            let mut slice = lhs.slice_mut(s![ii, ..]);
            slice *= *weight;
        }
    }

    let variance_y = maybe_weights.unwrap().into_iter()
        .map(|w| E::one() / *w)
        .collect::<Array1<_>>();
    let variance_matrix = Array2::from_diag(&variance_y);

    let scaling: Array1<E> = lhs
        .mapv(|val| Scalar::powi(val, 2))
        .sum_axis(Axis(0))
        .mapv(ndarray_linalg::Scalar::sqrt);

    lhs /= &scaling;
    let result = lhs.least_squares(&rhs)?;
    let solution = (&result.solution.t() / &scaling).t().to_owned();

    let covariance_matrix = (lhs.t().dot(&lhs)).inv()?;
    let outer_prod_of_scaling = outer_product(&scaling, &scaling)?;

    let mut covariance_matrix = covariance_matrix / &outer_prod_of_scaling;
    let covariance_matrix_core = lhs.t().dot(&variance_matrix.dot(&lhs)) * outer_prod_of_scaling;
    covariance_matrix = covariance_matrix.dot(&covariance_matrix_core.dot(&covariance_matrix));
    // let mut covariance_matrix = covariance_matrix / &outer_prod_of_scaling;
    // let covariance_matrix_core = lhs.t().dot(&variance_matrix.dot(&lhs)) * outer_prod_of_scaling;
    // covariance_matrix = covariance_matrix.dot(&covariance_matrix_core.dot(&covariance_matrix));
    let covariance = Scaling::Unscaled;
    // if covariance == Scaling::Scaled {
    //     let factor = result
    //         .residual_sum_of_squares
    //         .as_ref()
    //         .unwrap()
    //         .mapv(|re| E::from_real(re) / E::from(x.len() - degree).unwrap());
    //     covariance_matrix = covariance_matrix * factor;
    // };

    // These unwraps are safe because we Error at the start of the function if x contains any NaN
    // or infinite values. This means if `x` contains at least two unique elements we can safely
    // find a minimimum and a maximum.
    let x_min = x
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .to_owned();
    let x_max = x
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .to_owned();

    Ok(FitResult {
        solution,
        covariance: covariance_matrix,
        singular_values: result.singular_values,
        rank: result.rank,
        residual_sum_of_squares: result.residual_sum_of_squares,
        window: x_min..x_max,
    })
}

#[cfg(test)]
mod tests {
    use super::polyfit;
    use super::Scaling;

    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn quadratic_polynomials_are_fit_correctly() {
        let degree = 2;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_samples = rng.gen_range(10..255);
        let coeffs = (0..=degree).map(|_| rng.gen()).collect::<Vec<f64>>();
        let x = (0..num_samples).map(f64::from).collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|x| {
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(ii, ci)| ci * x.powi(i32::try_from(ii).unwrap()))
                    .sum()
            })
            .collect::<Vec<_>>();

        let result = polyfit(&x, &y, degree, None, Scaling::Scaled).unwrap();

        for (coeff, fitted) in coeffs.into_iter().zip(result.solution.into_iter()) {
            approx::assert_relative_eq!(coeff, fitted, max_relative = 1e-10);
        }
    }

    #[test]
    fn cubic_polynomials_are_fit_correctly() {
        let degree = 3;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_samples = rng.gen_range(10..255);
        let coeffs = (0..=degree).map(|_| rng.gen()).collect::<Vec<f64>>();
        let x = (0..num_samples).map(f64::from).collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|x| {
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(ii, ci)| ci * x.powi(i32::try_from(ii).unwrap()))
                    .sum()
            })
            .collect::<Vec<_>>();

        let result = polyfit(&x, &y, degree, None, Scaling::Scaled).unwrap();

        for (coeff, fitted) in coeffs.into_iter().zip(result.solution.into_iter()) {
            approx::assert_relative_eq!(coeff, fitted, max_relative = 1e-10);
        }
    }
}
