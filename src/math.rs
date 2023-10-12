use ndarray::{Array, Array1, Array2, LinalgScalar};
use ndarray_linalg::Scalar;

use crate::Result;

/// Compute the outer product of two one-dimensional vectors of length (m x 1) and (n x 1)
///
/// The outer product is the (m x n) matrix whose elements are products of elements in the first
/// vector with those in the second.
///
/// # Examples
///
/// ```
/// use error_margin::math::outer_product;
/// use ndarray::{arr1, arr2, Array1};
///
/// let u: Array1<f64> = arr1(&[1., 2., 3.]);
/// let v = arr1(&[4., 5., 6.]);
/// let outer_product = outer_product(&u, &v).unwrap();
///
/// let expected = arr2(&[[4., 5., 6.], [8., 10., 12.], [12., 15., 18.]]);
/// assert_eq!(outer_product, expected);
///```
pub fn outer_product<T: LinalgScalar>(u: &Array1<T>, v: &Array1<T>) -> Result<Array2<T>> {
    let u: Array2<T> = u.clone().into_shape((u.len(), 1))?;
    let v: Array2<T> = v.clone().into_shape((1, v.len()))?;

    Ok(ndarray::linalg::kron(&u, &v))
}

/// Generate the Vandermode matrix of `degree` for observations `x`
///
/// The Vandermonde matrix is a (n + 1 x m + 1) matrix. Each rows of the matrix is a geometric
/// progression for an individual observation variable `x` from power `0` to `degree` inclusive.
///
/// # Panics
///
/// The generator panics in the event that `degree` cannot be converted to `i32`. As the maximum
/// value which can be represented by an `i32` is `2_147_483_647i32` this is unlikely to occur so
/// the error probably does not need to be gracefully handled.
///
/// # Examples
///
/// ```
/// use error_margin::math::vandermonde;
/// use ndarray::arr2;
///
/// let observations: Vec<f64> = vec![2., 3.];
/// let vander = vandermonde(&observations, 2).unwrap();
///
/// let expected = arr2(&[[1., 2., 4.], [1., 3., 9.]]);
/// assert_eq!(vander, expected);
/// ```
pub fn vandermonde<T: Copy + Scalar>(x: &[T], degree: usize) -> Result<Array2<T>> {
    let vals = x.iter().flat_map(|xi| {
        (0..=degree).map(|i| xi.powi(i32::try_from(i).expect("{i} doesn't fit in `i32`")))
    });

    Ok(Array::from_iter(vals).into_shape((x.len(), degree + 1))?)
}

#[cfg(test)]
mod tests {
    use crate::Result;

    use super::outer_product;
    use super::vandermonde;

    use itertools::Itertools;
    use ndarray::Array;
    use ndarray_linalg::Determinant;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::{rand::Rng, RandomExt};
    use rand_isaac::isaac64::Isaac64Rng;

    #[test]
    fn outer_products_are_generated_correctly() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let m = rng.gen::<u8>() as usize;
        let n = rng.gen::<u8>() as usize;
        let u = Array::random_using(m, Uniform::new(0., 10.), &mut rng);
        let v = Array::random_using(n, Uniform::new(0., 10.), &mut rng);

        let outer = outer_product(&u, &v).unwrap();

        for ii in 0..m {
            for jj in 0..n {
                approx::assert_relative_eq!(outer[[ii, jj]], u[ii] * v[jj]);
            }
        }
    }

    #[test]
    fn vandermonde_matrices_are_generated_correctly() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_data_points = 10;
        let degree = 5;

        let data_points = (0..num_data_points)
            .map(|_| rng.gen())
            .collect::<Vec<f64>>();

        let vandermonde = vandermonde(&data_points, degree).unwrap();

        for (ii, data_point) in data_points.iter().enumerate() {
            for jj in 0..=degree {
                let expected = data_point.powi(i32::try_from(jj).unwrap());
                let actual = vandermonde[[ii, jj]];
                approx::assert_relative_eq!(expected, actual);
            }
        }
    }

    #[test]
    fn determinant_of_square_vandermonde_matrix_equals_product_of_differences() -> Result<()> {
        let dim = 5;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let data_points = (0..dim).map(|_| rng.gen()).collect::<Vec<f64>>();

        let vandermonde = vandermonde(&data_points, dim - 1).unwrap();
        let determinant = vandermonde.det()?;

        let product_of_differences: f64 = data_points
            .iter()
            .combinations(2)
            .map(|vals| vals[0] - vals[1])
            .product();

        approx::assert_relative_eq!(determinant, product_of_differences);
        Ok(())
    }
}
