use ndarray::ScalarOperand;
use ndarray::{Array1, Array2};

use ndarray_linalg::Lapack;
use ndarray_linalg::Scalar;

use crate::{
    distributions::{
        DistributionToPower, Measure, Mixture, NormalDistribution, UncorrelatedProduct,
        WeightedDistribution,
    },
    polyfit::Polynomial,
};

#[derive(Clone, Copy, Debug)]
pub struct Measurement<E> {
    pub(crate) value: E,
    pub(crate) uncertainty: E,
}

impl<E: Scalar> Measurement<E> {
    pub(crate) fn from_centroid(value: E) -> Self {
        Self {
            value,
            uncertainty: E::zero(),
        }
    }
}

impl<E: Copy> From<&Measurement<E>> for NormalDistribution<E> {
    fn from(value: &Measurement<E>) -> Self {
        Self {
            mean: value.value,
            standard_deviation: value.uncertainty,
        }
    }
}

impl<E: Lapack + Scalar + ScalarOperand> Measurement<E> {
    pub(crate) fn compute_unknown(&self, fit: &Polynomial<E>) -> Self {
        let measurement: NormalDistribution<E> = NormalDistribution::from(self);
        let coefficient_distributions: Vec<NormalDistribution<E>> = fit
            .to_values()
            .iter()
            .map(NormalDistribution::from)
            .collect();

        let product_distributions =
            coefficient_distributions
                .iter()
                .enumerate()
                .map(|(ii, coeff_dist)| UncorrelatedProduct {
                    a: DistributionToPower {
                        distribution: measurement,
                        power: ii,
                    },
                    b: DistributionToPower {
                        distribution: *coeff_dist,
                        power: 1,
                    },
                });

        let weights = vec![E::one(); product_distributions.len()];
        let mixture: Mixture<E, UncorrelatedProduct<DistributionToPower<NormalDistribution<E>>>> =
            Mixture(
                product_distributions
                    .into_iter()
                    .zip(weights)
                    .map(|(d, w)| WeightedDistribution {
                        distribution: d,
                        weight: w,
                    })
                    .collect(),
            );

        let mean = mixture.expectation();
        let standard_deviation = mixture.variance().sqrt();

        Self {
            value: mean,
            uncertainty: standard_deviation,
        }
    }
}

fn sigma_xy<E: Scalar>(
    measurement: &NormalDistribution<E>,
    coeffs: &[NormalDistribution<E>],
) -> Array1<E> {
    let mut sigma_xy: Array1<E> = Array1::zeros(coeffs.len());

    let product_distributions =
        coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(ii, coeff_dist)| UncorrelatedProduct {
                a: DistributionToPower {
                    distribution: *measurement,
                    power: ii,
                },
                b: DistributionToPower {
                    distribution: *coeff_dist,
                    power: 1,
                },
            });
    let weights = vec![E::one(); coeffs.len()];
    let mixture: Mixture<E, UncorrelatedProduct<DistributionToPower<NormalDistribution<E>>>> =
        Mixture(
            product_distributions
                .into_iter()
                .zip(weights)
                .map(|(d, w)| WeightedDistribution {
                    distribution: d,
                    weight: w,
                })
                .collect(),
        );

    for ii in 0..coeffs.len() {
        sigma_xy[ii] = mixture.covariance(&DistributionToPower {
            distribution: *measurement,
            power: ii + 1,
        });
    }

    sigma_xy
}

fn sigma_x<E: Scalar>(measurement: &NormalDistribution<E>, degree: usize) -> Array2<E> {
    let mut sigma_x: Array2<E> = Array2::zeros((degree, degree));

    for ii in 0..degree {
        let m_ii = DistributionToPower {
            distribution: *measurement,
            power: ii + 1,
        };
        for jj in 0..degree {
            let m_jj = DistributionToPower {
                distribution: *measurement,
                power: jj + 1,
            };
            sigma_x[[ii, jj]] = m_ii.covariance(&m_jj);
        }
    }
    sigma_x
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use itertools::Itertools;
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use crate::distributions::{DistributionToPower, Measure, NormalDistribution};

    use super::sigma_x;


    #[test]
    fn polynomial_covariance_matrix_diagonal_is_generated_correctly() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let degree = rng.gen_range(2..10);

        let distribution: NormalDistribution<f64> = NormalDistribution {
            mean: rng.gen(),
            standard_deviation: rng.gen(),
        };

        let calculated = sigma_x(&distribution, degree);

        // Check the diagonal elements
        for (ii, &calculated) in calculated.diag().into_iter().enumerate() {
            let expected: f64 = DistributionToPower {
                distribution,
                power: ii + 1,
            }
            .variance();
            approx::assert_relative_eq!(calculated, expected);
        }
    }

    #[test]
    fn polynomial_covariance_matrix_diagonal_is_hermitian() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let degree = rng.gen_range(2..10);

        let distribution: NormalDistribution<f64> = NormalDistribution {
            mean: rng.gen(),
            standard_deviation: rng.gen(),
        };

        let calculated = sigma_x(&distribution, degree);

        for (ii, jj) in (0..degree).tuple_combinations().filter(|(ii, jj)| ii != jj) {
            approx::assert_relative_eq!(
                calculated.get((ii, jj)).unwrap(),
                calculated.get((jj, ii)).unwrap()
            );
        }
    }

    #[test]
    fn polynomial_covariance_matrix_upper_triangular_is_correct() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let degree = rng.gen_range(2..10);

        let distribution: NormalDistribution<f64> = NormalDistribution {
            mean: rng.gen(),
            standard_deviation: rng.gen(),
        };

        let calculated = sigma_x(&distribution, degree);

        for (ii, jj) in (0..degree).tuple_combinations().filter(|(ii, jj)| ii != jj) {
            let distribution_ii = DistributionToPower {
                distribution,
                power: ii + 1,
            };
            let distribution_jj = DistributionToPower {
                distribution,
                power: jj + 1,
            };
            let expected = distribution_ii.covariance(&distribution_jj);
            approx::assert_relative_eq!(*calculated.get((ii, jj)).unwrap(), expected);
        }
    }

    #[test]
    fn polynomial_covariance_matrix_upper_triangular_matches_tabulated() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let degree = 5;

        let mean = rng.gen();
        let standard_deviation = rng.gen();

        let distribution: NormalDistribution<f64> = NormalDistribution {
            mean,
            standard_deviation,
        };

        let calculated = sigma_x(&distribution, degree);

        let mut from_mathematica = HashMap::new();
        from_mathematica.insert((0, 1), 2. * mean * standard_deviation.powi(2));
        from_mathematica.insert(
            (0, 2),
            3. * standard_deviation.powi(2)
                * standard_deviation.mul_add(standard_deviation, mean.powi(2)),
        );
        from_mathematica.insert(
            (0, 3),
            4. * mean
                * standard_deviation.powi(2)
                * mean.mul_add(mean, 3. * standard_deviation.powi(2)),
        );
        from_mathematica.insert(
            (0, 4),
            5. * standard_deviation.powi(2)
                * 3.0f64.mul_add(
                    standard_deviation.powi(4),
                    (6. * mean.powi(2)).mul_add(standard_deviation.powi(2), mean.powi(4)),
                ),
        );
        from_mathematica.insert(
            (1, 2),
            6. * mean
                * standard_deviation.powi(2)
                * mean.mul_add(mean, 2. * standard_deviation.powi(2)),
        );
        from_mathematica.insert(
            (1, 3),
            4. * standard_deviation.powi(2)
                * 3.0f64.mul_add(
                    standard_deviation.powi(4),
                    2.0f64.mul_add(mean.powi(4), 9. * mean.powi(2) * standard_deviation.powi(2)),
                ),
        );
        from_mathematica.insert(
            (1, 4),
            10. * mean
                * standard_deviation.powi(2)
                * 9.0f64.mul_add(
                    standard_deviation.powi(4),
                    (8. * mean.powi(2)).mul_add(standard_deviation.powi(2), mean.powi(4)),
                ),
        );
        from_mathematica.insert(
            (2, 3),
            12. * mean
                * standard_deviation.powi(2)
                * 8.0f64.mul_add(
                    standard_deviation.powi(4),
                    (7. * mean.powi(2)).mul_add(standard_deviation.powi(2), mean.powi(4)),
                ),
        );
        from_mathematica.insert(
            (2, 4),
            15. * standard_deviation.powi(2)
                * 7.0f64.mul_add(
                    standard_deviation.powi(6),
                    (25. * mean.powi(2)).mul_add(
                        standard_deviation.powi(4),
                        (11. * mean.powi(4)).mul_add(standard_deviation.powi(2), mean.powi(6)),
                    ),
                ),
        );
        from_mathematica.insert(
            (3, 4),
            20. * mean
                * standard_deviation.powi(2)
                * 45.0f64.mul_add(
                    standard_deviation.powi(6),
                    (57. * mean.powi(2)).mul_add(
                        standard_deviation.powi(4),
                        (15. * mean.powi(4)).mul_add(standard_deviation.powi(2), mean.powi(6)),
                    ),
                ),
        );

        for (ii, jj) in (0..degree).tuple_combinations().filter(|(ii, jj)| ii != jj) {
            let expected = from_mathematica.get(&(ii, jj)).unwrap();
            approx::assert_relative_eq!(
                calculated.get((ii, jj)).unwrap(),
                expected,
                max_relative = 1e-10
            );
        }
    }

    // #[test]
    // fn vector_sigma_xy_matches_tabulated() {
    //     let seed = 40;
    //     let mut rng = Isaac64Rng::seed_from_u64(seed);
    //
    //     let degree = 5;
    //
    //     let mean = rng.gen();
    //     let standard_deviation = rng.gen();
    //
    //
    //     let measurement: NormalDistribution<f64> = NormalDistribution {
    //         mean,
    //         standard_deviation,
    //     };
    //
    //
    //     let coeffs: Vec<_> = (0..degree).map(|_| NormalDistribution { mean: rng.gen(), standard_deviation: rng.gen() }).collect();
    //     let m1 = coeffs[0].mean;
    //     let m2 = coeffs[1].mean;
    //     let m3 = coeffs[2].mean;
    // let m4 = coeffs[3].mean;
    // let m5 = coeffs[4].mean;
    //
    // let from_mathematica = [
    //     standard_deviation.powi(2) * (
    //         m1
    //             + 2. * mean * m2
    //             + 3. * (mean.powi(2) + standard_deviation.powi(2)) * m3
    //             + 4. * mean * (mean.powi(2) + 3. * standard_deviation.powi(2)) * m4
    //             + 5. * (mean.powi(4) + 6. * mean.powi(2) * standard_deviation.powi(2) + 3. * standard_deviation.powi(4)) * m5
    //     )
    // ];
    // let sigma_xy = sigma_xy(&measurement, &coeffs);
    // dbg!(&sigma_xy);
    //
    //
    // for (expected, actual) in from_mathematica.into_iter().zip(sigma_xy) {
    //     approx::assert_relative_eq!(expected, actual);
    // }

    // }
}
