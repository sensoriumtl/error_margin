use ndarray_linalg::Scalar;

#[derive(Clone, Copy)]
/// A normal distribution
///
/// We assume the *signal* measured in the system is characterised by a log10-normal distribution. This
/// lets us consider the signal (and fitting variables) to be characterised by a Normal
/// distribution.
///
/// The normal distribution is characterised by mean $\mu$, standard deviation $\sigma$
///
/// $$
///     f \left(x\right) = \frac{1}{\sqrt{2 \pi} \sigma} \exp\left[- \frac{1}{2} \frac{x -
///     \mu}{\sigma} \right]
/// $$
pub struct NormalDistribution<T> {
    mean: T,
    standard_deviation: T,
}

impl<T: PartialEq> PartialEq for NormalDistribution<T> {
    fn eq(&self, other: &Self) -> bool {
        self.mean == other.mean && self.standard_deviation == other.standard_deviation
    }
}

impl<T: Copy> NormalDistribution<T> {
    const fn mean(&self) -> T {
        self.mean
    }

    const fn standard_deviation(&self) -> T {
        self.standard_deviation
    }
}

#[derive(Clone, Copy)]
/// We commonly want to compute distributional properties of normal distributions raised to the
/// $n$th power.
pub struct DistributionToPower<D> {
    distribution: D,
    power: usize,
}

trait Moment<T> {
    fn moment(&self, n: usize) -> T;
}

impl<T: Scalar> Moment<T> for NormalDistribution<T> {
    /// The moments of a normal distribution are given by the recurrance relation
    ///
    /// $$
    ///     E[x^n] = \mu E[x^{n - 1}] + (n - 1) \sigma^2 E[x^{n-2}],
    /// $$
    fn moment(&self, n: usize) -> T {
        match n {
            0 => T::one(),
            1 => self.mean(),
            n => {
                self.mean() * self.moment(n - 1)
                    + T::from_usize(n - 1).expect("usize must fit in `T`")
                        * self.standard_deviation().powi(2)
                        * self.moment(n - 2)
            }
        }
    }
}

/// Interface trait to allow for computation of common properties of probability distributions
trait Measure<T: Scalar> {
    /// The variance of a distribution is always $E[x^2] - E[x]^2$
    /// We implement this on the concrete distributions to prevent implementation of a separate
    /// trait for squaring the distribution.
    fn variance(&self) -> T;

    fn expectation(&self) -> T;

    fn covariance(&self, other: Self) -> T;
}

impl<T: Scalar> Measure<T> for DistributionToPower<NormalDistribution<T>> {
    /// The expectation value of a normal distribution raised to `n` is the `n`th moment of the
    /// underlying distribution $E[x^n]$.
    fn expectation(&self) -> T {
        self.distribution.moment(self.power)
    }

    /// The variance of a distribution is $E[x^2] - E[x]^2$
    fn variance(&self) -> T {
        Self {
            distribution: self.distribution,
            power: 2 * self.power,
        }
        .expectation()
            - self.expectation().powi(2)
    }

    fn covariance(&self, other: Self) -> T {
        if self.distribution == other.distribution {
            // If the distributions are equal we calculate as E[xy] - E[x]E[y]
            Self {
                distribution: self.distribution,
                power: other.power + self.power,
            }
            .expectation()
                - self.expectation() * other.expectation()
        } else {
            // If not we assume the two random variables are uncorrelated
            // In this case E[xy] = E[x]E[y] and Cov(x, y) = 0
            T::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray_rand::rand::{Rng, SeedableRng};
    use proptest::prelude::*;
    use rand_isaac::Isaac64Rng;

    use super::{DistributionToPower, Measure, NormalDistribution};

    #[test]
    fn unit_normal_distribution_has_correct_expectation_value() {
        let mean = 0.0;
        let standard_deviation = 1.0;

        let distribution = DistributionToPower {
            distribution: NormalDistribution {
                mean,
                standard_deviation,
            },
            power: 1,
        };

        assert_eq!(distribution.expectation(), mean);
        assert_eq!(distribution.variance(), 1.0);
    }

    proptest! {
        #[test]
        // Odd powers of a unit normal distribution have zero expectation value
        fn odd_powers_of_unit_normal_distribution_have_correct_expectation_value(
            power in (1..20usize)
                .prop_filter("Values must not divisible by 2",
                     |power| !(0 == power % 2))
        ) {
            let mean = 0.0;
            let standard_deviation = 1.0;
            let distribution = DistributionToPower {
                distribution: NormalDistribution {mean, standard_deviation},
                power,
            };

            assert_eq!(distribution.expectation(), 0.0);
        }
    }

    #[test]
    fn even_powers_of_unit_normal_distribution_have_correct_expectation_value() {
        let mean = 0.0;
        let standard_deviation = 1.0;
        let powers = [2, 4];
        let expected_results = [1.0, 3.0, 15.0];

        for (power, expected) in powers.into_iter().zip(expected_results.into_iter()) {
            let distribution = DistributionToPower {
                distribution: NormalDistribution {
                    mean,
                    standard_deviation,
                },
                power,
            };

            assert_eq!(distribution.expectation(), expected, "failed at {power}");
        }
    }

    #[test]
    fn general_normal_distribution_has_correct_expectation_value() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let mean: f64 = rng.gen();
        let standard_deviation = rng.gen();

        let distribution = DistributionToPower {
            distribution: NormalDistribution {
                mean,
                standard_deviation,
            },
            power: 1,
        };

        approx::assert_relative_eq!(distribution.expectation(), mean);
        approx::assert_relative_eq!(distribution.variance(), standard_deviation.powi(2));
    }

    #[test]
    fn powers_of_normal_distribution_have_correct_expectation_value() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let mean: f64 = rng.gen();
        let standard_deviation: f64 = rng.gen();

        let powers = [2, 3, 4, 5, 6];
        let expected_results = [
            mean.powi(2) + standard_deviation.powi(2),
            mean.powi(3) + 3. * mean * standard_deviation.powi(2),
            mean.powi(4)
                + 6. * mean.powi(2) * standard_deviation.powi(2)
                + 3. * standard_deviation.powi(4),
            mean.powi(5)
                + 10. * mean.powi(3) * standard_deviation.powi(2)
                + 15. * mean * standard_deviation.powi(4),
            mean.powi(6)
                + 15. * mean.powi(4) * standard_deviation.powi(2)
                + 45. * mean.powi(2) * standard_deviation.powi(4)
                + 15. * standard_deviation.powi(6),
        ];

        for (power, expected) in powers.into_iter().zip(expected_results.into_iter()) {
            let distribution = DistributionToPower {
                distribution: NormalDistribution {
                    mean,
                    standard_deviation,
                },
                power,
            };
            approx::assert_relative_eq!(distribution.expectation(), expected);
        }
    }
}
