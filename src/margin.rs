use ndarray::ScalarOperand;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use ndarray_linalg::Lapack;
use ndarray_linalg::Scalar;

use crate::Result;
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
    pub(crate) fn compute_unknown(&self, fit: &Polynomial<E>) -> Result<Self> {
        // Create the distributions
        let measurement: NormalDistribution<E> = NormalDistribution::from(self);
        let coefficient_distributions: Vec<NormalDistribution<E>> = fit
            .to_values()
            .iter()
            .map(NormalDistribution::from)
            .collect();

        let sigma_xy = sigma_xy(&measurement, &coefficient_distributions);
        let sigma_x = sigma_x(&measurement, fit.degree()) / E::from(1e-10).unwrap();
        let inv_sigma_x = sigma_x.inv()?;

        let variance = sigma_xy.dot(&inv_sigma_x.dot(&sigma_xy));

        let standard_deviation = Scalar::sqrt(variance);

        Ok(Self {
            value: fit.evaluate_at(self.value),
            uncertainty: standard_deviation,
        })
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
