#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// #![warn(clippy::cargo)]

pub mod calibration;
pub(crate) mod distributions;
pub(crate) mod margin;
pub(crate) mod math;
pub(crate) mod multi;
pub(crate) mod polyfit;
pub(crate) mod single;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;
