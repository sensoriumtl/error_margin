
#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// #![warn(clippy::cargo)]

pub(crate) mod distributions;
pub(crate) mod math;
pub(crate) mod margin;
pub(crate) mod polyfit;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;
