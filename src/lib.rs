
#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// #![warn(clippy::cargo)]

mod distributions;
pub(crate) mod math;
mod polyfit;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;
