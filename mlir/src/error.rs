use core::fmt;
use std::{
    error,
    fmt::{Display, Formatter},
};

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    AddInstruction(String),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddInstruction(name) => {
                write!(formatter, "failed to add instruction: {}", name)
            }
        }
    }
}

impl error::Error for Error {}
