use core::fmt;
use std::{
    error::Error,
    fmt::{Display, Formatter},
};

#[derive(Debug, Eq, PartialEq)]
pub enum MlirError {
    AddInstruction(String),
}

impl Display for MlirError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddInstruction(name) => {
                write!(formatter, "failed to add instruction: {}", name)
            }
        }
    }
}

impl Error for MlirError {}
