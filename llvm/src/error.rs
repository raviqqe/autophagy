use core::fmt;
use std::{
    error::Error,
    fmt::{Display, Formatter},
};

#[derive(Debug, Eq, PartialEq)]
pub enum LlvmError {
    AddInstruction(String),
}

impl Display for LlvmError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddInstruction(name) => {
                write!(formatter, "failed to add instruction: {}", name)
            }
        }
    }
}

impl Error for LlvmError {}
