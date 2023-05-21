use core::fmt;
use std::{
    error::Error,
    fmt::{Display, Formatter},
};

#[derive(Debug, Eq, PartialEq)]
pub enum LlvmError {
    AddFn(String),
}

impl Display for LlvmError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddFn(name) => {
                write!(formatter, "failed to add a function: {}", name)
            }
        }
    }
}

impl Error for LlvmError {}
