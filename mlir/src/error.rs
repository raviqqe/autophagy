use core::fmt;
use std::{
    error,
    fmt::{Display, Formatter},
};

#[derive(Debug)]
pub enum Error {
    AddInstruction(String),
    Melior(melior::Error),
    Syn(syn::Error),
    VariableNotDefined(String),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddInstruction(name) => {
                write!(formatter, "failed to add instruction: {}", name)
            }
            Self::Melior(error) => write!(formatter, "{}", error),
            Self::Syn(error) => write!(formatter, "{}", error),
            Self::VariableNotDefined(name) => {
                write!(formatter, "variable not defined: {name}")
            }
        }
    }
}

impl error::Error for Error {}

impl From<melior::Error> for Error {
    fn from(error: melior::Error) -> Self {
        Self::Melior(error)
    }
}

impl From<syn::Error> for Error {
    fn from(error: syn::Error) -> Self {
        Self::Syn(error)
    }
}
