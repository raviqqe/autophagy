use core::fmt;
use std::{
    error,
    fmt::{Display, Formatter},
};

#[derive(Debug)]
pub enum Error {
    AddFn(String),
    Melior(melior::Error),
    Syn(syn::Error),
    ValueExpected(String),
    VariableNotDefined(String),
    NotSupported(&'static str),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddFn(name) => {
                write!(formatter, "failed to add a function: {}", name)
            }
            Self::Melior(error) => write!(formatter, "{}", error),
            Self::NotSupported(name) => {
                write!(formatter, "{name} not supported")
            }
            Self::Syn(error) => write!(formatter, "{}", error),
            Self::ValueExpected(message) => write!(formatter, "value expected: {}", message),
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
