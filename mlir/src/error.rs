use core::fmt;
use std::{
    error,
    fmt::{Display, Formatter},
};

#[derive(Debug)]
pub enum Error {
    AddFn(String),
    Melior(melior::Error),
    NotSupported(&'static str),
    Syn(syn::Error),
    StructFieldNotDefined(String, String),
    StructNotDefined(String),
    ValueExpected(String),
    VariableNotDefined(String),
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
            Self::StructFieldNotDefined(r#type, field) => {
                write!(
                    formatter,
                    "struct type field \"{}\" not defined: {}",
                    field, r#type
                )
            }
            Self::StructNotDefined(message) => {
                write!(formatter, "struct type not defined: {}", message)
            }
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
