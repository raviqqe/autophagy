use core::fmt;
use std::{
    error,
    fmt::{Display, Formatter},
};

#[derive(Debug)]
pub enum Error {
    AddInstruction(String),
    Syn(syn::Error),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddInstruction(name) => {
                write!(formatter, "failed to add instruction: {}", name)
            }
            Self::Syn(error) => write!(formatter, "{}", error),
        }
    }
}

impl error::Error for Error {}

impl From<syn::Error> for Error {
    fn from(error: syn::Error) -> Self {
        Self::Syn(error)
    }
}
