use crate::{compile::compile, LlvmError};
use autophagy::Fn;
use inkwell::execution_engine::ExecutionEngine;

pub trait Engine {
    fn add_fn(&self, r#fn: &Fn) -> Result<(), LlvmError>;
}

impl<'c> Engine for ExecutionEngine<'c> {
    fn add_fn(&self, r#fn: &Fn) -> Result<(), LlvmError> {
        compile(self, r#fn)
    }
}
