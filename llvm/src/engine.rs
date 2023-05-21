use crate::{compile::compile, LlvmError};
use autophagy::Function;
use inkwell::execution_engine::ExecutionEngine;

pub trait Engine {
    fn add_instruction(&self, instruction: &Function) -> Result<(), LlvmError>;
}

impl<'c> Engine for ExecutionEngine<'c> {
    fn add_instruction(&self, instruction: &Function) -> Result<(), LlvmError> {
        compile(self, instruction)
    }
}
