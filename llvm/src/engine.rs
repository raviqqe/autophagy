use crate::{compile::compile, LlvmError};
use autophagy::Instruction;
use inkwell::execution_engine::ExecutionEngine;

pub trait Engine {
    fn add_instruction(&self, instruction: &Instruction) -> Result<(), LlvmError>;
}

impl<'c> Engine for ExecutionEngine<'c> {
    fn add_instruction(&self, instruction: &Instruction) -> Result<(), LlvmError> {
        compile(self, instruction)
    }
}
