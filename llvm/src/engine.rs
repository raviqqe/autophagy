use crate::{compile::compile, MlirError};
use autophagy::Instruction;
use inkwell::execution_engine::ExecutionEngine;

pub trait Engine {
    fn add_instruction(&self, instruction: &Instruction) -> Result<(), MlirError>;
}

impl<'c> Engine for ExecutionEngine<'c> {
    fn add_instruction(&self, instruction: &Instruction) -> Result<(), MlirError> {
        compile(self, instruction)
    }
}
