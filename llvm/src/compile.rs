use crate::MlirError;
use autophagy::Instruction;
use inkwell::execution_engine::ExecutionEngine;

pub fn compile(_engine: &ExecutionEngine, _instruction: &Instruction) -> Result<(), MlirError> {
    todo!();
}
