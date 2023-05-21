use crate::LlvmError;
use autophagy::Fn;
use inkwell::execution_engine::ExecutionEngine;

pub fn compile(_engine: &ExecutionEngine, _instruction: &Fn) -> Result<(), LlvmError> {
    todo!();
}
