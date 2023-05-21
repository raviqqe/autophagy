use crate::LlvmError;
use autophagy::Fn;
use inkwell::execution_engine::ExecutionEngine;

pub fn compile(_engine: &ExecutionEngine, _fn: &Fn) -> Result<(), LlvmError> {
    todo!();
}
