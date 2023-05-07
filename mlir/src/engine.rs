use crate::MlirError;
use autophagy::Instruction;
use melior::ExecutionEngine;

pub struct Engine {
    engine: ExecutionEngine,
}

impl Engine {
    pub fn new(engine: ExecutionEngine) -> Self {
        Self { engine }
    }

    pub fn engine(&self) -> &ExecutionEngine {
        &self.engine
    }

    pub fn add_instruction(&self, _instruction: &Instruction) -> Result<(), MlirError> {
        todo!()
    }

    pub unsafe fn run_instruction(
        &self,
        name: &str,
        arguments_and_result: &mut [*mut ()],
    ) -> Result<(), melior::Error> {
        self.engine.invoke_packed(name, arguments_and_result)
    }
}
