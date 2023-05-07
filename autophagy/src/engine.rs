use crate::Instruction;

#[derive(Debug, Default)]
pub struct Engine {
    instructions: Vec<Instruction>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }
}
