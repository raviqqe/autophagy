use crate::Error;
use autophagy::Instruction;
use melior::ir::Module;

pub fn compile(_module: &Module, instruction: &Instruction) -> Result<(), Error> {
    let _function = instruction.r#fn();

    todo!()
}
