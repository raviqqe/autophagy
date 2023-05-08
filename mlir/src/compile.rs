use crate::Error;
use autophagy::Instruction;
use melior::ir::Module;

pub fn compile(module: &Module, instruction: &Instruction) -> Result<(), Error> {
    let function = instruction.r#fn();

    function.sig.foo;

    Ok(())
}
