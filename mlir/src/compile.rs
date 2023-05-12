use crate::Error;
use autophagy::Instruction;
use melior::ir::Module;

pub fn compile(_module: &Module, instruction: &Instruction) -> Result<(), Error> {
    let _function = instruction.r#fn();

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use autophagy::math;
    use melior::{
        dialect::DialectRegistry,
        ir::Location,
        utility::{register_all_dialects, register_all_llvm_translations},
        Context,
    };

    fn create_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        context
    }

    #[test]
    fn add() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(math::add_instruction()).unwrap();
    }
}
