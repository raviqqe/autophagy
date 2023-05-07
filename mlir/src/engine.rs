use crate::{compile::compile, MlirError};
use autophagy::Instruction;
use melior::ExecutionEngine;

pub trait Engine {
    fn add_instruction(&self, instruction: &Instruction) -> Result<(), MlirError>;
}

impl Engine for ExecutionEngine {
    fn add_instruction(&self, instruction: &Instruction) -> Result<(), MlirError> {
        compile(self, instruction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::{
        dialect,
        ir::Module,
        pass,
        utility::{register_all_dialects, register_all_llvm_translations},
        Context,
    };

    #[test]
    fn run() {
        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        register_all_llvm_translations(&context);

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        )
        .unwrap();

        let pass_manager = pass::Manager::new(&context);
        pass_manager.add_pass(pass::conversion::convert_func_to_llvm());

        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::convert_arithmetic_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        let engine = ExecutionEngine::new(&module, 2, &[], false);

        let mut argument = 42;
        let mut result = -1;

        assert_eq!(
            unsafe {
                engine.invoke_packed(
                    "add",
                    &mut [
                        &mut argument as *mut i32 as *mut (),
                        &mut result as *mut i32 as *mut (),
                    ],
                )
            },
            Ok(())
        );

        assert_eq!(argument, 42);
        assert_eq!(result, 84);
    }
}