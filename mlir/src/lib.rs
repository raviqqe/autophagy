mod compile;
mod error;

pub use compile::compile;
pub use error::Error;

#[cfg(test)]
mod tests {
    use autophagy::quote;
    use melior::{
        dialect::DialectRegistry,
        ir::{Location, Module},
        pass::{self, PassManager},
        utility::{register_all_dialects, register_all_llvm_translations},
        Context, ExecutionEngine,
    };

    fn create_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        register_all_llvm_translations(&context);

        context
    }

    #[quote]
    fn factorial(x: i64) -> i64 {
        let mut y = 42;

        while x != 0 {
            y = y * x;
        }

        x
    }

    #[test]
    fn compile_factorial() {
        let context = create_context();
        let location = Location::unknown(&context);

        let mut module = Module::new(location);

        compile();

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());

        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        let engine = ExecutionEngine::new(&module, 2, &[], false);

        let mut argument = 42;
        let mut result = -1;

        assert_eq!(
            unsafe {
                engine.invoke_packed(
                    "add",
                    &mut [
                        &mut argument as *mut _ as *mut _,
                        &mut result as *mut _ as *mut _,
                    ],
                )
            },
            Ok(())
        );

        assert_eq!(argument, 42);
        assert_eq!(result, 84);
    }
}
