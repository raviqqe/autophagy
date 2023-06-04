mod compiler;
mod error;

pub use compiler::Compiler;
pub use error::Error;

#[cfg(test)]
mod tests {
    use super::*;
    use autophagy::Fn;
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
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        context.attach_diagnostic_handler(|diagnostic| {
            eprintln!("{}", diagnostic);
            true
        });

        context
    }

    fn compile<'c>(context: &'c Context, module: &Module<'c>, r#fn: &Fn) -> Result<(), Error> {
        Compiler::new(context, module).compile(r#fn)?;

        Ok(())
    }

    #[allow(clippy::assign_op_pattern)]
    #[autophagy::quote]
    fn factorial(mut x: i32) -> i32 {
        let mut y = 1i32;

        while x > 0i32 {
            y = y * x;
            x = x - 1i32;
        }

        y
    }

    #[test]
    fn compile_factorial() {
        let context = create_context();
        let location = Location::unknown(&context);

        let mut module = Module::new(location);

        compile(&context, &module, &factorial_fn()).unwrap();

        assert!(module.as_operation().verify());

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());

        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_index_to_llvm_pass());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));
        assert!(module.as_operation().verify());

        let engine = ExecutionEngine::new(&module, 2, &[], false);

        let mut argument = 5;
        let mut result = -1;

        assert_eq!(
            unsafe {
                engine.invoke_packed(
                    "factorial",
                    &mut [
                        &mut argument as *mut _ as *mut _,
                        &mut result as *mut _ as *mut _,
                    ],
                )
            },
            Ok(())
        );

        assert_eq!(argument, 5);
        assert_eq!(result, factorial(argument));
    }

    #[allow(clippy::assign_op_pattern)]
    #[autophagy::quote]
    fn fibonacci(x: i32) -> i32 {
        if x <= 0i32 {
            0i32
        } else if x == 1i32 {
            1i32
        } else {
            fibonacci(x - 1i32) + fibonacci(x - 2i32)
        }
    }

    #[test]
    fn compile_fibonacci() {
        let context = create_context();
        let location = Location::unknown(&context);

        let mut module = Module::new(location);

        compile(&context, &module, &fibonacci_fn()).unwrap();

        assert!(module.as_operation().verify());

        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());

        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_index_to_llvm_pass());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));
        assert!(module.as_operation().verify());

        let engine = ExecutionEngine::new(&module, 2, &[], false);

        let mut argument = 5;
        let mut result = -1;

        assert_eq!(
            unsafe {
                engine.invoke_packed(
                    "fibonacci",
                    &mut [
                        &mut argument as *mut _ as *mut _,
                        &mut result as *mut _ as *mut _,
                    ],
                )
            },
            Ok(())
        );

        assert_eq!(argument, 5);
        assert_eq!(result, fibonacci(argument));
    }
}
