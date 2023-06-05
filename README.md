# Autophagy

Yet another AOT compiler for Rust to realize true isomorphism.

## Background

Existing Rust interpreters are mostly wrappers around cargo build or rustc. This incurs extra overhead, such as spawning processes and accessing the file system. Depending on the use case, this approach may not be suitable for compiling and running Rust code in Rust programs.

This crate aims to provide fully in-memory compilation of Rust code into assembly and its execution. This will empower software that requires dynamic code generation.

## Examples

```rust
use autophagy_mlir::Compiler;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_llvm_translations},
    Context, ExecutionEngine,
};

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

let location = Location::unknown(&context);

let mut module = Module::new(location);

Compiler::new(&context, &module).compile_fn(&fibonacci_fn()).unwrap();

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

pass_manager.run(&mut module).unwrap();

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

assert_eq!(result, fibonacci(argument));
```

## Supported features

### Syntax

- [x] Arithmetic/boolean/bit operators
- [x] Dereference operator
- [x] Function calls
- [x] `let` binding
- [x] `=` assignment
  - [x] Variables
  - [x] `struct` fields
  - [ ] Dereferenced pointer
- [x] `while` statement
- [x] `loop` statement
- [x] `if` expression
- [x] `struct` field access
- [ ] Operator assignment (e.g. `+=`)
- [ ] `impl` block and `self` receiver

### Types

- [x] Boolean
- [x] Integers
- [x] Floating-point numbers
- [x] References
- [x] Struct
- [ ] Array

### Literals

- [x] Boolean
- [x] Integers
- [x] Floating-point numbers
- [x] Struct
- [ ] Array

### Others

- [ ] HM type inference

## License

Dual-licensed under [MIT](https://github.com/raviqqe/autophagy/blob/main/LICENSE-MIT) and [Apache 2.0](https://github.com/raviqqe/autophagy/blob/main/LICENSE-APACHE).
