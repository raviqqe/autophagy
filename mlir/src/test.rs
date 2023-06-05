use melior::{
    dialect::DialectRegistry,
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};

pub fn create_test_context() -> Context {
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
