use crate::Error;
use autophagy::Instruction;
use melior::{
    dialect::{func, scf},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
        Block, Location, Module, OperationRef, Region,
    },
    Context,
};

pub fn compile(module: &Module, instruction: &Instruction) -> Result<(), Error> {
    let function = instruction.r#fn();
    let context = module.context();
    let location = Location::unknown(&context);

    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, &function.sig.ident.to_string()),
        TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
        {
            let block = Block::new(&[]);

            compile_block(&context, &block, &function.block)?;

            let region = Region::new();
            region.append_block(block);
            region
        },
        location,
    ));

    Ok(())
}

fn compile_block(context: &Context, builder: &Block, block: &syn::Block) -> Result<(), Error> {
    for statement in &block.stmts {
        compile_statement(&context, builder, statement, true)?;
    }

    Ok(())
}

fn compile_statement(
    context: &Context,
    builder: &Block,
    statement: &syn::Stmt,
    function_scope: bool,
) -> Result<(), Error> {
    let location = Location::unknown(context);

    match statement {
        syn::Stmt::Local(_) => todo!(),
        syn::Stmt::Item(_) => todo!(),
        syn::Stmt::Expr(expression, semicolon) => {
            let operation = compile_expression(context, builder, expression)?;
            let value = operation.result(0).unwrap().into();

            if semicolon.is_none() {
                builder.append_operation(if function_scope {
                    func::r#return(&[value], location)
                } else {
                    scf::r#yield(&[value], location)
                });
            }
        }
        syn::Stmt::Macro(_) => todo!(),
    }

    Ok(())
}

fn compile_expression<'a>(
    context: &Context,
    builder: &'a Block,
    expression: &syn::Expr,
) -> Result<OperationRef<'a>, Error> {
    match expression {
        syn::Expr::Lit(literal) => compile_expression_literal(context, builder, literal),
        _ => todo!(),
    }
}

fn compile_expression_literal<'a>(
    context: &Context,
    builder: &'a Block,
    literal: &syn::ExprLit,
) -> Result<OperationRef<'a>, Error> {
    match &literal.lit {
        syn::Lit::Bool(_) => todo!(),
        syn::Lit::Char(_) => todo!(),
        syn::Lit::Int(_) => todo!(),
        syn::Lit::Float(_) => todo!(),
        syn::Lit::Str(_) => todo!(),
        syn::Lit::ByteStr(_) => todo!(),
        syn::Lit::Byte(_) => todo!(),
        _ => todo!(),
    }
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

        compile(&module, &math::add_instruction()).unwrap();
    }
}
