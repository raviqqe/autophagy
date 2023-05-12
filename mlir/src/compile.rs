use crate::{chain_map::ChainMap, Error};
use autophagy::Instruction;
use melior::{
    dialect::{arith, func, scf},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, Location, Module, OperationRef, Region, Type, Value,
    },
    Context,
};

pub fn compile(module: &Module, instruction: &Instruction) -> Result<(), Error> {
    let function = instruction.r#fn();
    let context = &module.context();
    let location = Location::unknown(&context);
    let mut variables = ChainMap::new();

    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, &function.sig.ident.to_string()),
        TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
        {
            let region = Region::new();
            region.append_block(compile_block(
                &context,
                &function.block,
                true,
                &mut variables,
            )?);
            region
        },
        location,
    ));

    Ok(())
}

fn compile_block<'c>(
    context: &'c Context,
    block: &syn::Block,
    function_scope: bool,
    variables: &mut ChainMap<String, Value>,
) -> Result<Block<'c>, Error> {
    let builder = Block::new(&[]);
    let mut variables = variables.fork();

    for statement in &block.stmts {
        compile_statement(
            &context,
            &builder,
            statement,
            function_scope,
            &mut variables,
        )?;
    }

    Ok(builder)
}

fn compile_statement<'a>(
    context: &Context,
    builder: &'a Block,
    statement: &syn::Stmt,
    function_scope: bool,
    variables: &mut ChainMap<String, Value<'a>>,
) -> Result<(), Error> {
    let location = Location::unknown(context);

    match statement {
        syn::Stmt::Local(_) => todo!(),
        syn::Stmt::Item(_) => todo!(),
        syn::Stmt::Expr(expression, semicolon) => {
            let value = compile_expression(context, builder, expression, variables)?;

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
    variables: &mut ChainMap<String, Value<'a>>,
) -> Result<Value<'a>, Error> {
    Ok(match expression {
        syn::Expr::Binary(operation) => {
            compile_binary_operation(context, builder, operation, variables)?
                .result(0)?
                .into()
        }
        syn::Expr::Lit(literal) => compile_expression_literal(context, builder, literal)?
            .result(0)?
            .into(),
        syn::Expr::Path(path) => compile_path(context, builder, path, variables)?,
        _ => todo!(),
    })
}

fn compile_binary_operation<'a>(
    context: &Context,
    builder: &'a Block,
    operation: &syn::ExprBinary,
    variables: &mut ChainMap<String, Value<'a>>,
) -> Result<OperationRef<'a>, Error> {
    let location = Location::unknown(context);
    let left = compile_expression(&context, builder, &operation.left, variables)?;
    let right = compile_expression(&context, builder, &operation.right, variables)?;

    Ok(builder.append_operation(match &operation.op {
        syn::BinOp::Add(_) => arith::addi(left, right, location),
        _ => todo!(),
    }))
}

fn compile_expression_literal<'a>(
    context: &Context,
    builder: &'a Block,
    literal: &syn::ExprLit,
) -> Result<OperationRef<'a>, Error> {
    let location = Location::unknown(context);

    Ok(builder.append_operation(match &literal.lit {
        syn::Lit::Bool(boolean) => arith::constant(
            context,
            IntegerAttribute::new(boolean.value as i64, IntegerType::new(context, 1).into()).into(),
            location,
        ),
        syn::Lit::Char(_) => todo!(),
        syn::Lit::Int(integer) => arith::constant(
            context,
            IntegerAttribute::new(
                integer.base10_parse::<i64>()? as i64,
                match integer.suffix() {
                    "" => Type::index(context),
                    "i8" | "u8" => IntegerType::new(context, 8).into(),
                    "i16" | "u16" => IntegerType::new(context, 16).into(),
                    "i32" | "u32" => IntegerType::new(context, 32).into(),
                    "i64" | "u64" => IntegerType::new(context, 64).into(),
                    _ => todo!(),
                },
            )
            .into(),
            location,
        ),
        syn::Lit::Float(_) => todo!(),
        syn::Lit::Str(_) => todo!(),
        syn::Lit::ByteStr(_) => todo!(),
        syn::Lit::Byte(_) => todo!(),
        _ => todo!(),
    }))
}

fn compile_path<'a>(
    context: &Context,
    builder: &Block,
    path: &syn::ExprPath,
    variables: &ChainMap<String, Value<'a>>,
) -> Result<Value<'a>, Error> {
    let location = Location::unknown(context);

    Ok(if let Some(identifier) = path.path.get_ident() {
        let name = identifier.to_string();

        *variables
            .get(&name)
            .ok_or_else(|| Error::VariableNotDefined(name))?
    } else {
        todo!()
    })
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
