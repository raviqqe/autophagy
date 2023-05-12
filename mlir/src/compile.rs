use crate::Error;
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
use train_map::TrainMap;

pub fn compile(module: &Module, instruction: &Instruction) -> Result<(), Error> {
    let function = instruction.r#fn();
    let context = &module.context();
    let location = Location::unknown(context);
    let argument_types = function
        .sig
        .inputs
        .iter()
        .map(|argument| match argument {
            syn::FnArg::Typed(typed) => compile_type(context, &typed.ty),
            syn::FnArg::Receiver(_) => Err(Error::NotSupported("self receiver")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result_types = match &function.sig.output {
        syn::ReturnType::Default => vec![],
        syn::ReturnType::Type(_, r#type) => vec![compile_type(context, r#type)?],
    };

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, &function.sig.ident.to_string()),
        TypeAttribute::new(FunctionType::new(context, &argument_types, &result_types).into()),
        {
            let block = Block::new(
                &argument_types
                    .iter()
                    .map(|&r#type| (r#type, location))
                    .collect::<Vec<_>>(),
            );
            let mut variables = TrainMap::new();

            for (index, name) in function
                .sig
                .inputs
                .iter()
                .map(|argument| match argument {
                    syn::FnArg::Typed(typed) => Ok(match typed.pat.as_ref() {
                        syn::Pat::Ident(identifier) => identifier.ident.to_string(),
                        _ => todo!(),
                    }),
                    syn::FnArg::Receiver(_) => Err(Error::NotSupported("self receiver")),
                })
                .enumerate()
            {
                variables.insert(name?, block.argument(index)?.into());
            }

            compile_statements(context, &block, &function.block.stmts, true, &mut variables)?;

            let region = Region::new();
            region.append_block(block);
            region
        },
        location,
    ));

    Ok(())
}

fn compile_type<'c>(context: &'c Context, r#type: &syn::Type) -> Result<Type<'c>, Error> {
    Ok(match r#type {
        syn::Type::Path(path) => {
            if let Some(identifier) = path.path.get_ident() {
                match identifier.to_string().as_str() {
                    "i64" | "u64" => IntegerType::new(context, 64).into(),
                    "isize" | "usize" => Type::index(context),
                    _ => todo!(),
                }
            } else {
                return Err(Error::NotSupported("custom type"));
            }
        }
        _ => todo!(),
    })
}

// TODO Use this.
#[allow(dead_code)]
fn compile_block<'c>(
    context: &'c Context,
    block: &syn::Block,
    function_scope: bool,
    variables: &mut TrainMap<String, Value>,
) -> Result<Block<'c>, Error> {
    let builder = Block::new(&[]);
    let mut variables = variables.fork();

    compile_statements(
        context,
        &builder,
        &block.stmts,
        function_scope,
        &mut variables,
    )?;

    Ok(builder)
}

fn compile_statements<'a>(
    context: &Context,
    builder: &'a Block,
    statements: &[syn::Stmt],
    function_scope: bool,
    variables: &mut TrainMap<String, Value<'a>>,
) -> Result<(), Error> {
    for statement in statements {
        compile_statement(context, builder, statement, function_scope, variables)?;
    }

    Ok(())
}

fn compile_statement<'a>(
    context: &Context,
    builder: &'a Block,
    statement: &syn::Stmt,
    function_scope: bool,
    variables: &mut TrainMap<String, Value<'a>>,
) -> Result<(), Error> {
    let location = Location::unknown(context);

    match statement {
        syn::Stmt::Local(local) => {
            compile_local_binding(context, builder, local, function_scope, variables)?
        }
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
        syn::Stmt::Macro(_) => return Err(Error::NotSupported("macro")),
    }

    Ok(())
}

fn compile_local_binding<'a>(
    context: &Context,
    builder: &'a Block,
    local: &syn::Local,
    function_scope: bool,
    variables: &mut TrainMap<String, Value<'a>>,
) -> Result<(), Error> {
    let location = Location::unknown(context);
    let identifier = match &local.pat {
        syn::Pat::Ident(identifier) => identifier.ident.to_string(),
        _ => return Err(Error::NotSupported("non-identifier pattern")),
    };
    let value = compile_expression(context, builder, variables)?;

    Ok(())
}

fn compile_expression<'a>(
    context: &Context,
    builder: &'a Block,
    expression: &syn::Expr,
    variables: &mut TrainMap<String, Value<'a>>,
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
        syn::Expr::Path(path) => compile_path(path, variables)?,
        syn::Expr::Unary(operation) => {
            compile_unary_operation(context, builder, operation, variables)?
                .result(0)?
                .into()
        }
        _ => todo!(),
    })
}

fn compile_unary_operation<'a>(
    context: &Context,
    builder: &'a Block,
    operation: &syn::ExprUnary,
    variables: &mut TrainMap<String, Value<'a>>,
) -> Result<OperationRef<'a>, Error> {
    let location = Location::unknown(context);
    let value = compile_expression(context, builder, &operation.expr, variables)?;

    // spell-checker: disable
    Ok(builder.append_operation(match &operation.op {
        syn::UnOp::Deref(_) => todo!(),
        syn::UnOp::Neg(_) => arith::subi(
            builder
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(0, Type::index(context)).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
            value,
            location,
        ),
        syn::UnOp::Not(_) => arith::xori(
            builder
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(0, Type::index(context)).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
            value,
            location,
        ),
        _ => return Err(Error::NotSupported("unknown unary operator")),
    }))
    // spell-checker: enable
}

fn compile_binary_operation<'a>(
    context: &Context,
    builder: &'a Block,
    operation: &syn::ExprBinary,
    variables: &mut TrainMap<String, Value<'a>>,
) -> Result<OperationRef<'a>, Error> {
    let location = Location::unknown(context);
    let left = compile_expression(context, builder, &operation.left, variables)?;
    let right = compile_expression(context, builder, &operation.right, variables)?;

    // spell-checker: disable
    Ok(builder.append_operation(match &operation.op {
        syn::BinOp::Add(_) => arith::addi(left, right, location),
        syn::BinOp::Sub(_) => arith::subi(left, right, location),
        syn::BinOp::Mul(_) => arith::muli(left, right, location),
        syn::BinOp::Div(_) => arith::divsi(left, right, location),
        syn::BinOp::Rem(_) => arith::remsi(left, right, location),
        syn::BinOp::And(_) => arith::andi(left, right, location),
        syn::BinOp::Or(_) => arith::ori(left, right, location),
        syn::BinOp::BitXor(_) => arith::xori(left, right, location),
        syn::BinOp::BitAnd(_) => arith::andi(left, right, location),
        syn::BinOp::BitOr(_) => arith::ori(left, right, location),
        syn::BinOp::Shl(_) => arith::shli(left, right, location),
        syn::BinOp::Shr(_) => arith::shrsi(left, right, location),
        syn::BinOp::Eq(_) => arith::cmpi(context, arith::CmpiPredicate::Eq, left, right, location),
        syn::BinOp::Lt(_) => arith::cmpi(context, arith::CmpiPredicate::Slt, left, right, location),
        syn::BinOp::Le(_) => arith::cmpi(context, arith::CmpiPredicate::Sle, left, right, location),
        syn::BinOp::Ne(_) => arith::cmpi(context, arith::CmpiPredicate::Ne, left, right, location),
        syn::BinOp::Ge(_) => arith::cmpi(context, arith::CmpiPredicate::Sge, left, right, location),
        syn::BinOp::Gt(_) => arith::cmpi(context, arith::CmpiPredicate::Sgt, left, right, location),
        syn::BinOp::AddAssign(_) => todo!(),
        syn::BinOp::SubAssign(_) => todo!(),
        syn::BinOp::MulAssign(_) => todo!(),
        syn::BinOp::DivAssign(_) => todo!(),
        syn::BinOp::RemAssign(_) => todo!(),
        syn::BinOp::BitXorAssign(_) => todo!(),
        syn::BinOp::BitAndAssign(_) => todo!(),
        syn::BinOp::BitOrAssign(_) => todo!(),
        syn::BinOp::ShlAssign(_) => todo!(),
        syn::BinOp::ShrAssign(_) => todo!(),
        _ => return Err(Error::NotSupported("unknown binary operator")),
    }))
    // spell-checker: enable
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
                integer.base10_parse::<i64>()?,
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
    path: &syn::ExprPath,
    variables: &TrainMap<String, Value<'a>>,
) -> Result<Value<'a>, Error> {
    Ok(if let Some(identifier) = path.path.get_ident() {
        let name = identifier.to_string();

        *variables
            .get(&name)
            .ok_or(Error::VariableNotDefined(name))?
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

        assert!(module.as_operation().verify());
    }

    #[test]
    fn sub() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::sub_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn mul() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::mul_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn div() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::div_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn rem() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::rem_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn neg() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::neg_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn not() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::not_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn and() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::and_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn or() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::or_instruction()).unwrap();

        assert!(module.as_operation().verify());
    }
}
