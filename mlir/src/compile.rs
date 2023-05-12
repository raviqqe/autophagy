use std::collections::HashMap;

use crate::Error;
use autophagy::Instruction;
use melior::{
    dialect::{arith, func, scf},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, Location, Module, OperationRef, Region, Type, Value,
    },
};

struct Context<'c> {
    melior: &'c melior::Context,
    variables: HashMap<String, Value<'c>>,
}

impl<'c> Context<'c> {
    pub fn new(melior: &'c melior::Context) -> Self {
        Self {
            melior,
            variables: Default::default(),
        }
    }

    pub fn melior(&self) -> &melior::Context {
        self.melior
    }

    pub fn variable(&self, name: &str) -> Result<Value<'c>, Error> {
        self.variables
            .get(name)
            .copied()
            .ok_or_else(|| Error::VariableNotDefined(name.into()))
    }

    pub fn insert_variable(&mut self, name: &str, value: Value<'c>) {
        self.variables.insert(name.to_string(), value);
    }
}

pub fn compile(module: &Module, instruction: &Instruction) -> Result<(), Error> {
    let function = instruction.r#fn();
    let context = module.context();
    let context = Context::new(&context);
    let location = Location::unknown(context.melior());

    module.body().append_operation(func::func(
        context.melior(),
        StringAttribute::new(context.melior(), &function.sig.ident.to_string()),
        TypeAttribute::new(FunctionType::new(context.melior(), &[], &[]).into()),
        {
            let block = Block::new(&[]);

            compile_block(&context, &block, &function.block, true)?;

            let region = Region::new();
            region.append_block(block);
            region
        },
        location,
    ));

    Ok(())
}

fn compile_block(
    context: &Context,
    builder: &Block,
    block: &syn::Block,
    function_scope: bool,
) -> Result<(), Error> {
    for statement in &block.stmts {
        compile_statement(&context, builder, statement, function_scope)?;
    }

    Ok(())
}

fn compile_statement(
    context: &Context,
    builder: &Block,
    statement: &syn::Stmt,
    function_scope: bool,
) -> Result<(), Error> {
    let location = Location::unknown(context.melior());

    match statement {
        syn::Stmt::Local(_) => todo!(),
        syn::Stmt::Item(_) => todo!(),
        syn::Stmt::Expr(expression, semicolon) => {
            let value = compile_expression(context, builder, expression)?;

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
) -> Result<Value<'a>, Error> {
    Ok(match expression {
        syn::Expr::Binary(operation) => compile_binary_operation(context, builder, operation)?
            .result(0)?
            .into(),
        syn::Expr::Lit(literal) => compile_expression_literal(context, builder, literal)?
            .result(0)?
            .into(),
        syn::Expr::Path(path) => compile_path(context, builder, path)?,
        _ => todo!(),
    })
}

fn compile_binary_operation<'a>(
    context: &Context,
    builder: &'a Block,
    operation: &syn::ExprBinary,
) -> Result<OperationRef<'a>, Error> {
    let location = Location::unknown(context.melior());
    let left = compile_expression(&context, builder, &operation.left)?;
    let right = compile_expression(&context, builder, &operation.right)?;

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
    let location = Location::unknown(context.melior());

    Ok(builder.append_operation(match &literal.lit {
        syn::Lit::Bool(boolean) => arith::constant(
            context.melior(),
            IntegerAttribute::new(
                boolean.value as i64,
                IntegerType::new(context.melior(), 1).into(),
            )
            .into(),
            location,
        ),
        syn::Lit::Char(_) => todo!(),
        syn::Lit::Int(integer) => arith::constant(
            context.melior(),
            IntegerAttribute::new(
                integer.base10_parse::<i64>()? as i64,
                match integer.suffix() {
                    "" => Type::index(context.melior()),
                    "i8" | "u8" => IntegerType::new(context.melior(), 8).into(),
                    "i16" | "u16" => IntegerType::new(context.melior(), 16).into(),
                    "i32" | "u32" => IntegerType::new(context.melior(), 32).into(),
                    "i64" | "u64" => IntegerType::new(context.melior(), 64).into(),
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

fn compile_path<'c>(
    context: &Context<'c>,
    builder: &Block,
    path: &syn::ExprPath,
) -> Result<Value<'c>, Error> {
    let location = Location::unknown(context.melior());

    Ok(if let Some(identifier) = path.path.get_ident() {
        context.variable(&identifier.to_string())?
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
