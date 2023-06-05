use crate::Error;
use autophagy::{Fn, Struct};
use melior::{
    dialect::{arith, func, llvm, memref, scf},
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute,
            IntegerAttribute, StringAttribute, TypeAttribute,
        },
        r#type::{FunctionType, IntegerType, MemRefType},
        Attribute, Block, Identifier, Location, Module, OperationRef, Region, Type, TypeLike,
        Value, ValueLike,
    },
    Context,
};
use std::collections::HashMap;
use train_map::TrainMap;

struct StructInfo<'c> {
    r#type: Type<'c>,
    field_types: Vec<Type<'c>>,
    field_indices: HashMap<String, usize>,
}

pub struct Compiler<'c, 'm> {
    context: &'c Context,
    module: &'m Module<'c>,
    functions: HashMap<String, FunctionType<'c>>,
    structs: HashMap<String, StructInfo<'c>>,
}

impl<'c, 'm> Compiler<'c, 'm> {
    pub fn new(context: &'c Context, module: &'m Module<'c>) -> Self {
        Self {
            context,
            module,
            functions: Default::default(),
            structs: Default::default(),
        }
    }

    pub fn compile_struct(&mut self, r#struct: &Struct) -> Result<(), Error> {
        let types = r#struct
            .ast()
            .fields
            .iter()
            .map(|field| self.compile_type(&field.ty))
            .collect::<Result<Vec<_>, _>>()?;

        self.structs.insert(
            r#struct.name().into(),
            StructInfo {
                r#type: llvm::r#type::r#struct(self.context, &types, false),
                field_types: types,
                field_indices: r#struct
                    .ast()
                    .fields
                    .iter()
                    .enumerate()
                    .flat_map(|(index, field)| {
                        field.ident.as_ref().map(|ident| (ident.to_string(), index))
                    })
                    .collect(),
            },
        );

        Ok(())
    }

    pub fn compile_fn(&mut self, r#fn: &Fn) -> Result<(), Error> {
        let function = r#fn.ast();
        let context = self.context;
        let location = Location::unknown(context);
        let argument_types = function
            .sig
            .inputs
            .iter()
            .map(|argument| match argument {
                syn::FnArg::Typed(typed) => self.compile_type(&typed.ty),
                syn::FnArg::Receiver(_) => Err(Error::NotSupported("self receiver")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result_types = match &function.sig.output {
            syn::ReturnType::Default => vec![],
            syn::ReturnType::Type(_, r#type) => vec![self.compile_type(r#type)?],
        };
        let mut variables = TrainMap::new();

        let name = function.sig.ident.to_string();
        let function_type = FunctionType::new(context, &argument_types, &result_types);
        self.functions.insert(name.clone(), function_type);

        self.module.body().append_operation(func::func(
            context,
            StringAttribute::new(context, &name),
            TypeAttribute::new(function_type.into()),
            {
                let block = Block::new(
                    &argument_types
                        .iter()
                        .map(|&r#type| (r#type, location))
                        .collect::<Vec<_>>(),
                );

                for result in function.sig.inputs.iter().map(|argument| match argument {
                    syn::FnArg::Typed(typed) => match typed.pat.as_ref() {
                        syn::Pat::Ident(identifier) => {
                            Ok((identifier.ident.to_string(), &typed.ty))
                        }
                        _ => Err(Error::NotSupported("non-identifier pattern")),
                    },
                    syn::FnArg::Receiver(_) => Err(Error::NotSupported("self receiver")),
                }) {
                    let (name, r#type) = result?;

                    let ptr = block
                        .append_operation(memref::alloca(
                            context,
                            MemRefType::new(self.compile_type(r#type)?, &[], None, None),
                            &[],
                            &[],
                            None,
                            location,
                        ))
                        .result(0)?
                        .into();

                    block.append_operation(memref::store(
                        block.argument(0)?.into(),
                        ptr,
                        &[],
                        location,
                    ));

                    variables.insert(name, ptr);
                }

                self.compile_statements(&block, &function.block.stmts, true, &mut variables)?;

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[(
                Identifier::new(context, "llvm.emit_c_interface"),
                Attribute::unit(context),
            )],
            location,
        ));

        Ok(())
    }

    fn compile_type(&self, r#type: &syn::Type) -> Result<Type<'c>, Error> {
        Ok(match r#type {
            syn::Type::Path(path) => {
                if let Some(identifier) = path.path.get_ident() {
                    self.compile_primitive_type(&identifier.to_string())?
                } else {
                    return Err(Error::NotSupported("custom type"));
                }
            }
            syn::Type::Reference(reference) => {
                MemRefType::new(self.compile_type(&reference.elem)?, &[], None, None).into()
            }
            _ => todo!(),
        })
    }

    fn compile_primitive_type(&self, name: &str) -> Result<Type<'c>, Error> {
        let context = self.context;

        Ok(match name {
            "bool" => IntegerType::new(context, 1).into(),
            "f32" => Type::float32(context),
            "f64" => Type::float64(context),
            "isize" | "usize" => Type::index(context),
            "i8" | "u8" => IntegerType::new(context, 8).into(),
            "i16" | "u16" => IntegerType::new(context, 16).into(),
            "i32" | "u32" => IntegerType::new(context, 32).into(),
            "i64" | "u64" => IntegerType::new(context, 64).into(),
            name => {
                self.structs
                    .get(name)
                    .ok_or_else(|| Error::TypeNotDefined(name.into()))?
                    .r#type
            }
        })
    }

    fn compile_block(
        &self,
        block: &syn::Block,
        function_scope: bool,
        variables: &mut TrainMap<String, Value<'c, '_>>,
    ) -> Result<Region<'c>, Error> {
        Ok(self
            .compile_block_expression(block, function_scope, variables)?
            .0)
    }

    fn compile_block_expression(
        &self,
        block: &syn::Block,
        function_scope: bool,
        variables: &mut TrainMap<String, Value<'c, '_>>,
    ) -> Result<(Region<'c>, Option<Type<'c>>), Error> {
        let builder = Block::new(&[]);
        let mut variables = variables.fork();

        let r#type =
            self.compile_statements(&builder, &block.stmts, function_scope, &mut variables)?;

        let region = Region::new();
        region.append_block(builder);
        Ok((region, r#type))
    }

    fn compile_statements<'a>(
        &self,
        builder: &'a Block<'c>,
        statements: &[syn::Stmt],
        function_scope: bool,
        variables: &mut TrainMap<String, Value<'c, 'a>>,
    ) -> Result<Option<Type<'c>>, Error> {
        let context = self.context;
        let location = Location::unknown(context);
        let terminator = if function_scope {
            func::r#return
        } else {
            scf::r#yield
        };
        let mut return_value = None;

        for statement in statements {
            match statement {
                syn::Stmt::Local(local) => self.compile_local_binding(builder, local, variables)?,
                syn::Stmt::Item(_) => return Err(Error::NotSupported("local item definition")),
                syn::Stmt::Expr(expression, semicolon) => {
                    let value = self.compile_expression(builder, expression, variables)?;

                    if semicolon.is_none() {
                        return_value = value;
                    }
                }
                syn::Stmt::Macro(_) => return Err(Error::NotSupported("macro")),
            }
        }

        builder.append_operation(if let Some(value) = return_value {
            terminator(&[value], location)
        } else {
            terminator(&[], location)
        });

        Ok(if function_scope {
            None
        } else {
            return_value.map(|value| value.r#type())
        })
    }

    fn compile_local_binding<'a>(
        &self,
        builder: &'a Block<'c>,
        local: &syn::Local,
        variables: &mut TrainMap<String, Value<'c, 'a>>,
    ) -> Result<(), Error> {
        let context = self.context;

        let value = self.compile_expression_value(
            builder,
            if let Some(initial) = &local.init {
                &initial.expr
            } else {
                return Err(Error::NotSupported("uninitialized let binding"));
            },
            variables,
        )?;
        let ptr = builder
            .append_operation(memref::alloca(
                context,
                MemRefType::new(value.r#type(), &[], None, None),
                &[],
                &[],
                None,
                Location::unknown(context),
            ))
            .result(0)?
            .into();

        builder.append_operation(memref::store(
            value,
            ptr,
            &[],
            Location::unknown(self.context),
        ));

        variables.insert(
            match &local.pat {
                syn::Pat::Ident(identifier) => identifier.ident.to_string(),
                _ => return Err(Error::NotSupported("non-identifier pattern")),
            },
            ptr,
        );

        Ok(())
    }

    fn compile_expression_value<'a>(
        &self,
        builder: &'a Block<'c>,
        expression: &syn::Expr,
        variables: &mut TrainMap<String, Value<'c, 'a>>,
    ) -> Result<Value<'c, 'a>, Error> {
        Ok(
            if let Some(value) = self.compile_expression(builder, expression, variables)? {
                value
            } else {
                self.compile_unit(builder)?
            },
        )
    }

    fn compile_expression<'a>(
        &self,
        builder: &'a Block<'c>,
        expression: &syn::Expr,
        variables: &mut TrainMap<String, Value<'c, 'a>>,
    ) -> Result<Option<Value<'c, 'a>>, Error> {
        let context = self.context;
        let location = Location::unknown(context);

        Ok(match expression {
            syn::Expr::Assign(assign) => {
                let value = self.compile_expression_value(builder, &assign.right, variables)?;

                builder.append_operation(memref::store(
                    value,
                    self.compile_ptr(&assign.left, variables)?,
                    &[],
                    location,
                ));

                Some(value)
            }
            syn::Expr::Binary(operation) => Some(
                self.compile_binary_operation(builder, operation, variables)?
                    .result(0)?
                    .into(),
            ),
            syn::Expr::Block(block) => {
                let (region, r#type) =
                    self.compile_block_expression(&block.block, false, variables)?;

                Some(
                    builder
                        .append_operation(scf::execute_region(
                            &r#type.into_iter().collect::<Vec<_>>(),
                            region,
                            location,
                        ))
                        .result(0)?
                        .into(),
                )
            }
            syn::Expr::Call(call) => {
                let function = self.compile_expression_value(builder, &call.func, variables)?;
                let r#type = FunctionType::try_from(function.r#type())?;

                builder
                    .append_operation(func::call_indirect(
                        function,
                        &call
                            .args
                            .iter()
                            .map(|argument| {
                                self.compile_expression_value(builder, argument, variables)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        &r#type.result(0).into_iter().collect::<Vec<_>>(),
                        location,
                    ))
                    .result(0)
                    .map(Into::into)
                    .ok()
            }
            syn::Expr::Field(field) => {
                let value = self
                    .compile_expression(builder, &field.base, variables)?
                    .ok_or_else(|| {
                        Error::ValueExpected("struct field access requires struct value".into())
                    })?;
                let info = self
                    .structs
                    .values()
                    .find(|info| info.r#type == value.r#type())
                    .ok_or_else(|| Error::StructNotDefined(value.r#type().to_string()))?;
                let index = match &field.member {
                    syn::Member::Named(name) => {
                        *info.field_indices.get(&name.to_string()).ok_or_else(|| {
                            Error::StructFieldNotDefined(
                                value.r#type().to_string(),
                                name.to_string(),
                            )
                        })?
                    }
                    syn::Member::Unnamed(index) => index.index as usize,
                };

                Some(if value.r#type().is_mem_ref() {
                    builder
                        .append_operation(memref::load(
                            builder
                                .append_operation(llvm::get_element_ptr(
                                    context,
                                    value,
                                    DenseI32ArrayAttribute::new(context, &[index as i32]),
                                    value.r#type(),
                                    info.field_types[index],
                                    location,
                                ))
                                .result(0)?
                                .into(),
                            &[],
                            location,
                        ))
                        .result(0)?
                        .into()
                } else {
                    builder
                        .append_operation(llvm::extract_value(
                            context,
                            value,
                            DenseI64ArrayAttribute::new(context, &[index as i64]),
                            info.field_types[index],
                            location,
                        ))
                        .result(0)?
                        .into()
                })
            }
            syn::Expr::If(r#if) => {
                let condition = self.compile_expression_value(builder, &r#if.cond, variables)?;
                let (then_region, then_type) =
                    self.compile_block_expression(&r#if.then_branch, false, variables)?;
                let (else_region, else_type) = if let Some((_, expression)) = &r#if.else_branch {
                    let block = Block::new(&[]);
                    let mut variables = variables.fork();

                    let value = self.compile_expression(&block, expression, &mut variables)?;
                    block.append_operation(scf::r#yield(
                        &value.into_iter().collect::<Vec<_>>(),
                        location,
                    ));

                    let r#type = value.map(|value| value.r#type());
                    let region = Region::new();
                    region.append_block(block);

                    (region, r#type)
                } else {
                    (Region::new(), None)
                };

                builder
                    .append_operation(scf::r#if(
                        condition,
                        &then_type.or(else_type).into_iter().collect::<Vec<_>>(),
                        then_region,
                        else_region,
                        location,
                    ))
                    .result(0)
                    .map(Into::into)
                    .ok()
            }
            syn::Expr::Lit(literal) => self
                .compile_expression_literal(builder, literal)?
                .result(0)
                .map(Into::into)
                .ok(),
            syn::Expr::Loop(r#loop) => {
                builder.append_operation(scf::r#while(
                    &[],
                    &[],
                    {
                        let block = Block::new(&[]);

                        block.append_operation(scf::condition(
                            block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(
                                        true as i64,
                                        IntegerType::new(context, 1).into(),
                                    )
                                    .into(),
                                    location,
                                ))
                                .result(0)?
                                .into(),
                            &[],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    self.compile_block(&r#loop.body, false, variables)?,
                    location,
                ));

                None
            }
            syn::Expr::Paren(parenthesis) => {
                self.compile_expression(builder, &parenthesis.expr, variables)?
            }
            syn::Expr::Path(path) => Some(self.compile_path(builder, path, variables)?),
            syn::Expr::Unary(operation) => self
                .compile_unary_operation(builder, operation, variables)?
                .result(0)
                .map(Into::into)
                .ok(),
            syn::Expr::While(r#while) => {
                builder.append_operation(scf::r#while(
                    &[],
                    &[],
                    {
                        let block = Block::new(&[]);
                        let mut variables = variables.fork();

                        block.append_operation(scf::condition(
                            self.compile_expression_value(&block, &r#while.cond, &mut variables)?,
                            &[],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    self.compile_block(&r#while.body, false, variables)?,
                    location,
                ));

                None
            }
            _ => todo!("{:?}", expression),
        })
    }

    fn compile_ptr<'a>(
        &self,
        expression: &syn::Expr,
        variables: &mut TrainMap<String, Value<'c, 'a>>,
    ) -> Result<Value<'c, 'a>, Error> {
        Ok(match expression {
            syn::Expr::Path(path) => {
                self.compile_variable(&self.convert_path_to_identifier(path)?, variables)?
            }
            _ => todo!("{:?}", expression),
        })
    }

    fn compile_unary_operation<'a>(
        &self,
        builder: &'a Block<'c>,
        operation: &syn::ExprUnary,
        variables: &mut TrainMap<String, Value<'c, 'a>>,
    ) -> Result<OperationRef<'c, 'a>, Error> {
        let context = self.context;
        let location = Location::unknown(context);
        let value = self.compile_expression_value(builder, &operation.expr, variables)?;

        // spell-checker: disable
        Ok(builder.append_operation(match &operation.op {
            syn::UnOp::Deref(_) => memref::load(value, &[], location),
            syn::UnOp::Neg(_) => arith::subi(
                builder
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(0, Type::index(context)).into(),
                        location,
                    ))
                    .result(0)?
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
                    .result(0)?
                    .into(),
                value,
                location,
            ),
            _ => return Err(Error::NotSupported("unknown unary operator")),
        }))
        // spell-checker: enable
    }

    fn compile_binary_operation<'a>(
        &self,
        builder: &'a Block<'c>,
        operation: &syn::ExprBinary,
        variables: &mut TrainMap<String, Value<'c, 'a>>,
    ) -> Result<OperationRef<'c, 'a>, Error> {
        let context = self.context;
        let location = Location::unknown(context);
        let left = self.compile_expression_value(builder, &operation.left, variables)?;
        let right = self.compile_expression_value(builder, &operation.right, variables)?;

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
            syn::BinOp::Eq(_) => {
                arith::cmpi(context, arith::CmpiPredicate::Eq, left, right, location)
            }
            syn::BinOp::Lt(_) => {
                arith::cmpi(context, arith::CmpiPredicate::Slt, left, right, location)
            }
            syn::BinOp::Le(_) => {
                arith::cmpi(context, arith::CmpiPredicate::Sle, left, right, location)
            }
            syn::BinOp::Ne(_) => {
                arith::cmpi(context, arith::CmpiPredicate::Ne, left, right, location)
            }
            syn::BinOp::Ge(_) => {
                arith::cmpi(context, arith::CmpiPredicate::Sge, left, right, location)
            }
            syn::BinOp::Gt(_) => {
                arith::cmpi(context, arith::CmpiPredicate::Sgt, left, right, location)
            }
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
        &self,
        builder: &'a Block<'c>,
        literal: &syn::ExprLit,
    ) -> Result<OperationRef<'c, 'a>, Error> {
        let context = self.context;
        let location = Location::unknown(context);

        Ok(builder.append_operation(match &literal.lit {
            syn::Lit::Bool(boolean) => arith::constant(
                context,
                IntegerAttribute::new(boolean.value as i64, IntegerType::new(context, 1).into())
                    .into(),
                location,
            ),
            syn::Lit::Char(_) => todo!(),
            syn::Lit::Int(integer) => arith::constant(
                context,
                IntegerAttribute::new(
                    integer.base10_parse::<i64>()?,
                    match integer.suffix() {
                        "" => Type::index(context),
                        name => self.compile_primitive_type(name)?,
                    },
                )
                .into(),
                location,
            ),
            syn::Lit::Float(float) => arith::constant(
                context,
                FloatAttribute::new(
                    context,
                    float.base10_parse::<f64>()?,
                    match float.suffix() {
                        "" => Type::index(context),
                        name => self.compile_primitive_type(name)?,
                    },
                )
                .into(),
                location,
            ),
            syn::Lit::Str(_) => todo!(),
            syn::Lit::ByteStr(_) => todo!(),
            syn::Lit::Byte(_) => todo!(),
            _ => todo!(),
        }))
    }

    fn compile_path<'a>(
        &self,
        builder: &'a Block<'c>,
        path: &syn::ExprPath,
        variables: &TrainMap<String, Value<'c, 'a>>,
    ) -> Result<Value<'c, 'a>, Error> {
        let context = self.context;
        let name = self.convert_path_to_identifier(path)?;

        if let Some(&r#type) = self.functions.get(&name) {
            Ok(builder
                .append_operation(func::constant(
                    context,
                    FlatSymbolRefAttribute::new(context, &name),
                    r#type,
                    Location::unknown(context),
                ))
                .result(0)?
                .into())
        } else {
            Ok(builder
                .append_operation(memref::load(
                    self.compile_variable(&name, variables)?,
                    &[],
                    Location::unknown(context),
                ))
                .result(0)?
                .into())
        }
    }

    fn convert_path_to_identifier(&self, path: &syn::ExprPath) -> Result<String, Error> {
        if let Some(identifier) = path.path.get_ident() {
            Ok(identifier.to_string())
        } else {
            Err(Error::NotSupported("non-identifier path"))
        }
    }

    fn compile_variable<'a>(
        &self,
        name: &str,
        variables: &TrainMap<String, Value<'c, 'a>>,
    ) -> Result<Value<'c, 'a>, Error> {
        variables
            .get(name)
            .ok_or_else(|| Error::VariableNotDefined(name.into()))
            .copied()
    }

    fn compile_unit<'a>(&self, builder: &'a Block<'c>) -> Result<Value<'c, 'a>, Error> {
        let context = self.context;

        Ok(builder
            .append_operation(llvm::undef(
                // TODO Should we use zero-field struct instead?
                llvm::r#type::void(context),
                Location::unknown(context),
            ))
            .result(0)?
            .into())
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
        context.attach_diagnostic_handler(|diagnostic| {
            println!("{}", diagnostic);
            true
        });
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        context
    }

    fn compile<'c>(context: &'c Context, module: &Module<'c>, r#fn: &Fn) -> Result<(), Error> {
        Compiler::new(context, module).compile_fn(r#fn)?;

        Ok(())
    }

    #[test]
    fn add() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::add_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn sub() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::sub_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn mul() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::mul_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn div() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::div_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn rem() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::rem_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn neg() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::neg_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn not() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::not_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn and() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::and_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn or() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &math::or_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    mod literal {
        use super::*;

        #[test]
        fn bool() {
            #[allow(dead_code)]
            #[autophagy::quote]
            fn foo() -> bool {
                true
            }

            let context = create_context();

            let location = Location::unknown(&context);
            let module = Module::new(location);

            compile(&context, &module, &foo_fn()).unwrap();

            assert!(module.as_operation().verify());
        }

        #[test]
        fn float32() {
            #[allow(dead_code)]
            #[autophagy::quote]
            fn foo() -> f32 {
                42f32
            }

            let context = create_context();

            let location = Location::unknown(&context);
            let module = Module::new(location);

            compile(&context, &module, &foo_fn()).unwrap();

            assert!(module.as_operation().verify());
        }

        #[test]
        fn float64() {
            #[allow(dead_code)]
            #[autophagy::quote]
            fn foo() -> f64 {
                42f64
            }

            let context = create_context();

            let location = Location::unknown(&context);
            let module = Module::new(location);

            compile(&context, &module, &foo_fn()).unwrap();

            assert!(module.as_operation().verify());
        }
    }

    #[test]
    fn call() {
        #[allow(dead_code)]
        #[autophagy::quote]
        fn foo(x: usize, y: usize) -> usize {
            x + y
        }

        #[allow(dead_code)]
        #[autophagy::quote]
        fn bar() -> usize {
            foo(1, 2)
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let mut compiler = Compiler::new(&context, &module);

        compiler.compile_fn(&foo_fn()).unwrap();
        compiler.compile_fn(&bar_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn dereference() {
        #[allow(dead_code)]
        #[autophagy::quote]
        fn foo(x: &usize) -> usize {
            *x
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn r#if() {
        #[allow(dead_code)]
        #[autophagy::quote]
        fn foo() -> usize {
            if true {
                42usize
            } else {
                13usize
            }
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn r#let() {
        #[allow(dead_code, clippy::let_and_return)]
        #[autophagy::quote]
        fn foo() -> usize {
            let x = 42usize;

            x
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn r#loop() {
        #[allow(dead_code)]
        #[autophagy::quote]
        fn foo() {
            #[allow(clippy::empty_loop)]
            loop {}
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn struct_field() {
        #[autophagy::quote]
        struct Foo {
            bar: i32,
        }

        #[allow(dead_code)]
        #[autophagy::quote]
        fn foo(x: Foo) -> i32 {
            x.bar
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let mut compiler = Compiler::new(&context, &module);

        compiler.compile_struct(&foo_struct()).unwrap();
        compiler.compile_fn(&foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn struct_reference_field() {
        #[autophagy::quote]
        struct Foo {
            bar: i32,
        }

        #[allow(dead_code)]
        #[autophagy::quote]
        fn foo(x: &Foo) -> i32 {
            x.bar
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let mut compiler = Compiler::new(&context, &module);

        compiler.compile_struct(&foo_struct()).unwrap();
        compiler.compile_fn(&foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn r#while() {
        #[allow(dead_code)]
        #[autophagy::quote]
        fn foo() -> usize {
            #[allow(while_true)]
            while true {}

            42usize
        }

        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&context, &module, &foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }
}
