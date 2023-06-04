use crate::Error;
use autophagy::Fn;
use melior::{
    dialect::{arith, func, memref, scf},
    ir::{
        attribute::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, MemRefType},
        Attribute, Block, Identifier, Location, Module, OperationRef, Region, Type, Value,
        ValueLike,
    },
    ContextRef,
};
use std::collections::HashMap;
use train_map::TrainMap;

pub struct Compiler<'c, 'm> {
    module: &'m Module<'c>,
    functions: HashMap<String, FunctionType<'c>>,
}

impl<'c, 'm> Compiler<'c, 'm> {
    pub fn new(module: &'m Module<'c>) -> Self {
        Self {
            module,
            functions: Default::default(),
        }
    }

    fn context(&self) -> ContextRef<'c> {
        self.module.context()
    }

    pub fn compile(&mut self, r#fn: &Fn) -> Result<(), Error> {
        let function = r#fn.ast();
        let context = self.module.context();
        let location = Location::unknown(&context);
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
        let function_type =
            FunctionType::new(unsafe { context.to_ref() }, &argument_types, &result_types);
        self.functions.insert(name.clone(), function_type);

        self.module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, &name),
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
                            &context,
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
                Identifier::new(&context, "llvm.emit_c_interface"),
                Attribute::unit(&context),
            )],
            location,
        ));

        Ok(())
    }

    fn compile_type(&self, r#type: &syn::Type) -> Result<Type<'c>, Error> {
        Ok(match r#type {
            syn::Type::Path(path) => {
                if let Some(identifier) = path.path.get_ident() {
                    self.compile_primitive_type(&identifier.to_string())
                } else {
                    return Err(Error::NotSupported("custom type"));
                }
            }
            _ => todo!(),
        })
    }

    fn compile_primitive_type(&self, name: &str) -> Type<'c> {
        let context = self.context();
        let context = unsafe { context.to_ref() };

        match name {
            "bool" => IntegerType::new(&context, 1).into(),
            "f32" => Type::float32(&context),
            "f64" => Type::float64(&context),
            "isize" | "usize" => Type::index(&context),
            "i8" | "u8" => IntegerType::new(&context, 8).into(),
            "i16" | "u16" => IntegerType::new(&context, 16).into(),
            "i32" | "u32" => IntegerType::new(&context, 32).into(),
            "i64" | "u64" => IntegerType::new(&context, 64).into(),
            _ => todo!(),
        }
    }

    fn compile_block(
        &self,
        block: &syn::Block,
        function_scope: bool,
        variables: &mut TrainMap<String, Value>,
    ) -> Result<Region, Error> {
        let builder = Block::new(&[]);
        let mut variables = variables.fork();

        self.compile_statements(&builder, &block.stmts, function_scope, &mut variables)?;

        let region = Region::new();
        region.append_block(builder);
        Ok(region)
    }

    fn compile_statements<'a>(
        &self,
        builder: &'a Block,
        statements: &[syn::Stmt],
        function_scope: bool,
        variables: &mut TrainMap<String, Value<'a>>,
    ) -> Result<(), Error> {
        let context = self.context();
        let location = Location::unknown(&context);
        let terminator = if function_scope {
            func::r#return
        } else {
            scf::r#yield
        };
        let mut terminated = false;

        for (index, statement) in statements.iter().enumerate() {
            match statement {
                syn::Stmt::Local(local) => self.compile_local_binding(builder, local, variables)?,
                syn::Stmt::Item(_) => return Err(Error::NotSupported("local item definition")),
                syn::Stmt::Expr(expression, semicolon) => {
                    let value = self.compile_expression(builder, expression, variables)?;

                    if index == statements.len() - 1 && semicolon.is_none() {
                        builder.append_operation(terminator(&[value], location));
                        terminated = true;
                    }
                }
                syn::Stmt::Macro(_) => return Err(Error::NotSupported("macro")),
            }
        }

        if !terminated {
            builder.append_operation(terminator(&[], location));
        }

        Ok(())
    }

    fn compile_local_binding<'a>(
        &self,
        builder: &'a Block,
        local: &syn::Local,
        variables: &mut TrainMap<String, Value<'a>>,
    ) -> Result<(), Error> {
        let value = self.compile_expression(
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
                &self.context(),
                MemRefType::new(value.r#type(), &[], None, None),
                &[],
                &[],
                None,
                Location::unknown(&self.context()),
            ))
            .result(0)?
            .into();

        builder.append_operation(memref::store(
            value,
            ptr,
            &[],
            Location::unknown(&self.context()),
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

    fn compile_expression<'a>(
        &self,
        builder: &'a Block,
        expression: &syn::Expr,
        variables: &mut TrainMap<String, Value<'a>>,
    ) -> Result<Value<'a>, Error> {
        let context = &self.context();
        let location = Location::unknown(context);

        Ok(match expression {
            syn::Expr::Assign(assign) => {
                let value = self.compile_expression(builder, &assign.right, variables)?;

                builder.append_operation(memref::store(
                    value,
                    self.compile_ptr(&assign.left, variables)?,
                    &[],
                    location,
                ));

                value
            }
            syn::Expr::Binary(operation) => self
                .compile_binary_operation(builder, operation, variables)?
                .result(0)?
                .into(),
            syn::Expr::Block(block) => builder
                .append_operation(scf::execute_region(
                    // TODO
                    &[Type::index(context)],
                    self.compile_block(&block.block, false, variables)?,
                    location,
                ))
                .result(0)?
                .into(),
            syn::Expr::Call(call) => builder
                .append_operation(func::call_indirect(
                    self.compile_expression(builder, &call.func, variables)?,
                    &call
                        .args
                        .iter()
                        .map(|argument| self.compile_expression(builder, argument, variables))
                        .collect::<Result<Vec<_>, _>>()?,
                    location,
                ))
                .result(0)?
                .into(),
            syn::Expr::If(r#if) => builder
                .append_operation(scf::r#if(
                    self.compile_expression(builder, &r#if.cond, variables)?,
                    // TODO
                    &[Type::index(context)],
                    self.compile_block(&r#if.then_branch, false, variables)?,
                    if let Some((_, expression)) = &r#if.else_branch {
                        let block = Block::new(&[]);
                        let mut variables = variables.fork();

                        block.append_operation(scf::r#yield(
                            &[self.compile_expression(&block, expression, &mut variables)?],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    } else {
                        Region::new()
                    },
                    location,
                ))
                .result(0)?
                .into(),
            syn::Expr::Lit(literal) => self
                .compile_expression_literal(builder, literal)?
                .result(0)?
                .into(),
            syn::Expr::Path(path) => self.compile_path(builder, path, variables)?,
            syn::Expr::Unary(operation) => self
                .compile_unary_operation(builder, operation, variables)?
                .result(0)?
                .into(),
            syn::Expr::While(r#while) => {
                builder.append_operation(scf::r#while(
                    &[],
                    &[],
                    {
                        let block = Block::new(&[]);
                        let mut variables = variables.fork();

                        block.append_operation(scf::condition(
                            self.compile_expression(&block, &r#while.cond, &mut variables)?,
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

                self.compile_unit(builder)?
            }
            _ => todo!("{:?}", expression),
        })
    }

    fn compile_ptr<'a>(
        &self,
        expression: &syn::Expr,
        variables: &mut TrainMap<String, Value<'a>>,
    ) -> Result<Value<'a>, Error> {
        Ok(match expression {
            syn::Expr::Path(path) => self.compile_path_ptr(path, variables)?,
            _ => todo!("{:?}", expression),
        })
    }

    fn compile_unary_operation<'a>(
        &self,
        builder: &'a Block,
        operation: &syn::ExprUnary,
        variables: &mut TrainMap<String, Value<'a>>,
    ) -> Result<OperationRef<'a>, Error> {
        let context = &self.context();
        let location = Location::unknown(context);
        let value = self.compile_expression(builder, &operation.expr, variables)?;

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
        builder: &'a Block,
        operation: &syn::ExprBinary,
        variables: &mut TrainMap<String, Value<'a>>,
    ) -> Result<OperationRef<'a>, Error> {
        let context = &self.context();
        let location = Location::unknown(context);
        let left = self.compile_expression(builder, &operation.left, variables)?;
        let right = self.compile_expression(builder, &operation.right, variables)?;

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
        builder: &'a Block,
        literal: &syn::ExprLit,
    ) -> Result<OperationRef<'a>, Error> {
        let context = &self.context();
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
                        name => self.compile_primitive_type(name),
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
                        name => self.compile_primitive_type(name),
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
        builder: &'a Block,
        path: &syn::ExprPath,
        variables: &TrainMap<String, Value<'a>>,
    ) -> Result<Value<'a>, Error> {
        Ok(builder
            .append_operation(memref::load(
                self.compile_path_ptr(path, variables)?,
                &[],
                Location::unknown(&self.context()),
            ))
            .result(0)?
            .into())
    }

    fn compile_path_ptr<'a>(
        &self,
        path: &syn::ExprPath,
        variables: &TrainMap<String, Value<'a>>,
    ) -> Result<Value<'a>, Error> {
        if let Some(identifier) = path.path.get_ident() {
            let name = identifier.to_string();

            Ok(*variables
                .get(&name)
                .ok_or(Error::VariableNotDefined(name))?)
        } else {
            Err(Error::NotSupported("non-identifier path"))
        }
    }

    // TODO Use a zero-sized type. (LLVM struct?)
    fn compile_unit<'a>(&self, builder: &'a Block) -> Result<Value<'a>, Error> {
        let context = &self.context();

        Ok(builder
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
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

    fn compile(module: &Module, r#fn: &Fn) -> Result<(), Error> {
        Compiler::new(module).compile(r#fn)?;

        Ok(())
    }

    #[test]
    fn add() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::add_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn sub() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::sub_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn mul() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::mul_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn div() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::div_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn rem() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::rem_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn neg() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::neg_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn not() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::not_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn and() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::and_fn()).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn or() {
        let context = create_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        compile(&module, &math::or_fn()).unwrap();

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

            compile(&module, &foo_fn()).unwrap();

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

            compile(&module, &foo_fn()).unwrap();

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

            compile(&module, &foo_fn()).unwrap();

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

        compile(&module, &foo_fn()).unwrap();
        compile(&module, &bar_fn()).unwrap();

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

        compile(&module, &foo_fn()).unwrap();

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

        compile(&module, &foo_fn()).unwrap();

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

        compile(&module, &foo_fn()).unwrap();

        assert!(module.as_operation().verify());
    }
}
