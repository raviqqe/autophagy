use syn::ItemFn;

pub struct Instruction {
    name: &'static str,
    r#fn: ItemFn,
}

impl Instruction {
    pub const fn new(name: &'static str, r#fn: ItemFn) -> Self {
        Self { name, r#fn }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn r#fn(&self) -> &ItemFn {
        &self.r#fn
    }
}

#[autophagy_macro::instruction(crate = "crate")]
fn foo() {}
