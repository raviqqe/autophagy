use derivative::Derivative;
use syn::ItemFn;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Instruction {
    name: &'static str,
    #[derivative(Debug = "ignore")]
    r#fn: ItemFn,
}

impl Instruction {
    pub const fn new(name: &'static str, r#fn: ItemFn) -> Self {
        Self { name, r#fn }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn r#fn(&self) -> &ItemFn {
        &self.r#fn
    }
}
