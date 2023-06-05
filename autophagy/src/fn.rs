use derivative::Derivative;
use syn::ItemFn;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fn {
    name: &'static str,
    #[derivative(Debug = "ignore")]
    ast: ItemFn,
}

impl Fn {
    pub const fn new(name: &'static str, ast: ItemFn) -> Self {
        Self { name, ast }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn ast(&self) -> &ItemFn {
        &self.ast
    }
}
