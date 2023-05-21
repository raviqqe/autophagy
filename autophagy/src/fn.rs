use derivative::Derivative;
use syn::ItemFn;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fn {
    name: &'static str,
    #[derivative(Debug = "ignore")]
    r#fn: ItemFn,
}

impl Fn {
    pub const fn new(name: &'static str, r#fn: ItemFn) -> Self {
        Self { name, r#fn }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn ast(&self) -> &ItemFn {
        &self.r#fn
    }
}
