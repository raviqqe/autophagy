use derivative::Derivative;
use syn::ItemStruct;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Struct {
    name: &'static str,
    #[derivative(Debug = "ignore")]
    ast: ItemStruct,
}

impl Struct {
    pub const fn new(name: &'static str, ast: ItemStruct) -> Self {
        Self { name, ast }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn ast(&self) -> &ItemStruct {
        &self.ast
    }
}
