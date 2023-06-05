use derivative::Derivative;
use syn::ItemStruct;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Struct {
    name: &'static str,
    #[derivative(Debug = "ignore")]
    r#struct: ItemStruct,
}

impl Struct {
    pub const fn new(name: &'static str, r#struct: ItemStruct) -> Self {
        Self { name, r#struct }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn ast(&self) -> &ItemStruct {
        &self.r#struct
    }
}
