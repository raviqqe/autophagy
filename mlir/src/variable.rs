use melior::ir::Value;

pub struct Variable<'a> {
    value: Value<'a>,
    local: bool,
}

impl<'a> Variable<'a> {
    pub fn new(value: Value<'a>, local: bool) -> Self {
        Self { value, local }
    }

    pub fn value(&self) -> Value<'a> {
        self.value
    }

    pub fn local(&self) -> bool {
        self.local
    }
}
