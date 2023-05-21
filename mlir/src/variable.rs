use melior::ir::Value;

pub struct Variable<'a> {
    value: Value<'a>,
}

impl<'a> Variable<'a> {
    pub fn new(value: Value<'a>, _local: bool) -> Self {
        Self { value }
    }

    pub fn value(&self) -> Value<'a> {
        self.value
    }
}
