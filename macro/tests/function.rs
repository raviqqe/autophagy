use autophagy_macro::quote;

#[quote]
pub fn foo() {}

#[quote]
pub fn bar(x: usize) -> usize {
    x
}
