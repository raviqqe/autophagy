use autophagy_macro::instruction;

#[instruction]
pub fn foo(x: usize) -> usize {
    x
}

#[instruction]
pub fn bar(x: usize) -> usize {
    x
}
