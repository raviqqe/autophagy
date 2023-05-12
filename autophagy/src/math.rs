#[autophagy_macro::instruction(crate = "crate")]
pub fn add(x: usize, y: usize) -> usize {
    x + y
}

#[autophagy_macro::instruction(crate = "crate")]
pub fn sub(x: usize, y: usize) -> usize {
    x - y
}

#[autophagy_macro::instruction(crate = "crate")]
pub fn mul(x: usize, y: usize) -> usize {
    x * y
}

#[autophagy_macro::instruction(crate = "crate")]
pub fn div(x: usize, y: usize) -> usize {
    x / y
}

#[autophagy_macro::instruction(crate = "crate")]
pub fn rem(x: usize, y: usize) -> usize {
    x % y
}
