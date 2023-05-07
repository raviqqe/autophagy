#[autophagy_macro::instruction(crate = "crate")]
pub fn add(x: usize, y: usize) -> usize {
    x + y
}
