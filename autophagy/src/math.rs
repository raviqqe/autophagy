#[autophagy_macro::quote(crate = "crate")]
pub fn add(x: usize, y: usize) -> usize {
    x + y
}

#[autophagy_macro::quote(crate = "crate")]
pub fn sub(x: usize, y: usize) -> usize {
    x - y
}

#[autophagy_macro::quote(crate = "crate")]
pub fn mul(x: usize, y: usize) -> usize {
    x * y
}

#[autophagy_macro::quote(crate = "crate")]
pub fn div(x: usize, y: usize) -> usize {
    x / y
}

#[autophagy_macro::quote(crate = "crate")]
pub fn rem(x: usize, y: usize) -> usize {
    x % y
}

#[autophagy_macro::quote(crate = "crate")]
pub fn neg(x: isize) -> isize {
    -x
}

#[autophagy_macro::quote(crate = "crate")]
pub fn not(x: usize) -> usize {
    !x
}

#[autophagy_macro::quote(crate = "crate")]
pub fn and(x: usize, y: usize) -> usize {
    x & y
}

#[autophagy_macro::quote(crate = "crate")]
pub fn or(x: usize, y: usize) -> usize {
    x | y
}
