use autophagy_macro::instruction;

#[instruction]
pub fn foo(mut x: f64) {
    x += 42.0;

    println!("{x}");
}
