use pen_ffi_macro::bindgen;

#[instruction(foo)]
fn foo(mut x: f64) {
    x += 42.0;

    println!("{x}");
}
