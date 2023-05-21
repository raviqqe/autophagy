mod r#fn;
pub mod math;

pub use autophagy_macro::*;
pub use r#fn::Fn;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_sum() {}
}
