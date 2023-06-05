mod attribute_list;
mod quote;
mod utility;

use self::attribute_list::AttributeList;
use proc_macro::TokenStream;
use std::error::Error;
use syn::{parse_macro_input, Item};

#[proc_macro_attribute]
pub fn quote(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let attributes = parse_macro_input!(attributes as AttributeList);
    let item = parse_macro_input!(item as Item);

    convert_result(quote::generate(&attributes, &item))
}

fn convert_result(result: Result<TokenStream, Box<dyn Error>>) -> TokenStream {
    result.unwrap_or_else(|error| {
        let message = error.to_string();

        ::quote::quote! { compile_error!(#message) }.into()
    })
}
