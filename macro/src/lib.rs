mod attribute_list;
mod instruction;

use self::attribute_list::AttributeList;
use proc_macro::TokenStream;
use quote::quote;
use std::error::Error;
use syn::{parse_macro_input, ItemFn, ItemStruct};

#[proc_macro_attribute]
pub fn instruction(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let attributes = parse_macro_input!(attributes as AttributeList);
    let function = parse_macro_input!(item as ItemFn);

    convert_result(bindgen::generate(&attributes, &function))
}
