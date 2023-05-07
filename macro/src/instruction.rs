use crate::attribute_list::AttributeList;
use proc_macro::TokenStream;
use quote::quote;
use std::error::Error;
use syn::{FnArg, ItemFn};

pub fn generate(
    attributes: &AttributeList,
    function: &ItemFn,
) -> Result<TokenStream, Box<dyn Error>> {
    if function
        .sig
        .inputs
        .iter()
        .any(|input| matches!(input, FnArg::Receiver(_)))
    {
        return Err("receiver not supported".into());
    } else if function.sig.abi.is_some() {
        return Err("custom function ABI not supported".into());
    } else if !function.sig.generics.params.is_empty() {
        return Err("generic function not supported".into());
    } else if function.sig.asyncness.is_some() {
        return Err("async function not supported".into());
    }

    // TODO Register instructions.
    let _ = attributes.variables();

    Ok(quote! {
        #function
    }
    .into())
}
