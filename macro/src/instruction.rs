use crate::{attribute_list::AttributeList, utility::parse_crate_path};
use proc_macro::TokenStream;
use proc_macro2::Ident;
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

    let crate_path = parse_crate_path(attributes)?;
    let ident = &function.sig.ident;
    let variable_name = Ident::new(
        &(ident.to_string() + "_instruction"),
        function.sig.ident.span(),
    );

    Ok(quote! {
        pub fn #variable_name() -> #crate_path::Instruction {
            let stream = quote::quote!(#function);
            let function = syn::parse_macro_input!(stream as ItemFn);

            #crate_path::Instruction::new(
                "#ident",
                function,
            )
        }

        #function
    }
    .into())
}
