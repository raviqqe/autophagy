use crate::{attribute_list::AttributeList, utility::parse_crate_path};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;
use syn::{Expr, ExprLit, FnArg, ItemFn, Lit, LitStr};

const RAW_STRING_PREFIX: &str = "r#";

pub fn generate(
    attributes: &AttributeList,
    function: &ItemFn,
) -> Result<TokenStream, Box<dyn Error>> {
    let crate_path = parse_crate_path(attributes)?;
    let ident = &function.sig.ident;
    let ident_string = ident
        .to_string()
        .strip_prefix(RAW_STRING_PREFIX)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| ident.to_string());
    let visibility = &function.vis;
    let instruction_name = Ident::new(
        &(ident_string.clone() + "_instruction"),
        function.sig.ident.span(),
    );
    let test_module_name = Ident::new(&(ident_string + "_tests"), function.sig.ident.span());
    let name_string = Expr::Lit(ExprLit {
        attrs: Vec::new(),
        lit: Lit::Str(LitStr::new(&ident.to_string(), ident.span())),
    });

    Ok(quote! {
        #visibility fn #instruction_name() -> #crate_path::Instruction {
            #crate_path::Instruction::new(#name_string, syn::parse2(quote::quote!(#function)).unwrap())
        }

        #[cfg(test)]
        mod #test_module_name {
            use super::*;

            #[test]
            fn no_panic() {
                #instruction_name();
            }
        }

        #function
    }
    .into())
}
