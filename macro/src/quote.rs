use crate::{attribute_list::AttributeList, utility::parse_crate_path};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;
use syn::{Expr, ExprLit, ItemFn, Lit, LitStr};

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
    let quote_name = Ident::new(&(ident_string + "_fn"), function.sig.ident.span());
    let name_string = Expr::Lit(ExprLit {
        attrs: Vec::new(),
        lit: Lit::Str(LitStr::new(&ident.to_string(), ident.span())),
    });

    Ok(quote! {
        #visibility fn #quote_name() -> #crate_path::Fn {
            #crate_path::Fn::new(#name_string, syn::parse2(quote::quote!(#function)).unwrap())
        }

        #function
    }
    .into())
}
