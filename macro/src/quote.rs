use crate::{attribute_list::AttributeList, utility::parse_crate_path};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;
use syn::{Expr, ExprLit, Item, ItemFn, ItemStruct, Lit, LitStr};

const RAW_STRING_PREFIX: &str = "r#";

pub fn generate(attributes: &AttributeList, item: &Item) -> Result<TokenStream, Box<dyn Error>> {
    match item {
        Item::Fn(function) => generate_function(attributes, function),
        Item::Struct(r#struct) => generate_struct(attributes, r#struct),
        _ => Err("only functions and structs can be quoted".into()),
    }
}

fn generate_function(
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
    let quote_name = Ident::new(&(ident_string + "_fn"), ident.span());
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

fn generate_struct(
    attributes: &AttributeList,
    r#struct: &ItemStruct,
) -> Result<TokenStream, Box<dyn Error>> {
    let crate_path = parse_crate_path(attributes)?;
    let ident = &r#struct.ident;
    let ident_string = ident
        .to_string()
        .strip_prefix(RAW_STRING_PREFIX)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| ident.to_string());
    let visibility = &r#struct.vis;
    let quote_name = Ident::new(&(ident_string + "_struct"), ident.span());
    let name_string = Expr::Lit(ExprLit {
        attrs: Vec::new(),
        lit: Lit::Str(LitStr::new(&ident.to_string(), ident.span())),
    });

    Ok(quote! {
        #visibility fn #quote_name() -> #crate_path::Struct {
            #crate_path::Struct::new(#name_string, syn::parse2(quote::quote!(#r#struct)).unwrap())
        }

        #r#struct
    }
    .into())
}
