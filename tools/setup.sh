#!/bin/sh

set -e

llvm_version=19

brew install llvm@$llvm_version zstd

echo PATH=$(brew --prefix llvm@$llvm_version)/bin:$PATH >>$GITHUB_ENV
