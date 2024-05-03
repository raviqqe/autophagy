#!/bin/sh

set -e

llvm_version=18

brew install llvm@$llvm_version

echo PATH=$(brew --prefix llvm@$llvm_version)/bin:$PATH >>$GITHUB_ENV
