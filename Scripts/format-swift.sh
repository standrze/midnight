#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
    swift format format --in-place --configuration .swift-format --recursive \
        Sources Tests Shared/ModelFiles/Sources Optional/Vision/Sources Optional/Vision/Tests Scripts
    swift format format --in-place --configuration .swift-format \
        Package.swift Shared/ModelFiles/Package.swift Optional/Vision/Package.swift
else
    swift format format --in-place --configuration .swift-format "$@"
fi
