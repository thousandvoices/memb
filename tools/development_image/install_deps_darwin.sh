#!/usr/bin/env bash

set -e

function install_if_missing {
    local package="$1"
    if ! brew ls --versions "$package" >/dev/null; then
        HOMEBREW_NO_AUTO_UPDATE=1 brew install "$1"
    else
        echo "$package is already installed"
    fi
}

REQUIRED_PACKAGES=(python3 cmake boost flatbuffers)

for package in "${REQUIRED_PACKAGES[@]}" ; do
    install_if_missing "$package"
done
