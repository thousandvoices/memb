#!/usr/bin/env bash

install_or_upgrade() {
    if brew ls --versions "$1" >/dev/null; then
        HOMEBREW_NO_AUTO_UPDATE=1 brew upgrade "$1"
    else
        HOMEBREW_NO_AUTO_UPDATE=1 brew install "$1"
    fi
}

REQUIRED_PACKAGES="python3 cmake boost flatbuffers"

for package in "${PREQUIRED_PACKAGES[@]}" ; do
    install_or_upgrade "$package"
done
