#!/usr/bin/env bash

set -e

TESTS_BUILD_DIR="build"

pip3 wheel .

mkdir -p "${TESTS_BUILD_DIR}"
cd "${TESTS_BUILD_DIR}"
cmake ..
make
BOOST_TEST_LOG_LEVEL="test_suite" ./test_runner
