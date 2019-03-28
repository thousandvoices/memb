cmake -A x64 $env:APPVEYOR_BUILD_DIR
cmake --build .
$env:BOOST_TEST_LOG_LEVEL = "test_suite"

pip install wheel
pip wheel $env:APPVEYOR_BUILD_DIR
