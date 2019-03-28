pip install wheel
pip wheel $env:APPVEYOR_BUILD_FOLDER

$tests_build_dir = $env:APPVEYOR_BUILD_FOLDER/build
New-Item -ItemType directory -Path $tests_build_dir
Set-Location $tests_build_dir

cmake -A x64 -DCMAKE_BUILD_TYPE=Debug $env:APPVEYOR_BUILD_FOLDER
cmake --build .
$env:BOOST_TEST_LOG_LEVEL = "test_suite"
Debug/test_runner.exe
