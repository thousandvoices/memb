project(mapped_embeddings)
cmake_minimum_required(VERSION 3.9)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

add_subdirectory(pybind11)

find_package(flatbuffers REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.64 COMPONENTS iostreams unit_test_framework REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++14 -fPIC -Werror")

set(FLATBUFFER_SCHEMAS src/flatbuffers/embeddings.fbs)
set(MEMB_SOURCES src/builder.cpp src/reader.cpp src/quantization.cpp)
set(MEMB_HEADERS src/builder.h src/reader.h src/quantization.h)
set(TEST_SOURCES src/tests.cpp)
set(BINDING_SOURCES python/memb_bindings.cpp)

flatbuffers_generate_c_headers(FLATBUFFER_GENERATED ${FLATBUFFER_SCHEMAS})
include_directories(${PROJECT_BINARY_DIR}/generated)
include_directories(${Boost_INCLUDE_DIR})
include_directories(src)
add_custom_target(generate_flatbuffer_headers DEPENDS ${FLATBUFFER_GENERATED_OUTPUTS} ${FLATBUFFER_SCHEMAS})
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/generated)

add_library(memb STATIC ${MEMB_SOURCES} ${MEMB_HEADERS})
add_dependencies(memb generate_flatbuffer_headers)
target_link_libraries(memb ${Boost_LIBRARIES})

add_executable(test_runner ${TEST_SOURCES})
target_link_libraries(test_runner memb)

pybind11_add_module(_memb ${BINDING_SOURCES})
target_link_libraries(_memb PRIVATE memb)
