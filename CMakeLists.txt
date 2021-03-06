project(mapped_embeddings)
cmake_minimum_required(VERSION 3.9)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(pybind11)

find_package(flatbuffers REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.67 COMPONENTS iostreams unit_test_framework REQUIRED)

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
endif()

set(FLATBUFFER_SCHEMAS
    src/flatbuffers/kmeans.fbs
    src/flatbuffers/huffman_decoder.fbs
    src/flatbuffers/full_compression.fbs
    src/flatbuffers/uniform_compression.fbs
    src/flatbuffers/trained_compression.fbs
    src/flatbuffers/embeddings.fbs)

set(MEMB_SOURCES
    src/builder.cpp
    src/reader.cpp
    src/compression_strategy.cpp
    src/kmeans.cpp
    src/huffman_encoder.cpp
    src/huffman_decoder.cpp
    src/prefix_code.cpp
    src/trained_compression.cpp
    src/full_compression.cpp
    src/uniform_compression.cpp)

set(MEMB_HEADERS
    src/builder.h
    src/reader.h
    src/bit_stream.h
    src/bit_stream_reader.h
    src/compression_strategy.h
    src/kmeans.h
    src/huffman_encoder.h
    src/huffman_decoder.h
    src/huffman_table_decoder.h
    src/prefix_code.h
    src/trained_compression.h
    src/full_compression.h
    src/uniform_compression.h)

set(TEST_SOURCES
    src/kmeans_tests.cpp
    src/bit_stream_tests.cpp
    src/tests.cpp)

set(BINDING_SOURCES python/memb_bindings.cpp)

flatbuffers_generate_c_headers(FLATBUFFER_GENERATED ${FLATBUFFER_SCHEMAS})
include_directories(${PROJECT_BINARY_DIR}/generated)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${FLATBUFFERS_INCLUDE_DIR})
include_directories(src)
add_custom_target(generate_flatbuffer_headers DEPENDS ${FLATBUFFER_GENERATED_OUTPUTS} ${FLATBUFFER_SCHEMAS})
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/generated)

add_library(memb STATIC ${MEMB_SOURCES} ${MEMB_HEADERS})
add_dependencies(memb generate_flatbuffer_headers)
target_link_libraries(memb ${Boost_LIBRARIES})
if (UNIX AND NOT APPLE)
    target_link_libraries(memb pthread)
endif()

add_executable(test_runner ${TEST_SOURCES})
target_link_libraries(test_runner memb)

pybind11_add_module(_memb ${BINDING_SOURCES})
target_link_libraries(_memb PRIVATE memb)
