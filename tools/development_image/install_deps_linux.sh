#!/usr/bin/env bash

set -e

export BOOST_VERSION="1.67.0"
export FLATBUFFERS_VERSION="1.9.0"

add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt update && apt install -y \
    python3 \
    python3-pip \
    git \
    g++ \
    g++-7 \
    wget \
    cmake

export BOOST_PATH=`echo "boost_${BOOST_VERSION}" | sed 's/\./_/g'`
cd /tmp && wget -q "http://downloads.sourceforge.net/project/boost/boost/${BOOST_VERSION}/${BOOST_PATH}.tar.gz"
tar xfz "${BOOST_PATH}.tar.gz"
rm "${BOOST_PATH}.tar.gz"
cd "${BOOST_PATH}"
./bootstrap.sh --prefix=/usr/local
./b2 cxxflags="-fPIC" install --with-test --with-iostreams > /dev/null
cd /tmp 
rm -rf "${BOOST_PATH}"

cd /tmp && wget -q "https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.tar.gz"
tar xfz "v${FLATBUFFERS_VERSION}.tar.gz"
rm "v${FLATBUFFERS_VERSION}.tar.gz"
mkdir "flatbuffers-${FLATBUFFERS_VERSION}"/build
cd "flatbuffers-${FLATBUFFERS_VERSION}"/build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
make install 
rm -rf "flatbuffers-${FLATBUFFERS_VERSION}"
