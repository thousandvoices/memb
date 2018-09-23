#pragma once

#include "prefix_code.h"

#include <vector>
#include <string>

namespace memb {

class BitStream {
public:
    BitStream();

    void push(const PrefixCode& prefixCode);
    std::vector<uint8_t> data();

private:
    std::vector<uint8_t> data_;
    size_t freeBits_;
};

class BitStreamReader {
public:
    BitStreamReader(const uint8_t* data, const uint8_t* dataEnd);

    uint32_t pull(size_t bitsCount);

private:
    uint32_t accumulator_;
    const uint8_t* data_;
    const uint8_t* dataEnd_;
    int extraBits_;
};

} // namespace memb
