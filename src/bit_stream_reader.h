#pragma once

#include <cstdint>

namespace memb {

class BitStreamReader {
public:
    BitStreamReader(const uint8_t* data, const uint8_t* dataEnd):
        accumulator_(0),
        data_(data),
        dataEnd_(dataEnd),
        extraBits_(0)
    {}

    uint64_t pull(size_t bitsCount)
    {
        extraBits_ -= bitsCount;
        while (extraBits_ < 0) {
            for (size_t i = 0; i < 4; ++i) {
                accumulator_ <<= 8;
                if (data_ < dataEnd_) {
                    accumulator_ += *data_;
                    ++data_;
                }
            }
            extraBits_ += 32;
        }

        return accumulator_ >> extraBits_;
    }

private:
    uint64_t accumulator_;
    const uint8_t* data_;
    const uint8_t* dataEnd_;
    int extraBits_;
};


} // namespace memb
