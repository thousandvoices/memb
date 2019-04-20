#pragma once

#include "prefix_code.h"

#include <algorithm>
#include <vector>
#include <string>

namespace memb {

class BitStream {
public:
    BitStream():
        data_(0, 0),
        freeBits_(0)
    {}

    void push(const PrefixCode& prefixCode)
    {
        auto bitsCount = prefixCode.bitsCount;
        while (bitsCount > 0) {
            if (freeBits_ == 0) {
                data_.push_back(0);
                freeBits_ = 8;
            }

            size_t bitsToTake = std::min(bitsCount, freeBits_);
            uint16_t slicedValue = (prefixCode.code & ((1U << bitsCount) - 1)) >> (bitsCount - bitsToTake);
            uint16_t shiftedValue = slicedValue << (freeBits_ - bitsToTake);
            data_.back() += shiftedValue;
            bitsCount -= bitsToTake;
            freeBits_ -= bitsToTake;
        }
    }

    std::vector<uint8_t> data()
    {
        return data_;
    }

private:
    std::vector<uint8_t> data_;
    size_t freeBits_;
};

} // namespace memb
