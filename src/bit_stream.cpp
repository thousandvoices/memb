#include "bit_stream.h"

namespace memb {

BitStream::BitStream():
    data_(0, 0),
    freeBits_(0)
{}

void BitStream::push(const PrefixCode& prefixCode)
{
    auto bitsCount = prefixCode.bitsCount;
    while (bitsCount > 0) {
        if (freeBits_ == 0) {
            data_.push_back(0);
            freeBits_ = 8;
        }

        size_t bitsToTake = std::min(bitsCount, freeBits_);
        uint16_t slicedValue = prefixCode.code % (1 << bitsCount) >> (bitsCount - bitsToTake);
        uint16_t shiftedValue = slicedValue << (freeBits_ - bitsToTake);
        data_.back() += shiftedValue;
        bitsCount -= bitsToTake;
        freeBits_ -= bitsToTake;
    }
}

std::vector<uint8_t> BitStream::data()
{
    return data_;
}

BitStreamReader::BitStreamReader(const uint8_t* data, const uint8_t* dataEnd):
    accumulator_(0),
    data_(data),
    dataEnd_(dataEnd),
    extraBits_(0)
{}

uint32_t BitStreamReader::pull(size_t bitsCount)
{
    extraBits_ -= bitsCount;
    while (extraBits_ < 0) {
        accumulator_ <<= 8;
        if (data_ < dataEnd_) {
            accumulator_ += *data_;
            ++data_;
        }
        extraBits_ += 8;
    }

    return accumulator_ >> extraBits_;
}

} // namespace memb
