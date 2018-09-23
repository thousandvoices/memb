#include <boost/test/unit_test.hpp>

#include "bit_stream.h"

using namespace memb;

std::string prettyBitString(uint16_t value, size_t maxBits)
{
    std::string result;
    for (size_t i = 0; i < maxBits; ++i) {
        result.push_back('0' + value % 2);
        value /= 2;
    }
    std::reverse(result.begin(), result.end());

    return result;
}

std::string repr(const std::vector<uint8_t>& data)
{
    std::ostringstream stream;
    for (auto value : data) {
        stream << prettyBitString(value, 8);
    }

    return stream.str();
}

BOOST_AUTO_TEST_SUITE(bitStream)

BOOST_AUTO_TEST_CASE(bitStreamWorks)
{
    BitStream bitStream;

    std::vector<PrefixCode> prefixCodes = {
        {1023, 14},
        {33, 6},
        {0, 4},
        {1234, 11},
        {7, 2}
    };

    for (const auto& code : prefixCodes) {
        bitStream.push(code);
    }

    std::string bitStreamRepr = repr(bitStream.data());

    std::ostringstream stream;
    size_t totalLength = 0;
    for (const auto& code: prefixCodes) {
        stream << prettyBitString(code.code, code.bitsCount);
        totalLength += code.bitsCount;
    }
    stream << std::string(8 - totalLength % 8, '0');
    std::string simpleRepr = stream.str();

    BOOST_CHECK_EQUAL(bitStreamRepr, simpleRepr);
}

BOOST_AUTO_TEST_SUITE_END()
