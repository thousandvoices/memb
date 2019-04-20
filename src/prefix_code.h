#pragma once

#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace memb {

struct PrefixCode {
    uint16_t code;
    size_t bitsCount;
};

struct CodeInfo {
    uint8_t key;
    size_t length;
};

std::unordered_map<uint8_t, PrefixCode> createCanonicalPrefixCodes(
    const std::vector<CodeInfo>& codeLengths);

}
