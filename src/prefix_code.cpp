#include "prefix_code.h"

namespace memb {

std::unordered_map<uint8_t, PrefixCode> createCanonicalPrefixCodes(
    const std::vector<CodeInfo>& codeLengths)
{
    std::unordered_map<uint8_t, PrefixCode> codebook;
    PrefixCode currentCode{0, 0};

    for (size_t i = 0; i < codeLengths.size(); ++i) {
        while (currentCode.bitsCount < codeLengths[i].length) {
            ++currentCode.bitsCount;
            currentCode.code <<= 1;
        }

        codebook[codeLengths[i].key] = currentCode;
        ++currentCode.code;
    }

    return codebook;
}

}
