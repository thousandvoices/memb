#pragma once

#include "prefix_code.h"
#include "huffman_decoder.h"

namespace memb {

class HuffmanEncoder {
public:
    HuffmanEncoder(const std::unordered_map<uint8_t, size_t>& counts);

    std::vector<uint8_t> encode(const std::vector<uint8_t>& data) const;
    
    HuffmanDecoder createDecoder() const;

private:
    std::unordered_map<uint8_t, PrefixCode> codebook_;
    std::vector<CodeInfo> codeLengths_;
};

class HuffmanEncoderBuilder {
public:
    void updateFrequencies(const std::vector<uint8_t>& data);
    
    HuffmanEncoder createEncoder() const;

private:
    std::unordered_map<uint8_t, size_t> counts_;
};

}
