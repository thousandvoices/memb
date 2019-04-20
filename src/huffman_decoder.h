#pragma once

#include "huffman_decoder_generated.h"
#include "huffman_table_decoder.h"

#include <vector>

namespace memb {

class HuffmanDecoder {
public:
    HuffmanDecoder(const std::vector<uint8_t>& keys, const std::vector<uint32_t>& offsets);

    HuffmanTableDecoder createTableDecoder(size_t maxDirectDecodeBitLength) const;

    flatbuffers::Offset<wire::HuffmanDecoder> save(flatbuffers::FlatBufferBuilder& builder) const;
    static HuffmanDecoder load(const wire::HuffmanDecoder* serialized);

private:
    std::vector<uint8_t> keys_;
    std::vector<uint32_t> sizeOffsets_;
};

}
