#pragma once

#include "huffman_decoder_generated.h"
#include "prefix_code.h"

#include <unordered_map>

namespace memb {

struct CodeInfo {
    uint8_t key;
    size_t length;
};

std::unordered_map<uint8_t, PrefixCode> createCanonicalPrefixCodes(
    const std::vector<CodeInfo>& codeLengths);

class HuffmanTableDecoder {
public:
    HuffmanTableDecoder(
        const std::vector<uint8_t>& keys,
        const std::vector<uint32_t>& sizeOffsets,
        size_t maxDirectDecodeBitLength);

    std::vector<uint8_t> decode(const std::vector<uint8_t>& encodedData, size_t count) const;
    void decode(
        const uint8_t* source,
        size_t sourceSize,
        uint8_t* destination,
        size_t unpackCount) const;

private:
    uint16_t baseOffset(const std::unordered_map<uint8_t, PrefixCode>& codes, uint8_t key);
    
    struct DirectDecodeData {
        uint8_t key;
        uint8_t bitsCount;
    };

    struct IndirectDecodeData {
        size_t offset;
        uint8_t maxBitsCount;
    };

    size_t maxDirectDecodeBitLength_;
    uint32_t decodeTableBitMask_;
    std::vector<DirectDecodeData> decodeTable_;
    std::vector<IndirectDecodeData> indirectOffsetsTable_;
    std::vector<DirectDecodeData> indirectDecodeTable_;
};

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
