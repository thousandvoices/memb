#include "huffman_decoder.h"
#include "bit_stream.h"

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

HuffmanTableDecoder::HuffmanTableDecoder(
        const std::vector<uint8_t>& keys,
        const std::vector<uint32_t>& sizeOffsets,
        size_t maxDirectDecodeBitLength):
    maxDirectDecodeBitLength_(maxDirectDecodeBitLength),
    decodeTableBitMask_((1U << maxDirectDecodeBitLength_) - 1)
{
    size_t decodeTableSize = 1U << maxDirectDecodeBitLength_;
    decodeTable_.reserve(decodeTableSize);

    std::vector<CodeInfo> codeLengths;
    codeLengths.reserve(keys.size());

    size_t currentSize = 0;
    for (size_t keyIndex = 0; keyIndex < keys.size(); ++keyIndex) {
        while (keyIndex >= sizeOffsets[currentSize]) {
            ++currentSize;
        }
        codeLengths.push_back({keys[keyIndex], currentSize});
    }

    auto codes = createCanonicalPrefixCodes(codeLengths);
    size_t directTableSize = (sizeOffsets.size() > maxDirectDecodeBitLength_)
        ? sizeOffsets[maxDirectDecodeBitLength_]
        : keys.size();

    for (size_t keyIndex = 0; keyIndex < directTableSize; ++keyIndex) {
        auto key = keys[keyIndex];
        auto currentSize = codes[key].bitsCount;

        DirectDecodeData tableEntry{
            key,
            static_cast<uint8_t>(currentSize)};

        auto decodeTableRepeats = 1U << (maxDirectDecodeBitLength_ - currentSize);
        for (size_t repeat = 0; repeat < decodeTableRepeats; ++repeat) {
            decodeTable_.push_back(tableEntry);
        }
    }

    if (directTableSize < keys.size()) {
        auto minIndirectKey = keys[directTableSize];
        auto minIndirectOffset = baseOffset(codes, minIndirectKey);

        std::vector<size_t> maxBits(decodeTableSize - minIndirectOffset, 0);
        for (size_t keyIndex = directTableSize; keyIndex < keys.size(); ++keyIndex) {
            auto key = keys[keyIndex];

            auto indirectTableOffset = baseOffset(codes, key) - minIndirectOffset;
            maxBits[indirectTableOffset] = std::max(
                maxBits[indirectTableOffset], codes[key].bitsCount);
        }

        uint16_t previousBaseOffset = std::numeric_limits<uint16_t>::max();
        for (size_t keyIndex = directTableSize; keyIndex < keys.size(); ++keyIndex) {
            auto key = keys[keyIndex];
            auto currentCode = codes[key];
            auto currentBaseOffset = baseOffset(codes, key);

            auto currentMaxBits = maxBits[currentBaseOffset - minIndirectOffset];
            if (currentBaseOffset != previousBaseOffset) {
                previousBaseOffset = currentBaseOffset;
                indirectOffsetsTable_.push_back({
                    indirectDecodeTable_.size(),
                    static_cast<uint8_t>(currentMaxBits - maxDirectDecodeBitLength_)});
            }

            DirectDecodeData tableEntry{
                key,
                static_cast<uint8_t>(currentCode.bitsCount - maxDirectDecodeBitLength_)};

            for (size_t repeat = 0; repeat < 1U << (currentMaxBits - currentCode.bitsCount); ++repeat) {
                indirectDecodeTable_.push_back(tableEntry);
            }
        }
    }
}

void HuffmanTableDecoder::decode(
    const uint8_t* source,
    size_t sourceSize,
    uint8_t* destination,
    size_t unpackCount) const
{
    BitStreamReader reader(source, source + sourceSize);

    size_t bitsToPull = maxDirectDecodeBitLength_;
    for (size_t i = 0; i < unpackCount; ++i) {
        uint8_t unpackedValue;
        size_t offset = reader.pull(bitsToPull) & decodeTableBitMask_;

        if (offset < decodeTable_.size()) {
            auto entry = decodeTable_[offset];
            unpackedValue = entry.key;
            bitsToPull = entry.bitsCount;
        } else {
            auto indirectEntry = indirectOffsetsTable_[offset - decodeTable_.size()];
            size_t bitMask = (1U << indirectEntry.maxBitsCount) - 1;
            auto indirectKey = reader.pull(indirectEntry.maxBitsCount) & bitMask;
            auto entry = indirectDecodeTable_[indirectEntry.offset + indirectKey];
            unpackedValue = entry.key;
            bitsToPull = maxDirectDecodeBitLength_ - indirectEntry.maxBitsCount + entry.bitsCount;
        }
        *(destination + i) = unpackedValue;
    }
}

uint16_t HuffmanTableDecoder::baseOffset(const std::unordered_map<uint8_t, PrefixCode>& codes, uint8_t key)
{
    auto code = codes.at(key);
    return code.code >> (code.bitsCount - maxDirectDecodeBitLength_);
}

HuffmanDecoder::HuffmanDecoder(const std::vector<uint8_t>& keys, const std::vector<uint32_t>& offsets):
    keys_(keys),
    sizeOffsets_(offsets)
{
}

HuffmanTableDecoder HuffmanDecoder::createTableDecoder(
    size_t maxDirectDecodeBitLength) const
{
    return HuffmanTableDecoder(keys_, sizeOffsets_, maxDirectDecodeBitLength);
}

flatbuffers::Offset<wire::HuffmanDecoder> HuffmanDecoder::save(
    flatbuffers::FlatBufferBuilder& builder) const
{
    return wire::CreateHuffmanDecoder(
        builder,
        builder.CreateVector(keys_),
        builder.CreateVector(sizeOffsets_));
}

HuffmanDecoder HuffmanDecoder::load(const wire::HuffmanDecoder* serialized)
{
    return HuffmanDecoder(
        std::vector<uint8_t>(serialized->keys()->begin(), serialized->keys()->end()),
        std::vector<uint32_t>(serialized->size_offsets()->begin(), serialized->size_offsets()->end()));
}

}
