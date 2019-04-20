#include "huffman_decoder.h"

namespace memb {

HuffmanDecoder::HuffmanDecoder(const std::vector<uint8_t>& keys, const std::vector<uint32_t>& offsets):
    keys_(keys),
    sizeOffsets_(offsets)
{}

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
