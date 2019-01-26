#include "ubyte_compression.h"

namespace memb {

UbyteCompressor::UbyteCompressor(flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight):
    builder_(builder),
    quantizationLevels_(std::min(1 << bitsPerWeight, 255))
{}

void UbyteCompressor::add(
    const std::string& word,
    const float* source,
    size_t dim)
{
    auto minMaxValues = std::minmax_element(source, source + dim);
    auto minValue = *(minMaxValues.first);
    auto maxValue = *(minMaxValues.second);

    std::vector<uint8_t> quantizedValues(dim);
    std::transform(
        source,
        source + dim,
        quantizedValues.begin(),
        [minValue, maxValue, this](float value)
        {
            auto floatResult = quantizationLevels_ * (value - minValue) / (maxValue - minValue);
            return static_cast<uint8_t>(floatResult);
        });

    auto values = builder_.CreateVector(quantizedValues);
    embeddings_.emplace(
        word,
        wire::CreateUbyteVector(builder_, minValue, maxValue, values));
}

flatbuffers::Offset<void> UbyteCompressor::finalize()
{
    std::vector<flatbuffers::Offset<wire::UbyteNode>> nodes;
    for (const auto& item : embeddings_) {
        auto word = builder_.CreateString(item.first);

        nodes.push_back(wire::CreateUbyteNode(builder_, word, item.second));
    }

    auto flatNodes = builder_.CreateVectorOfSortedTables(&nodes);

    return CreateUbyte(builder_, flatNodes, quantizationLevels_).Union();
}

UbyteCompressedStorage::UbyteCompressedStorage(const void* flatStorage):
    flatStorage_(static_cast<const wire::Ubyte*>(flatStorage))
{}

void UbyteCompressedStorage::extract(const std::string& word, float* destination) const
{
    auto resultNode = flatStorage_->nodes()->LookupByKey(word.c_str());
    if (resultNode) {
        auto ubyteStorage = resultNode->compressed_values();
        auto minValue = ubyteStorage->min_value();
        auto maxValue = ubyteStorage->max_value();
        auto values = ubyteStorage->values();
        auto quantizationLevels = flatStorage_->quantization_levels();

        std::transform(
            values->begin(),
            values->end(),
            destination,
            [minValue, maxValue, quantizationLevels](uint8_t value)
            {
                auto floatValue = static_cast<float>(value);
                return minValue + (maxValue - minValue) * floatValue / quantizationLevels;
            });
    }
}

std::shared_ptr<Compressor> UbyteCompressionStrategy::createCompressor(
    flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const
{
    return std::make_shared<UbyteCompressor>(builder, bitsPerWeight);
}

std::shared_ptr<CompressedStorage> UbyteCompressionStrategy::createCompressedStorage(
    const void* flatStorage, size_t /*dim*/) const
{
    return std::make_shared<UbyteCompressedStorage>(flatStorage);
}

std::string UbyteCompressionStrategy::storageName() const
{
    return "ubyte";
}

wire::Storage UbyteCompressionStrategy::storageType() const
{
    return wire::Storage_Ubyte;
}

}
