#include "ubyte_compression.h"

namespace memb {

UbyteCompressor::UbyteCompressor(flatbuffers::FlatBufferBuilder& builder):
    builder_(builder)
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
        [minValue, maxValue](float value)
        {
            auto floatResult = 255 * (value - minValue) / (maxValue - minValue);
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

    return CreateUbyte(builder_, flatNodes).Union();
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

        std::transform(
            values->begin(),
            values->end(),
            destination,
            [minValue, maxValue](uint8_t value)
            {
                auto floatValue = static_cast<float>(value);
                return minValue + (maxValue - minValue) * floatValue / 255;
            });
    }
}

std::shared_ptr<Compressor> UbyteCompressionStrategy::createCompressor(
    flatbuffers::FlatBufferBuilder& builder) const
{
    return std::make_shared<UbyteCompressor>(builder);
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
