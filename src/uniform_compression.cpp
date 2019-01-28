#include "uniform_compression.h"

namespace memb {

UniformCompressor::UniformCompressor(flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight):
    builder_(builder),
    quantizationLevels_(std::min(1 << bitsPerWeight, 255))
{}

void UniformCompressor::add(
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
        wire::CreateUniformQuantizedVector(builder_, minValue, maxValue, values));
}

flatbuffers::Offset<void> UniformCompressor::finalize()
{
    std::vector<flatbuffers::Offset<wire::UniformQuantizedNode>> nodes;
    for (const auto& item : embeddings_) {
        auto word = builder_.CreateString(item.first);

        nodes.push_back(wire::CreateUniformQuantizedNode(builder_, word, item.second));
    }

    auto flatNodes = builder_.CreateVectorOfSortedTables(&nodes);

    return CreateUniform(builder_, flatNodes, quantizationLevels_).Union();
}

UniformCompressedStorage::UniformCompressedStorage(const void* flatStorage):
    flatStorage_(static_cast<const wire::Uniform*>(flatStorage))
{}

void UniformCompressedStorage::extract(const std::string& word, float* destination) const
{
    auto resultNode = flatStorage_->nodes()->LookupByKey(word.c_str());
    if (resultNode) {
        auto uniformStorage = resultNode->compressed_values();
        auto minValue = uniformStorage->min_value();
        auto maxValue = uniformStorage->max_value();
        auto values = uniformStorage->values();
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

std::vector<std::string> UniformCompressedStorage::keys() const
{
    std::vector<std::string> result;
    result.reserve(flatStorage_->nodes()->size());

    for (const auto& node : *flatStorage_->nodes()) {
        result.emplace_back(node->word()->begin(), node->word()->end());
    }

    return result;
}


std::shared_ptr<Compressor> UniformCompressionStrategy::createCompressor(
    flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const
{
    return std::make_shared<UniformCompressor>(builder, bitsPerWeight);
}

std::shared_ptr<CompressedStorage> UniformCompressionStrategy::createCompressedStorage(
    const void* flatStorage, size_t /*dim*/) const
{
    return std::make_shared<UniformCompressedStorage>(flatStorage);
}

std::string UniformCompressionStrategy::storageName() const
{
    return "uniform";
}

wire::Storage UniformCompressionStrategy::storageType() const
{
    return wire::Storage_Uniform;
}

}
