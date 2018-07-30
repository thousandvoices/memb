#include "quantization.h"

namespace memb {

flatbuffers::Offset<void> FullQuantizer::quantize(
    flatbuffers::FlatBufferBuilder& builder,
    const float* source,
    size_t dim) const
{
    auto values = builder.CreateVector(source, dim);
    return CreateFull(builder, values).Union();
}

void FullQuantizer::dequantize(const void* storage, float* destination) const
{
    auto values = static_cast<const Full*>(storage)->values();
    std::copy(values->begin(), values->end(), destination);
}

Storage FullQuantizer::storageType() const
{
    return Storage_Full;
}

flatbuffers::Offset<void> UbyteQuantizer::quantize(
    flatbuffers::FlatBufferBuilder& builder,
    const float* source,
    size_t dim) const
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

    auto values = builder.CreateVector(quantizedValues);
    return CreateUbyte(builder, minValue, maxValue, values).Union();
}

void UbyteQuantizer::dequantize(const void* storage, float* destination) const
{
    auto ubyteStorage = static_cast<const Ubyte*>(storage);
    auto values = ubyteStorage->values();
    auto minValue = ubyteStorage->min_value();
    auto maxValue = ubyteStorage->max_value();

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

Storage UbyteQuantizer::storageType() const
{
    return Storage_Ubyte;
}

std::unordered_map<Storage, std::shared_ptr<Quantizer>> createQuantizationMap()
{
    std::unordered_map<Storage, std::shared_ptr<Quantizer>> result = {
        {Storage_Full, std::make_shared<FullQuantizer>()},
        {Storage_Ubyte, std::make_shared<UbyteQuantizer>()}
    };

    return result;
}

}
