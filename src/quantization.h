#pragma once

#include "embeddings_generated.h"
#include <unordered_map>

namespace memb {

class Quantizer {
public:
    virtual flatbuffers::Offset<void> quantize(
        flatbuffers::FlatBufferBuilder& builder,
        const float* source,
        size_t dim) const = 0;

    virtual void dequantize(const void* storage, float* destination) const = 0;
    virtual Storage storageType() const = 0;
    virtual ~Quantizer() {};
};

class FullQuantizer : public Quantizer {
public:
    virtual flatbuffers::Offset<void> quantize(
        flatbuffers::FlatBufferBuilder& builder,
        const float* source,
        size_t dim) const override;

    virtual void dequantize(const void* storage, float* destination) const override;

    virtual Storage storageType() const override;
};

class UbyteQuantizer : public Quantizer {
public:
    virtual flatbuffers::Offset<void> quantize(
        flatbuffers::FlatBufferBuilder& builder,
        const float* source,
        size_t dim) const override;

    virtual void dequantize(const void* storage, float* destination) const override;

    virtual Storage storageType() const override;
};

std::unordered_map<Storage, std::shared_ptr<Quantizer>> createQuantizationMap();

}
