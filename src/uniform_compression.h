#pragma once

#include "compression_strategy.h"

#include <unordered_map>

namespace memb {

class UniformCompressedStorage : public CompressedStorage {
public:
    UniformCompressedStorage(const void* flatStorage);
    virtual bool extract(const std::string& word, float* destination) const override;
    virtual std::vector<std::string> keys() const override;

private:
    const wire::Uniform* flatStorage_;
};

class UniformCompressor : public Compressor {
public:
    UniformCompressor(flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight);

    virtual void add(
        const std::string& word,
        const float* source,
        size_t dim) override;

    virtual flatbuffers::Offset<void> finalize() override;

private:
    std::unordered_map<std::string, flatbuffers::Offset<wire::UniformQuantizedVector>> embeddings_;
    flatbuffers::FlatBufferBuilder& builder_;
    uint8_t quantizationLevels_;
};

class UniformCompressionStrategy : public CompressionStrategy {
public:
    virtual std::shared_ptr<Compressor> createCompressor(
        flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const override;

    virtual std::shared_ptr<CompressedStorage> createCompressedStorage(
        const void* flatStorage, size_t dim) const override;

    virtual std::string storageName() const override;

    virtual wire::Storage storageType() const override;
};

}
