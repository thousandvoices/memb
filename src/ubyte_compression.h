#pragma once

#include "compression_strategy.h"

#include <unordered_map>

namespace memb {

class UbyteCompressedStorage : public CompressedStorage {
public:
    UbyteCompressedStorage(const void* flatStorage);
    virtual void extract(const std::string& word, float* destination) const override;

private:
    const wire::Ubyte* flatStorage_;
};

class UbyteCompressor : public Compressor {
public:
    UbyteCompressor(flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight);

    virtual void add(
        const std::string& word,
        const float* source,
        size_t dim) override;

    virtual flatbuffers::Offset<void> finalize() override;

private:
    std::unordered_map<std::string, flatbuffers::Offset<wire::UbyteVector>> embeddings_;
    flatbuffers::FlatBufferBuilder& builder_;
    uint8_t quantizationLevels_;
};

class UbyteCompressionStrategy : public CompressionStrategy {
public:
    virtual std::shared_ptr<Compressor> createCompressor(
        flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const override;

    virtual std::shared_ptr<CompressedStorage> createCompressedStorage(
        const void* flatStorage, size_t dim) const override;

    virtual std::string storageName() const override;

    virtual wire::Storage storageType() const override;
};

}
