#pragma once

#include "compression_strategy.h"

#include <unordered_map>

namespace memb {

class FullCompressedStorage : public CompressedStorage {
public:
    FullCompressedStorage(const void* flatStorage);
    virtual void extract(const std::string& word, float* destination) const override;
    virtual std::vector<std::string> keys() const override;

private:
    const wire::Full* flatStorage_;
};

class FullCompressor : public Compressor {
public:
    FullCompressor(flatbuffers::FlatBufferBuilder& builder);

    virtual void add(
        const std::string& word,
        const float* source,
        size_t dim) override;

    virtual flatbuffers::Offset<void> finalize() override;

private:
    std::unordered_map<std::string, flatbuffers::Offset<flatbuffers::Vector<float>>> embeddings_;
    flatbuffers::FlatBufferBuilder& builder_;
};

class FullCompressionStrategy : public CompressionStrategy {
public:
    virtual std::shared_ptr<Compressor> createCompressor(
        flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const override;

    virtual std::shared_ptr<CompressedStorage> createCompressedStorage(
        const void* flatStorage, size_t dim) const override;

    virtual std::string storageName() const override;

    virtual wire::Storage storageType() const override;
};

}
