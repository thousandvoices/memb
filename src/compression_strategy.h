#pragma once

#include "embeddings_generated.h"

namespace memb {

class CompressedStorage {
public:
    virtual void extract(const std::string& word, float* destination) const = 0;
    virtual ~CompressedStorage() {};
};

class Compressor {
public:
    virtual void add(
        const std::string& word,
        const float* source,
        size_t dim) = 0;

    virtual flatbuffers::Offset<void> finalize() = 0;
    virtual ~Compressor() {};
};

class CompressionStrategy {
public:
    virtual std::shared_ptr<Compressor> createCompressor(
        flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const = 0;

    virtual std::shared_ptr<CompressedStorage> createCompressedStorage(
        const void* flatStorage, size_t dim) const = 0;

    virtual std::string storageName() const = 0;

    virtual wire::Storage storageType() const = 0;

    virtual ~CompressionStrategy() {};
};

std::shared_ptr<CompressionStrategy> createCompressionStrategy(wire::Storage storage);
std::shared_ptr<CompressionStrategy> createCompressionStrategy(const std::string& storageName);

std::vector<std::string> availableCompressionStrategies();

}
