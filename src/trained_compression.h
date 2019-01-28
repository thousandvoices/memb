#pragma once

#include "prefix_code.h"
#include "compression_strategy.h"
#include "huffman_decoder.h"

namespace memb {

class TrainedCompressedStorage : public CompressedStorage {
public:
    static const size_t DEFAULT_DECODE_TABLE_BIT_LENGTH = 10;

    TrainedCompressedStorage(
        const void* flatStorage,
        size_t dim,
        size_t maxDirectDecodeBitLength = DEFAULT_DECODE_TABLE_BIT_LENGTH);
    virtual void extract(const std::string& word, float* destination) const override;
    virtual std::vector<std::string> keys() const override;

private:
    const wire::Trained* flatStorage_;
    size_t dim_;
    HuffmanTableDecoder huffmanDecoder_;
    std::vector<float> centroids_;
};

class TrainedCompressor : public Compressor {
public:
    TrainedCompressor(
        flatbuffers::FlatBufferBuilder& builder,
        size_t bitsPerWeight);

    virtual void add(
        const std::string& word,
        const float* source,
        size_t dim) override;

    virtual flatbuffers::Offset<void> finalize() override;

private:
    struct WordVector {
        std::string word;
        std::vector<float> values;
    };

    std::vector<WordVector> embeddings_;
    flatbuffers::FlatBufferBuilder& builder_;
    uint8_t quantizationLevels_;
};

class TrainedCompressionStrategy : public CompressionStrategy {
public:
    virtual std::shared_ptr<Compressor> createCompressor(
        flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const override;

    virtual std::shared_ptr<CompressedStorage> createCompressedStorage(
        const void* flatStorage, size_t dim) const override;

    virtual std::string storageName() const override;

    virtual wire::Storage storageType() const override;
};

}
