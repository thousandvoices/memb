#pragma once

#include "embeddings_generated.h"
#include "compression_strategy.h"

#include <boost/iostreams/device/mapped_file.hpp>

namespace memb {

class Reader {
public:
    Reader(const std::string& filename);
    Reader(
        const std::string& filename,
        std::shared_ptr<CompressionStrategy> compressionStrategy);

    size_t dim() const;

    std::vector<std::string> keys() const;

    void wordEmbeddingToBuffer(const std::string& word, float* buffer) const;
    void batchEmbeddingToBuffer(const std::vector<std::string>& words, float* buffer) const;

    std::vector<float> wordEmbedding(const std::string& word) const;
    std::vector<float> batchEmbedding(const std::vector<std::string>& words) const;

private:
    boost::iostreams::mapped_file_source mappedFile_;
    const wire::Index* flatIndex_;
    std::shared_ptr<CompressedStorage> compressedStorage_;
};

}
