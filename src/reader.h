#pragma once

#include "embeddings_generated.h"
#include "quantization.h"

#include <boost/iostreams/device/mapped_file.hpp>

namespace memb {

class Reader {
public:
    Reader(const std::string& filename);

    size_t dim() const;

    void wordEmbeddingToBuffer(const std::string& word, float* buffer) const;
    void batchEmbeddingToBuffer(const std::vector<std::string>& words, float* buffer) const;

    std::vector<float> wordEmbedding(const std::string& word) const;
    std::vector<float> batchEmbedding(const std::vector<std::string>& words) const;

private:
    boost::iostreams::mapped_file_source mappedFile_;
    std::unordered_map<Storage, std::shared_ptr<Quantizer>> quantizationMap_;
    const Index* flatIndex_;
};

}
