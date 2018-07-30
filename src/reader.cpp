#include "reader.h"

namespace memb {

Reader::Reader(const std::string& filename):
    mappedFile_(filename),
    quantizationMap_(createQuantizationMap()),
    flatIndex_(GetIndex(mappedFile_.data()))
{}

size_t Reader::dim() const
{
    return flatIndex_->dim();
}

void Reader::wordEmbeddingToBuffer(const std::string& word, float* buffer) const
{
    auto resultNode = flatIndex_->nodes()->LookupByKey(word.c_str());
    if (resultNode) {
        quantizationMap_.at(resultNode->storage_type())->dequantize(resultNode->storage(), buffer);
    }
}

void Reader::batchEmbeddingToBuffer(const std::vector<std::string>& words, float* buffer) const
{
    size_t stride = flatIndex_->dim();
    for (size_t idx = 0; idx < words.size(); ++idx) {
        wordEmbeddingToBuffer(words[idx], buffer + stride * idx);
    }
}

std::vector<float> Reader::wordEmbedding(const std::string& word) const
{
    std::vector<float> result(flatIndex_->dim(), 0);
    wordEmbeddingToBuffer(word, result.data());

    return result;
}

std::vector<float> Reader::batchEmbedding(const std::vector<std::string>& words) const
{
    std::vector<float> result(flatIndex_->dim() * words.size(), 0);
    batchEmbeddingToBuffer(words, result.data());

    return result;
}

}
