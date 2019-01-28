#include "reader.h"

namespace memb {

Reader::Reader(const std::string& filename,
               std::shared_ptr<CompressionStrategy> compressionStrategy):
    mappedFile_(filename),
    flatIndex_(wire::GetIndex(mappedFile_.data())),
    compressedStorage_(compressionStrategy->createCompressedStorage(
        flatIndex_->storage(), flatIndex_->dim()))
{
}

Reader::Reader(const std::string& filename):
    mappedFile_(filename),
    flatIndex_(wire::GetIndex(mappedFile_.data())),
    compressedStorage_(createCompressionStrategy(flatIndex_->storage_type())->createCompressedStorage(
        flatIndex_->storage(), flatIndex_->dim()))
{}

size_t Reader::dim() const
{
    return flatIndex_->dim();
}

std::vector<std::string> Reader::keys() const
{
    return compressedStorage_->keys();
}

void Reader::wordEmbeddingToBuffer(const std::string& word, float* buffer) const
{
    return compressedStorage_->extract(word, buffer);
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
