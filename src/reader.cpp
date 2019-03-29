#include "reader.h"

#include <future>

namespace memb {

namespace {

const size_t THREADED_DECODER_THRESHOLD = 1024;

} // namespace

Reader::Reader(const std::string& filename,
               std::shared_ptr<CompressionStrategy> compressionStrategy,
               size_t numThreads):
    numThreads_(adjustedNumThreads(numThreads)),
    mappedFile_(filename),
    flatIndex_(getIndexChecked()),
    compressedStorage_(compressionStrategy->createCompressedStorage(
        flatIndex_->storage(), flatIndex_->dim()))
{}

Reader::Reader(const std::string& filename, size_t numThreads):
    numThreads_(adjustedNumThreads(numThreads)),
    mappedFile_(filename),
    flatIndex_(getIndexChecked()),
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
    auto extractResult = compressedStorage_->extract(word, buffer);
    if (!extractResult) {
        std::fill(buffer, buffer + dim(), 0);
    }
}

void Reader::batchEmbeddingToBufferImpl(
    boost::iterator_range<std::vector<std::string>::const_iterator> words,
    float* buffer) const
{
    size_t stride = flatIndex_->dim();
    for (size_t idx = 0; idx < words.size(); ++idx) {
        wordEmbeddingToBuffer(words[idx], buffer + stride * idx);
    }
}

void Reader::batchEmbeddingToBuffer(const std::vector<std::string>& words, float* buffer) const
{
    if (words.size() < THREADED_DECODER_THRESHOLD || numThreads_ == 1) {
        batchEmbeddingToBufferImpl(
            boost::make_iterator_range(words.begin(), words.end()), buffer);
    } else {
        size_t jobSize = (words.size() + numThreads_ - 1) / numThreads_;

        size_t startIndex = 0;
        std::vector<std::future<void>> results;
        while (startIndex < words.size()) {
            size_t endIndex = std::min(startIndex + jobSize, words.size());
            auto request = boost::make_iterator_range(
                words.begin() + startIndex, words.begin() + endIndex);
            float* bufferOffset = buffer + startIndex * flatIndex_->dim();
            results.push_back(std::async(
                [this, request, bufferOffset]
                {
                    batchEmbeddingToBufferImpl(request, bufferOffset);
                }));
            startIndex += jobSize;
        }

        for (auto& future : results) {
            future.get();
        }
    }
}

std::vector<float> Reader::wordEmbedding(const std::string& word) const
{
    std::vector<float> result(flatIndex_->dim());
    wordEmbeddingToBuffer(word, result.data());

    return result;
}

std::vector<float> Reader::batchEmbedding(const std::vector<std::string>& words) const
{
    std::vector<float> result(flatIndex_->dim() * words.size());
    batchEmbeddingToBuffer(words, result.data());

    return result;
}

const wire::Index* Reader::getIndexChecked() const
{
    if (mappedFile_.size() < 8 || !wire::IndexBufferHasIdentifier(mappedFile_.data())) {
        throw std::runtime_error("File format verification failed");
    }

    return wire::GetIndex(mappedFile_.data());
}

size_t Reader::adjustedNumThreads(size_t numThreads) const
{
    if (numThreads > 0) {
        return numThreads;
    }

    return std::max(std::thread::hardware_concurrency(), 1u);
}

}
