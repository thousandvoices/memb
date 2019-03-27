#include "trained_compression.h"
#include "kmeans.h"
#include "bit_stream.h"
#include "huffman_encoder.h"
#include "huffman_decoder.h"

namespace memb {

namespace {

struct QuantizedWordVector {
    std::string word;
    std::vector<uint8_t> values;
};

struct StorageNode {
    std::string word;
    uint32_t offset;
};

const size_t CLUSTER_SAMPLE_SIZE = 10000;

} // namespace

TrainedCompressor::TrainedCompressor(
        flatbuffers::FlatBufferBuilder& builder,
        size_t bitsPerWeight):
    builder_(builder),
    quantizationLevels_(std::min(1 << bitsPerWeight, 255))
{}

void TrainedCompressor::add(
    const std::string& word,
    const float* source,
    size_t dim)
{
    embeddings_.push_back({word, std::vector<float>(source, source + dim)});
}

flatbuffers::Offset<void> TrainedCompressor::finalize()
{
    std::vector<float> valuesSample;

    for (size_t i = 0; i < std::min(CLUSTER_SAMPLE_SIZE, embeddings_.size()); ++i) {
        valuesSample.insert(
            valuesSample.end(), embeddings_[i].values.begin(), embeddings_[i].values.end());
    }

    HuffmanEncoderBuilder encoderBuilder;
    std::vector<QuantizedWordVector> quantizedVectors;
    KMeansClusterizer clusterizer(quantizationLevels_);
    clusterizer.fit(valuesSample);

    for (const auto& wordVector : embeddings_) {
        auto quantizedWordVector = clusterizer.predict(wordVector.values);
        encoderBuilder.updateFrequencies(quantizedWordVector);
        quantizedVectors.push_back({wordVector.word, quantizedWordVector});
    }

    auto encoder = encoderBuilder.createEncoder();

    std::vector<StorageNode> nodes;
    std::vector<uint8_t> packedValues;

    for (const auto& item : quantizedVectors) {
        auto offset = packedValues.size();
        auto encodedValues = encoder.encode(item.values);
        packedValues.insert(
            packedValues.end(), encodedValues.begin(), encodedValues.end());
        nodes.push_back(StorageNode{item.word, static_cast<uint32_t>(offset)});
    }

    std::sort(
        nodes.begin(),
        nodes.end(),
        [](const StorageNode& lhs, const StorageNode& rhs)
        {
            return lhs.word < rhs.word;
        });

    std::string packedWords;
    std::vector<uint32_t> wordOffsets;
    std::vector<uint32_t> valueOffsets;

    for (const auto& node : nodes) {
        wordOffsets.push_back(packedWords.size());
        valueOffsets.push_back(node.offset);

        packedWords.insert(packedWords.size(), node.word.c_str(), node.word.size() + 1);
    }

    return wire::CreateTrained(
        builder_,
        builder_.CreateVector(wordOffsets),
        builder_.CreateVector(valueOffsets),
        builder_.CreateString(packedWords),
        builder_.CreateVector(packedValues),
        encoder.createDecoder().save(builder_),
        clusterizer.save(builder_)
    ).Union();
}

TrainedCompressedStorage::TrainedCompressedStorage(
        const void* flatStorage,
        size_t dim,
        size_t maxDirectDecodeBitLength):
    flatStorage_(static_cast<const wire::Trained*>(flatStorage)),
    dim_(dim),
    huffmanDecoder_(HuffmanDecoder::load(flatStorage_->decoder()).createTableDecoder(maxDirectDecodeBitLength)),
    centroids_(KMeansClusterizer::load(flatStorage_->clusterizer()).centroids())
{}

void TrainedCompressedStorage::extract(const std::string& word, float* destination) const
{
    auto wordData = flatStorage_->packed_words()->data();
    auto resultIt = std::lower_bound(
        flatStorage_->word_offsets()->begin(),
        flatStorage_->word_offsets()->end(),
        word.c_str(),
        [wordData](uint32_t offset, const char* word)
        {
            return strcmp(wordData + offset, word) < 0;
        });

    if (strcmp(flatStorage_->packed_words()->data() + *resultIt, word.c_str()) == 0) {
        size_t offset = flatStorage_->value_offsets()->Get(
            resultIt - flatStorage_->word_offsets()->begin());

        auto decodeState = huffmanDecoder_.decode(
            flatStorage_->packed_values()->data() + offset,
            flatStorage_->packed_values()->size() - offset);

        for (size_t i = 0; i < dim_; ++i) {
            destination[i] = centroids_[huffmanDecoder_.next(decodeState)];
        }
    }
}

std::vector<std::string> TrainedCompressedStorage::keys() const
{
    std::vector<std::string> result;
    result.reserve(flatStorage_->word_offsets()->size());
    auto wordData = flatStorage_->packed_words()->data();

    for (auto wordOffset : *flatStorage_->word_offsets()) {
        result.emplace_back(reinterpret_cast<const char*>(wordData + wordOffset));
    }

    return result;
}

std::shared_ptr<Compressor> TrainedCompressionStrategy::createCompressor(
    flatbuffers::FlatBufferBuilder& builder, size_t bitsPerWeight) const
{
    return std::make_shared<TrainedCompressor>(builder, bitsPerWeight);
}

std::shared_ptr<CompressedStorage> TrainedCompressionStrategy::createCompressedStorage(
    const void* flatStorage, size_t dim) const
{
    return std::make_shared<TrainedCompressedStorage>(flatStorage, dim);
}

std::string TrainedCompressionStrategy::storageName() const
{
    return "trained";
}

wire::Storage TrainedCompressionStrategy::storageType() const
{
    return wire::Storage_Trained;
}

}
