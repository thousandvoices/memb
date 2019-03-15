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
        for (auto value : embeddings_[i].values) {
            valuesSample.push_back(value);
        }
    }

    KMeansClusterizer clusterizer(quantizationLevels_);
    clusterizer.fit(valuesSample);

    std::vector<QuantizedWordVector> quantizedVectors;
    std::transform(
        embeddings_.begin(),
        embeddings_.end(),
        std::back_inserter(quantizedVectors),
        [&clusterizer](const WordVector& wordVector)
        {
            return QuantizedWordVector{wordVector.word, clusterizer.predict(wordVector.values)};
        });

    HuffmanEncoderBuilder encoderBuilder;
    for (auto wordVector : quantizedVectors) {
        encoderBuilder.updateFrequencies(wordVector.values);
    }
    auto encoder = encoderBuilder.createEncoder();

    auto flatDecoder = encoder.createDecoder().save(builder_);
    auto flatClusterizer = clusterizer.save(builder_);

    std::vector<flatbuffers::Offset<wire::TrainedQuantizedNode>> nodes;
    std::vector<uint8_t> compressedValues;

    for (const auto& item : quantizedVectors) {
        auto offset = compressedValues.size();
        auto encodedValues = encoder.encode(item.values);
        compressedValues.insert(
            compressedValues.end(), encodedValues.begin(), encodedValues.end());
        auto word = builder_.CreateString(item.word);
        nodes.push_back(wire::CreateTrainedQuantizedNode(builder_, word, offset));
    }

    auto flatCompressedValues = builder_.CreateVector(compressedValues);
    auto flatNodes = builder_.CreateVectorOfSortedTables(&nodes);

    return wire::CreateTrained(builder_, flatNodes, flatCompressedValues, flatDecoder, flatClusterizer).Union();
}

TrainedCompressedStorage::TrainedCompressedStorage(
        const void* flatStorage,
        size_t dim,
        size_t maxDirectDecodeBitLength):
    flatStorage_(static_cast<const wire::Trained*>(flatStorage)),
    dim_(dim),
    huffmanDecoder_(HuffmanDecoder::load(flatStorage_->decoder()).createTableDecoder(maxDirectDecodeBitLength)),
    centroids_(KMeansClusterizer::load(flatStorage_->clusterizer()).centroids())
{
}

void TrainedCompressedStorage::extract(const std::string& word, float* destination) const
{
    auto resultNode = flatStorage_->nodes()->LookupByKey(word.c_str());
    if (resultNode) {
        uint8_t unpackBuffer[dim_];

        size_t offset = resultNode->offset();
        huffmanDecoder_.decode(
            flatStorage_->compressed_values()->data() + offset,
            flatStorage_->compressed_values()->size() - offset,
            unpackBuffer,
            dim_);

        for (size_t i = 0; i < dim_; ++i) {
            destination[i] = centroids_[unpackBuffer[i]];
        }
    }
}

std::vector<std::string> TrainedCompressedStorage::keys() const
{
    std::vector<std::string> result;
    result.reserve(flatStorage_->nodes()->size());

    for (const auto& node : *flatStorage_->nodes()) {
        result.emplace_back(node->word()->begin(), node->word()->end());
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
