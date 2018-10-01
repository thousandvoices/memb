#include "huffman_compression.h"
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

const size_t QUANTIZATION_LEVELS = 16;
const size_t CLUSTER_SAMPLE_SIZE = 10000;

} // namespace

HuffmanCompressor::HuffmanCompressor(flatbuffers::FlatBufferBuilder& builder):
    builder_(builder)
{}

void HuffmanCompressor::add(
    const std::string& word,
    const float* source,
    size_t dim)
{
    embeddings_.push_back({word, std::vector<float>(source, source + dim)});
}

flatbuffers::Offset<void> HuffmanCompressor::finalize()
{
    std::vector<float> valuesSample;

    for (size_t i = 0; i < std::min(CLUSTER_SAMPLE_SIZE, embeddings_.size()); ++i) {
        for (auto value : embeddings_[i].values) {
            valuesSample.push_back(value);
        }
    }

    KMeansClusterizer clusterizer(QUANTIZATION_LEVELS);
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

    std::unordered_map<std::string, flatbuffers::Offset<flatbuffers::Vector<uint8_t>>> compressedVectors;
    for (const auto& item : quantizedVectors) {
        auto values = builder_.CreateVector(encoder.encode(item.values));
        compressedVectors[item.word] = values;
    }

    std::vector<flatbuffers::Offset<wire::HuffmanNode>> nodes;
    for (const auto& item : embeddings_) {
        auto word = builder_.CreateString(item.word);
        nodes.push_back(wire::CreateHuffmanNode(builder_, word, compressedVectors[item.word]));
    }

    auto flatNodes = builder_.CreateVectorOfSortedTables(&nodes);

    return wire::CreateHuffman(builder_, flatNodes, flatDecoder, flatClusterizer).Union();
}

HuffmanCompressedStorage::HuffmanCompressedStorage(
        const void* flatStorage,
        size_t dim,
        size_t maxDirectDecodeBitLength):
    flatStorage_(static_cast<const wire::Huffman*>(flatStorage)),
    dim_(dim),
    huffmanDecoder_(HuffmanDecoder::load(flatStorage_->decoder()).createTableDecoder(maxDirectDecodeBitLength)),
    centroids_(KMeansClusterizer::load(flatStorage_->clusterizer()).centroids())
{
}

void HuffmanCompressedStorage::extract(const std::string& word, float* destination) const
{
    auto resultNode = flatStorage_->nodes()->LookupByKey(word.c_str());
    if (resultNode) {
        uint8_t unpackBuffer[dim_];

        huffmanDecoder_.decode(
            resultNode->compressed_values()->data(),
            resultNode->compressed_values()->size(),
            unpackBuffer,
            dim_);

        for (size_t i = 0; i < dim_; ++i) {
            *(destination + i) = centroids_[unpackBuffer[i]];
        }
    }
}

std::shared_ptr<Compressor> HuffmanCompressionStrategy::createCompressor(
    flatbuffers::FlatBufferBuilder& builder) const
{
    return std::make_shared<HuffmanCompressor>(builder);
}

std::shared_ptr<CompressedStorage> HuffmanCompressionStrategy::createCompressedStorage(
    const void* flatStorage, size_t dim) const
{
    return std::make_shared<HuffmanCompressedStorage>(flatStorage, dim);
}

std::string HuffmanCompressionStrategy::storageName() const
{
    return "huffman";
}

wire::Storage HuffmanCompressionStrategy::storageType() const
{
    return wire::Storage_Huffman;
}

}
