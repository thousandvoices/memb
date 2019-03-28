#include "full_compression.h"

namespace memb {

FullCompressor::FullCompressor(flatbuffers::FlatBufferBuilder& builder):
    builder_(builder)
{}

void FullCompressor::add(
    const std::string& word,
    const float* source,
    size_t dim)
{
    qwerty;
    embeddings_.emplace(
        word,
        builder_.CreateVector(source, dim));
}

flatbuffers::Offset<void> FullCompressor::finalize()
{
    std::vector<flatbuffers::Offset<wire::FullNode>> nodes;
    for (const auto& item : embeddings_) {
        auto word = builder_.CreateString(item.first);

        nodes.push_back(wire::CreateFullNode(builder_, word, item.second));
    }

    auto flatNodes = builder_.CreateVectorOfSortedTables(&nodes);

    return wire::CreateFull(builder_, flatNodes).Union();
}

FullCompressedStorage::FullCompressedStorage(const void* flatStorage):
    flatStorage_(static_cast<const wire::Full*>(flatStorage))
{}

void FullCompressedStorage::extract(const std::string& word, float* destination) const
{
    auto resultNode = flatStorage_->nodes()->LookupByKey(word.c_str());
    if (resultNode) {
        auto values = resultNode->values();
        std::copy(values->begin(), values->end(), destination);
    }
}

std::vector<std::string> FullCompressedStorage::keys() const
{
    std::vector<std::string> result;
    result.reserve(flatStorage_->nodes()->size());

    for (const auto& node : *flatStorage_->nodes()) {
        result.emplace_back(node->word()->begin(), node->word()->end());
    }

    return result;
}

std::shared_ptr<Compressor> FullCompressionStrategy::createCompressor(
    flatbuffers::FlatBufferBuilder& builder, size_t /*bitsPerWeight*/) const
{
    return std::make_shared<FullCompressor>(builder);
}

std::shared_ptr<CompressedStorage> FullCompressionStrategy::createCompressedStorage(
    const void* flatStorage, size_t /*dim*/) const
{
    return std::make_shared<FullCompressedStorage>(flatStorage);
}

std::string FullCompressionStrategy::storageName() const
{
    return "full";
}

wire::Storage FullCompressionStrategy::storageType() const
{
    return wire::Storage_Full;
}

}
