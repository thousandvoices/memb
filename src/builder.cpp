#include "builder.h"

#include <fstream>

namespace memb {

namespace {

std::unordered_map<std::string, Storage> STORAGE_TYPES = {
    {"full", Storage_Full},
    {"ubyte", Storage_Ubyte},
};

} // namespace

Builder::Builder(size_t dim, Storage storageType):
    dim_(dim),
    quantizer_(createQuantizationMap().at(storageType))
{}

Builder::Builder(size_t dim, const std::string& storageType):
    dim_(dim),
    quantizer_(createQuantizationMap().at(STORAGE_TYPES.at(storageType)))
{}

void Builder::addWord(const std::string& word, const std::vector<float>& embedding)
{
    if (embedding.size() != dim_) {
        throw std::runtime_error("Word vector and builder dimensions don't match");
    }

    auto insertionResult = embeddings_.emplace(
        word,
        quantizer_->quantize(builder_, embedding.data(), dim_));
    
    if (!insertionResult.second) {
        throw std::runtime_error("Attempt to add duplicate word to index");
    }
}

void Builder::dump(std::ostream& sink)
{
    std::vector<flatbuffers::Offset<Node>> nodes;
    for (const auto& item : embeddings_) {
        auto word = builder_.CreateString(item.first);

        NodeBuilder nodeBuilder(builder_);
        nodeBuilder.add_storage_type(quantizer_->storageType());
        nodeBuilder.add_storage(item.second);
        nodeBuilder.add_word(word);
        nodes.push_back(nodeBuilder.Finish());
    }

    auto flatNodes = builder_.CreateVectorOfSortedTables(&nodes);

    IndexBuilder indexBuilder(builder_);
    indexBuilder.add_dim(dim_);
    indexBuilder.add_nodes(flatNodes);
    builder_.Finish(indexBuilder.Finish());

    sink << std::string(
        reinterpret_cast<char*>(builder_.GetBufferPointer()),
        builder_.GetSize());
}

void Builder::save(const std::string& filename)
{
    std::ofstream f(filename, std::ios::binary);
    dump(f);
}

}
