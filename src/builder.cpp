#include "builder.h"
#include "compression_strategy.h"

#include <boost/format.hpp>

#include <fstream>

namespace memb {

namespace {

const std::string DIMENSION_MISMATCH_MESSAGE_TEMPLATE =
    "Vector dimension (%d) for word %s doesn't match builder dimension (%d)";

const std::string DUPLICATE_MESSAGE_TEMPLATE =
    "Attempt to add duplicate word %s to index";

} // namespace

Builder::Builder(size_t dim, wire::Storage storageType):
    dim_(dim),
    storageType_(storageType),
    compressor_(createCompressionStrategy(storageType)->createCompressor(builder_))
{}

Builder::Builder(size_t dim, const std::string& storageName):
    dim_(dim)
{
    auto compressionStrategy = createCompressionStrategy(storageName);
    storageType_ = compressionStrategy->storageType();
    compressor_ = compressionStrategy->createCompressor(builder_);
}

void Builder::addWord(const std::string& word, const std::vector<float>& embedding)
{
    if (embedding.size() != dim_) {
        throw std::runtime_error(boost::str(
            boost::format(DIMENSION_MISMATCH_MESSAGE_TEMPLATE) % embedding.size() % word % dim_));
    }

    auto insertionResult = addedWords_.insert(word);
    if (!insertionResult.second) {
        throw std::runtime_error(boost::str(
            boost::format(DUPLICATE_MESSAGE_TEMPLATE) % word));
    }

    compressor_->add(word, embedding.data(), dim_);
}

void Builder::dump(std::ostream& sink)
{
    auto storage = compressor_->finalize();
    wire::IndexBuilder indexBuilder(builder_);
    indexBuilder.add_dim(dim_);
    indexBuilder.add_storage_type(storageType_);
    indexBuilder.add_storage(storage);
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
