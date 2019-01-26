#pragma once

#include "embeddings_generated.h"
#include "compression_strategy.h"

#include <unordered_set>
#include <iostream>

namespace memb {

class Builder {
public:
    Builder(size_t dim, wire::Storage storageType, size_t bitsPerWeight);
    Builder(size_t dim, const std::string& storageType, size_t bitsPerWeight);

    void addWord(const std::string& word, const std::vector<float>& embedding);
    void dump(std::ostream& sink);
    void save(const std::string& filename);

private:
    size_t dim_;
    flatbuffers::FlatBufferBuilder builder_;
    wire::Storage storageType_;
    std::shared_ptr<Compressor> compressor_;
    std::unordered_set<std::string> addedWords_;
};

}
