#pragma once

#include "embeddings_generated.h"
#include "quantization.h"

#include <unordered_map>
#include <iostream>

namespace memb {

class Builder {
public:
    Builder(size_t dim, Storage storageType);
    Builder(size_t dim, const std::string& storageType);

    void addWord(const std::string& word, const std::vector<float>& embedding);
    void dump(std::ostream& sink);
    void save(const std::string& filename);

private:
    size_t dim_;
    flatbuffers::FlatBufferBuilder builder_;
    std::shared_ptr<Quantizer> quantizer_;
    std::unordered_map<std::string, flatbuffers::Offset<void>> embeddings_;
};

}
