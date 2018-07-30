#include "builder.h"
#include "reader.h"

#define BOOST_AUTO_TEST_MAIN
#include <boost/test/unit_test.hpp>

using namespace memb;

const std::string STORAGE_FILENAME = "data.bin";

struct WordVector {
    std::string word;
    std::vector<float> embedding;
};

BOOST_AUTO_TEST_CASE(builderWorks)
{
    std::vector<WordVector> wordVectors = {
        {"the", {0.0, 1.0, 2.0}},
        {"of", {0.0, -1.0, 2.0}},
        {"a", {1.0, 0.0, -2.0}}
    };

    Builder builder(3, Storage_Full);
    for (const auto& wordVector : wordVectors) {
        builder.addWord(wordVector.word, wordVector.embedding);
    }

    builder.save(STORAGE_FILENAME);

    Reader reader(STORAGE_FILENAME);

    for (const auto& wordVector : wordVectors) {
        auto embedding = reader.wordEmbedding(wordVector.word);
        BOOST_REQUIRE(embedding.size() == wordVector.embedding.size());
        for (size_t idx = 0; idx < embedding.size(); ++idx) {
            BOOST_CHECK_CLOSE_FRACTION(embedding[idx], wordVector.embedding[idx], 0.01);
        }
    }

    auto missingEmbedding = reader.wordEmbedding("o");
    for (auto item: missingEmbedding) {
        BOOST_REQUIRE(item == 0.0);
    }
}

BOOST_AUTO_TEST_CASE(invalidDimensionThrows)
{
    Builder builder(15, Storage_Full);

    BOOST_CHECK_THROW(
        builder.addWord("the", {0.0, 1.0, 2.0}),
        std::runtime_error);
}

BOOST_AUTO_TEST_CASE(duplicateWordThrows)
{
    Builder builder(3, Storage_Full);
    builder.addWord("the", {0.0, 1.0, 2.0});

    BOOST_CHECK_THROW(
        builder.addWord("the", {2.0, 1.0, 2.0}),
        std::runtime_error);
}
