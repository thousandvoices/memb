#include "builder.h"
#include "reader.h"
#include "trained_compression.h"

#include <sstream>
#include <fstream>

#define BOOST_AUTO_TEST_MAIN
#include <boost/test/unit_test.hpp>

using namespace memb;

const std::string STORAGE_FILENAME = "data.bin";

struct WordVector {
    std::string word;
    std::vector<float> embedding;
};

void builderTestImpl(wire::Storage storageType, std::shared_ptr<CompressionStrategy> compression)
{
    std::vector<WordVector> wordVectors = {
        {"the", {0.0, 1.0, 2.0}},
        {"of", {0.0, -1.0, 2.0}},
        {"th", {2.0, 0.0, 1.0}},
        {"a", {1.0, 0.0, -2.0}},
        {"tho", {2.0, 0.0, -1.0}},
        {"abc", {-2.0, 0.0, 1.0}},
    };
    std::vector<std::string> expectedKeys;

    Builder builder(3, storageType, 8);
    for (const auto& wordVector : wordVectors) {
        builder.addWord(wordVector.word, wordVector.embedding);
        expectedKeys.push_back(wordVector.word);
    }
    std::sort(expectedKeys.begin(), expectedKeys.end());

    builder.save(STORAGE_FILENAME);

    Reader reader(STORAGE_FILENAME, compression);
    BOOST_REQUIRE(expectedKeys == reader.keys());

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

BOOST_AUTO_TEST_SUITE(compressionStrategies)

BOOST_AUTO_TEST_CASE(fullBuilderWorks)
{
    builderTestImpl(wire::Storage_Full, createCompressionStrategy(wire::Storage_Full));
}

BOOST_AUTO_TEST_CASE(uniformBuilderWorks)
{
    builderTestImpl(wire::Storage_Uniform, createCompressionStrategy(wire::Storage_Uniform));
}

BOOST_AUTO_TEST_CASE(trainedBuilderWorks)
{
    builderTestImpl(wire::Storage_Trained, createCompressionStrategy(wire::Storage_Trained));
}

class TestTrainedCompressionStrategy : public TrainedCompressionStrategy {
public:
    virtual std::shared_ptr<CompressedStorage> createCompressedStorage(
        const void* flatIndex, size_t dim) const override
    {
        return std::make_shared<TrainedCompressedStorage>(flatIndex, dim, 1);
    }
};

BOOST_AUTO_TEST_CASE(trainedBuilderWorksWithIndirectDecoder)
{
    builderTestImpl(wire::Storage_Trained, std::make_shared<TestTrainedCompressionStrategy>());
}

BOOST_AUTO_TEST_CASE(invalidDimensionThrows)
{
    Builder builder(15, wire::Storage_Full, 8);

    BOOST_CHECK_THROW(
        builder.addWord("the", {0.0, 1.0, 2.0}),
        std::runtime_error);
}

BOOST_AUTO_TEST_CASE(duplicateWordThrows)
{
    Builder builder(3, wire::Storage_Full, 8);
    builder.addWord("the", {0.0, 1.0, 2.0});

    BOOST_CHECK_THROW(
        builder.addWord("the", {2.0, 1.0, 2.0}),
        std::runtime_error);
}

BOOST_AUTO_TEST_CASE(missingFileThrows)
{
    BOOST_CHECK_THROW(
        Reader("missing.bin"),
        std::exception);
}

BOOST_AUTO_TEST_CASE(invalidFileThrows)
{
    static const std::string INVALID_FILE = "invalid.bin";

    {
        std::ofstream f(INVALID_FILE);
        f << "0123456789";
    }

    BOOST_CHECK_THROW(
        auto reader = Reader(INVALID_FILE),
        std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
