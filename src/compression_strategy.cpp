#include "full_compression.h"
#include "ubyte_compression.h"
#include "huffman_compression.h"

#include <boost/format.hpp>

namespace memb {

namespace {

const std::string INVALID_STRATEGY_TEMPLATE = "Storage strategy %s is not supported";

const std::vector<std::shared_ptr<CompressionStrategy>>& compressionStrategies() {
    static const std::vector<std::shared_ptr<CompressionStrategy>> strategies = {
        std::make_shared<FullCompressionStrategy>(),
        std::make_shared<UbyteCompressionStrategy>(),
        std::make_shared<HuffmanCompressionStrategy>(),
    };

    return strategies;
}

} // namespace

std::shared_ptr<CompressionStrategy> createCompressionStrategy(wire::Storage storage)
{
    const std::vector<std::shared_ptr<CompressionStrategy>>& strategies = compressionStrategies();
    auto result = std::find_if(
        strategies.begin(),
        strategies.end(),
        [storage](std::shared_ptr<CompressionStrategy> strategy)
        {
            return strategy->storageType() == storage;
        });

    if (result == strategies.end()) {
        throw std::runtime_error(boost::str(
            boost::format(INVALID_STRATEGY_TEMPLATE) % std::to_string(storage)));
    } else {
        return *result;
    }
}

std::shared_ptr<CompressionStrategy> createCompressionStrategy(const std::string& name)
{
    const std::vector<std::shared_ptr<CompressionStrategy>>& strategies = compressionStrategies();
    auto result = std::find_if(
        strategies.begin(),
        strategies.end(),
        [&name](std::shared_ptr<CompressionStrategy> strategy)
        {
            return strategy->storageName() == name;
        });

    if (result == strategies.end()) {
        throw std::runtime_error(boost::str(
            boost::format(INVALID_STRATEGY_TEMPLATE) % name));
    } else {
        return *result;
    }
}

std::vector<std::string> availableCompressionStrategies()
{
    const std::vector<std::shared_ptr<CompressionStrategy>>& strategies = compressionStrategies();
    std::vector<std::string> result;

    std::transform(
        strategies.begin(),
        strategies.end(),
        std::back_inserter(result),
        [](std::shared_ptr<CompressionStrategy> strategy)
        {
            return strategy->storageName();
        });

    return result;
}

}
