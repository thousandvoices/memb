#include "huffman_encoder.h"
#include "bit_stream.h"

#include <queue>

namespace memb {

namespace {

struct TreeNode {
    uint8_t key;
    size_t count;
    std::shared_ptr<TreeNode> left;
    std::shared_ptr<TreeNode> right;
};

struct TreeNodeComparator {
    bool operator()(std::shared_ptr<TreeNode>& lhs, std::shared_ptr<TreeNode>& rhs) const
    {
        return lhs->count > rhs->count;
    }
};

void aggregatePrefixCodes(
    const std::shared_ptr<TreeNode>& node,
    size_t length,
    std::vector<CodeInfo>* codeLengths)
{
    if (node->left && node->right) {
        aggregatePrefixCodes(
            node->left,
            length + 1,
            codeLengths);
        aggregatePrefixCodes(
            node->right,
            length + 1,
            codeLengths);
    } else {
        codeLengths->push_back({node->key, length});
    }
}

std::vector<CodeInfo> calculatePrefixCodeLengths(
    const std::unordered_map<uint8_t, size_t>& valueCounts)
{
    std::priority_queue<
        std::shared_ptr<TreeNode>,
        std::vector<std::shared_ptr<TreeNode>>,
        TreeNodeComparator
    > mergedCounts;

    for (const auto& count : valueCounts) {
        mergedCounts.push(std::make_shared<TreeNode>(
            TreeNode{count.first, count.second, nullptr, nullptr}));
    }

    while (mergedCounts.size() > 1) {
        auto newLeftNode = mergedCounts.top();
        mergedCounts.pop();
        auto newRightNode = mergedCounts.top();
        mergedCounts.pop();

        mergedCounts.push(std::make_shared<TreeNode>(
            TreeNode{0, newLeftNode->count + newRightNode->count, newLeftNode, newRightNode}));
    }

    auto rootNode = mergedCounts.top();
    std::vector<CodeInfo> codeLengths;
    aggregatePrefixCodes(rootNode, 0, &codeLengths);

    return codeLengths;
}

} // namespace

HuffmanEncoder::HuffmanEncoder(const std::unordered_map<uint8_t, size_t>& counts):
    codeLengths_(calculatePrefixCodeLengths(counts))
{
    std::sort(
        codeLengths_.begin(),
        codeLengths_.end(),
        [](const CodeInfo& lhs, const CodeInfo& rhs)
        {
            return lhs.length < rhs.length;
        });

    codebook_ = createCanonicalPrefixCodes(codeLengths_);
}

std::vector<uint8_t> HuffmanEncoder::encode(const std::vector<uint8_t>& data) const
{
    BitStream valuesStream;
    for (const auto& value : data) {
        valuesStream.push(codebook_.at(value));
    }

    return valuesStream.data();
}
    
HuffmanDecoder HuffmanEncoder::createDecoder() const
{
    std::vector<uint8_t> keys;
    std::vector<uint32_t> sizeOffsets;
    size_t currentSize = 0;

    for (size_t i = 0; i < codeLengths_.size(); ++i) {
        while (currentSize < codeLengths_[i].length) {
            ++currentSize;
            sizeOffsets.push_back(i);
        }

        keys.push_back(codeLengths_[i].key);
    }
    sizeOffsets.push_back(codeLengths_.size());

    return HuffmanDecoder(keys, sizeOffsets);
}

void HuffmanEncoderBuilder::updateFrequencies(const std::vector<uint8_t>& data)
{
    for (auto value : data) {
        counts_[value] += 1;
    }
}
    
HuffmanEncoder HuffmanEncoderBuilder::createEncoder() const
{
    return HuffmanEncoder(counts_);
}

}
