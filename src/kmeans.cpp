#include "kmeans.h"

#include <algorithm>
#include <unordered_map>

namespace memb {

namespace {

const std::string NOT_FITTED_MESSAGE = "Attempt to use KMeansClusterizer before fitting";
const size_t MAX_ITERATIONS = 30;
const double SMALL_CLUSTER_FACTOR = 128;

} // namespace

KMeansClusterizer::KMeansClusterizer(size_t dim):
    dim_(dim)
{}

KMeansClusterizer::KMeansClusterizer(const std::vector<float>& centroids):
    dim_(centroids.size()),
    centroids_(centroids)
{}

void KMeansClusterizer::fit(const std::vector<float>& data)
{
    auto minMaxValues = std::minmax_element(data.begin(), data.end());
    float minValue = *(minMaxValues.first);
    float maxValue = *(minMaxValues.second);

    for (size_t centroidIdx = 0; centroidIdx < dim_; ++centroidIdx) {
        centroids_.push_back(minValue + centroidIdx / static_cast<float>(dim_ - 1) * (maxValue - minValue));
    }

    for (size_t epoch = 0; epoch < MAX_ITERATIONS; ++epoch) {
        updateCentroids(data, predict(data));
    }

    auto assignments = predict(data);
    std::unordered_map<uint8_t, size_t> counts;
    for (auto assignment : assignments) {
        counts[assignment] += 1;
    }

    auto maxCount = std::numeric_limits<size_t>::min();
    for (const auto& count : counts) {
        maxCount = std::max(count.second, maxCount);
    }

    auto smallClusterSizeLimit = maxCount / SMALL_CLUSTER_FACTOR;
    auto centroidsStart = &*centroids_.begin();

    centroids_.erase(
        std::remove_if(
            centroids_.begin(),
            centroids_.end(),
            [&counts, &centroidsStart, smallClusterSizeLimit](const float& centroid)
            {
                return counts[&centroid - centroidsStart] < smallClusterSizeLimit;
            }),
        centroids_.end()
    );

    updateCentroids(data, predict(data));
}

std::vector<uint8_t> KMeansClusterizer::predict(const std::vector<float>& data) const
{
    if (centroids_.size() == 0) {
        throw std::runtime_error(NOT_FITTED_MESSAGE);
    }

    std::vector<uint8_t> result;

    for (auto value : data) {
        auto pivot = std::lower_bound(centroids_.begin(), centroids_.end(), value);
        
        bool usePivot = (pivot == centroids_.begin()) 
            || ((pivot != centroids_.end()) && (*pivot - value < value - *(pivot - 1)));

        if (usePivot) {
            result.push_back(pivot - centroids_.begin());
        } else {
            result.push_back(pivot - 1 - centroids_.begin());
        }
    }

    return result;
}

std::vector<float> KMeansClusterizer::centroids() const
{
    return centroids_;
}

void KMeansClusterizer::updateCentroids(
    const std::vector<float>& data, const std::vector<uint8_t>& assignments)
{
    std::vector<size_t> counts(centroids_.size(), 0);
    centroids_.assign(centroids_.size(), 0);

    for (size_t i = 0; i < data.size(); ++i) {
        auto currentAssignment = assignments[i];
        auto currentCount = static_cast<float>(counts[currentAssignment]);
        centroids_[currentAssignment] =
            currentCount / (currentCount + 1) * centroids_[currentAssignment] + 1 / (currentCount + 1) * data[i];
        counts[currentAssignment] += 1;
    }

    std::sort(centroids_.begin(), centroids_.end());
}

flatbuffers::Offset<wire::KMeansClusterizer> KMeansClusterizer::save(
    flatbuffers::FlatBufferBuilder& builder) const
{
    auto serializedCentroids = builder.CreateVector(centroids_);
    return wire::CreateKMeansClusterizer(builder, serializedCentroids);
}

KMeansClusterizer KMeansClusterizer::load(const wire::KMeansClusterizer* serialized)
{
    std::vector<float> centroids(
        serialized->centroids()->begin(),
        serialized->centroids()->end());

    return KMeansClusterizer(centroids);
}

}
