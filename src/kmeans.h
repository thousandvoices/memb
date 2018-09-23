#pragma once

#include "kmeans_generated.h"

#include <vector>

namespace memb {

class KMeansClusterizer {
public:
    KMeansClusterizer(size_t dim);

    void fit(const std::vector<float>& data);

    std::vector<uint8_t> predict(const std::vector<float>& data) const;

    std::vector<float> centroids() const;

    flatbuffers::Offset<wire::KMeansClusterizer> save(flatbuffers::FlatBufferBuilder& builder) const;
    static KMeansClusterizer load(const wire::KMeansClusterizer* serialized);

private:
    KMeansClusterizer(const std::vector<float>& centroids);

    void updateCentroids(
        const std::vector<float>& data, const std::vector<uint8_t>& assignments);

    size_t dim_;
    std::vector<float> centroids_;
};

}
