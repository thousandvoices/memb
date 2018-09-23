#include "kmeans.h"

#include <boost/test/unit_test.hpp>

using namespace memb;

BOOST_AUTO_TEST_SUITE(kMeans)

BOOST_AUTO_TEST_CASE(kMeansWork)
{
    std::vector<float> data;
    std::vector<uint8_t> expectedClusterIds;

    const size_t MIDDLE_CLUSTER_SIZE = 8;
    const size_t FIRST_CLUSTER_SIZE = 16;
    const size_t LAST_CLUSTER_SIZE = 4;

    for (size_t i = 0; i < MIDDLE_CLUSTER_SIZE; ++i) {
        data.push_back(-0.5 + i * 0.125);
        expectedClusterIds.push_back(1);
    }

    for (size_t i = 0; i < FIRST_CLUSTER_SIZE; ++i) {
        data.push_back(-9 + i * 0.125);
        expectedClusterIds.push_back(0);
    }

    for (size_t i = 0; i < LAST_CLUSTER_SIZE; ++i) {
        data.push_back(11.75 + i * 0.125);
        expectedClusterIds.push_back(2);
    }

    KMeansClusterizer clusterizer(3);
    clusterizer.fit(data);

    auto clusterIds = clusterizer.predict(data);
    BOOST_REQUIRE(clusterIds == expectedClusterIds);
}

BOOST_AUTO_TEST_SUITE_END()
