#include <gtest/gtest.h>
#include "parallel_compute/system/cpu_features.hpp"
#include "parallel_compute/system/system_info.hpp"

using namespace parallel_compute;

TEST(CpuFeaturesTest, DetectsVendorAndBrand) {
    const auto& features = CpuFeatures::get();
    EXPECT_FALSE(features.vendor.empty());
    EXPECT_FALSE(features.brand.empty());
    EXPECT_FALSE(features.to_string().empty());
}

TEST(SystemInfoTest, DetectsCoresAndCache) {
    const auto& sys = SystemInfo::get();
    EXPECT_GE(sys.logical_cores, 1u);
    EXPECT_GE(sys.physical_cores, 1u);
    EXPECT_GE(sys.cache.l1d_bytes, 1024u);
    EXPECT_GE(sys.cache.l2_bytes, 1024u);
    EXPECT_GE(sys.cache.l3_bytes, 1024u);

    std::string json = sys.to_json();
    EXPECT_NE(json.find("\"cpu_model\""), std::string::npos);
    EXPECT_NE(json.find("\"cache\""), std::string::npos);
    EXPECT_NE(json.find("\"logical_cores\""), std::string::npos);
}
