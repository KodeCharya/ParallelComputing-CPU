#pragma once

#include <string_view>

namespace parallel_compute {

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;
inline constexpr std::string_view kVersionString = "0.1.0";

#ifndef PARALLEL_COMPUTE_COMPILER_ID
#define PARALLEL_COMPUTE_COMPILER_ID "Unknown"
#endif

#ifndef PARALLEL_COMPUTE_COMPILER_VERSION
#define PARALLEL_COMPUTE_COMPILER_VERSION "Unknown"
#endif

#ifndef PARALLEL_COMPUTE_BUILD_TYPE
#define PARALLEL_COMPUTE_BUILD_TYPE "Unknown"
#endif

inline constexpr std::string_view kCompilerId = PARALLEL_COMPUTE_COMPILER_ID;
inline constexpr std::string_view kCompilerVersion = PARALLEL_COMPUTE_COMPILER_VERSION;
inline constexpr std::string_view kBuildType = PARALLEL_COMPUTE_BUILD_TYPE;

} // namespace parallel_compute
