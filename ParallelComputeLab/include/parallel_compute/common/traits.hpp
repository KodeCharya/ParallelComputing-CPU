#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace parallel_compute {

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename T>
concept Integral = std::is_integral_v<T>;

} // namespace parallel_compute
