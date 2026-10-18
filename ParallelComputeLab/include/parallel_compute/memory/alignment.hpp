#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace parallel_compute {

inline constexpr std::size_t kCacheLineSize = 64;
inline constexpr std::size_t kDefaultSimdAlignment = 64;

template <typename T>
[[nodiscard]] constexpr bool is_aligned(const T* ptr, std::size_t alignment = kDefaultSimdAlignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

template <std::size_t Alignment = kDefaultSimdAlignment, typename T>
[[nodiscard]] constexpr T* assume_aligned(T* ptr) noexcept {
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 4)
    return static_cast<T*>(__builtin_assume_aligned(ptr, Alignment));
#else
    return ptr;
#endif
}

template <std::size_t Alignment = kDefaultSimdAlignment, typename T>
[[nodiscard]] constexpr const T* assume_aligned(const T* ptr) noexcept {
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 4)
    return static_cast<const T*>(__builtin_assume_aligned(ptr, Alignment));
#else
    return ptr;
#endif
}

} // namespace parallel_compute
