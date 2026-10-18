#pragma once

#include "parallel_compute/common/error.hpp"
#include "parallel_compute/common/traits.hpp"
#include "parallel_compute/memory/alignment.hpp"

#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <vector>

namespace parallel_compute {

template <typename T, std::size_t Alignment = kDefaultSimdAlignment>
class AlignedAllocator {
public:
    using value_type = T;
    static constexpr std::size_t alignment = Alignment;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::size_t(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        std::size_t bytes = n * sizeof(T);
        // Ensure byte size is multiple of alignment for std::aligned_alloc
        std::size_t rem = bytes % Alignment;
        if (rem != 0) {
            bytes += (Alignment - rem);
        }
        void* ptr = nullptr;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L || defined(__APPLE__) || defined(__linux__)
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#else
        ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

template <typename T, std::size_t Alignment = kDefaultSimdAlignment>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

template <Numeric T>
class MatrixView {
public:
    constexpr MatrixView(T* data, std::size_t rows, std::size_t cols, std::size_t lda) noexcept
        : data_(data), rows_(rows), cols_(cols), lda_(lda) {}

    constexpr MatrixView(T* data, std::size_t rows, std::size_t cols) noexcept
        : data_(data), rows_(rows), cols_(cols), lda_(cols) {}

    [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr std::size_t lda() const noexcept { return lda_; }
    [[nodiscard]] constexpr T* data() noexcept { return data_; }
    [[nodiscard]] constexpr const T* data() const noexcept { return data_; }

    [[nodiscard]] constexpr T& operator()(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw DimensionMismatchError("MatrixView index out of bounds");
        }
        return data_[r * lda_ + c];
    }

    [[nodiscard]] constexpr const T& operator()(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw DimensionMismatchError("MatrixView index out of bounds");
        }
        return data_[r * lda_ + c];
    }

    [[nodiscard]] constexpr T* row_ptr(std::size_t r) noexcept {
        return data_ + r * lda_;
    }

    [[nodiscard]] constexpr const T* row_ptr(std::size_t r) const noexcept {
        return data_ + r * lda_;
    }

private:
    T* data_{nullptr};
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::size_t lda_{0};
};

template <Numeric T>
class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, T initial_value = T{})
        : rows_(rows), cols_(cols), lda_(cols), storage_(rows * cols, initial_value) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t lda() const noexcept { return lda_; }
    [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }

    [[nodiscard]] T* data() noexcept { return storage_.data(); }
    [[nodiscard]] const T* data() const noexcept { return storage_.data(); }

    [[nodiscard]] T& operator()(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw DimensionMismatchError("Matrix index out of bounds");
        }
        return storage_[r * lda_ + c];
    }

    [[nodiscard]] const T& operator()(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw DimensionMismatchError("Matrix index out of bounds");
        }
        return storage_[r * lda_ + c];
    }

    [[nodiscard]] T* row_ptr(std::size_t r) noexcept {
        return storage_.data() + r * lda_;
    }

    [[nodiscard]] const T* row_ptr(std::size_t r) const noexcept {
        return storage_.data() + r * lda_;
    }

    [[nodiscard]] MatrixView<T> view() noexcept {
        return MatrixView<T>(storage_.data(), rows_, cols_, lda_);
    }

    [[nodiscard]] MatrixView<const T> view() const noexcept {
        return MatrixView<const T>(storage_.data(), rows_, cols_, lda_);
    }

    void fill(T value) {
        std::fill(storage_.begin(), storage_.end(), value);
    }

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::size_t lda_{0};
    AlignedVector<T> storage_{};
};

} // namespace parallel_compute
