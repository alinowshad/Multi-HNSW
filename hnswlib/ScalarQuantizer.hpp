#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <immintrin.h>

#include "defines.hpp"

class ScalarInt8Quantizer {
   private:
    size_t DIM{0};
    std::vector<float> scales_;

   public:
    explicit ScalarInt8Quantizer(size_t dim)
        : DIM(dim), scales_(dim, 1.0f) {}

    explicit ScalarInt8Quantizer() = default;

    size_t dim() const { return DIM; }

    const std::vector<float>& scales() const { return scales_; }

    void set_scales(const std::vector<float>& scales) {
        DIM = scales.size();
        scales_ = scales;
    }

    void train(const float* data, size_t num_points) {
        if (DIM == 0) {
            throw std::runtime_error("ScalarInt8Quantizer dimension is zero");
        }

        scales_.assign(DIM, 1.0f);
        for (size_t d = 0; d < DIM; ++d) {
            float max_abs = 0.0f;
            for (size_t i = 0; i < num_points; ++i) {
                float value = std::fabs(data[i * DIM + d]);
                max_abs = std::max(max_abs, value);
            }
            scales_[d] = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        }
    }

    void quantizeVector(const float* data, int8_t* code) const {
        for (size_t d = 0; d < DIM; ++d) {
            float scaled = data[d] / scales_[d];
            int value = static_cast<int>(std::lrint(scaled));
            value = std::max(-127, std::min(127, value));
            code[d] = static_cast<int8_t>(value);
        }
    }

    void prepareQuery(const float* query, float* scaled_query) const {
        for (size_t d = 0; d < DIM; ++d) {
            scaled_query[d] = query[d] * scales_[d];
        }
    }
};

FORCE_INLINE float scalar_int8_inner_product(
    const float* scaled_query,
    const int8_t* code,
    size_t dim
) {
    float result = 0.0f;
#if defined(__AVX512BW__) && defined(__AVX512F__)
    size_t d = 0;
    __m512 sum = _mm512_setzero_ps();
    for (; d + 32 <= dim; d += 32) {
        __m128i code8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + d));
        __m128i code8_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + d + 16));
        __m512i code32_lo = _mm512_cvtepi8_epi32(code8);
        __m512i code32_hi = _mm512_cvtepi8_epi32(code8_hi);
        __m512 codef_lo = _mm512_cvtepi32_ps(code32_lo);
        __m512 codef_hi = _mm512_cvtepi32_ps(code32_hi);
        __m512 q_lo = _mm512_loadu_ps(scaled_query + d);
        __m512 q_hi = _mm512_loadu_ps(scaled_query + d + 16);
        sum = _mm512_fmadd_ps(q_lo, codef_lo, sum);
        sum = _mm512_fmadd_ps(q_hi, codef_hi, sum);
    }
    result += _mm512_reduce_add_ps(sum);
    for (; d < dim; ++d) {
        result += scaled_query[d] * static_cast<float>(code[d]);
    }
#else
    for (size_t d = 0; d < dim; ++d) {
        result += scaled_query[d] * static_cast<float>(code[d]);
    }
#endif
    return result;
}
