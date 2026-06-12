// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <immintrin.h>
#include <cstring>

#include <iostream>

#include "../defines.hpp"
#include "../utils/memory.hpp"

FORCE_INLINE void accumulate_one_block(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint16_t* __restrict__ result,
    size_t D
) {
    size_t TOTAL = D << 2;  // FAST_SIZE(32) * D / 8
#if defined(__AVX512F__)
    __m512i c, lo, hi, lut, res_lo, res_hi;

    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    for (size_t i = 0; i < TOTAL; i += 128) {
        c = _mm512_load_si512(&codes[i]);
        lut = _mm512_load_si512(&LUT[i]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));

        c = _mm512_load_si512(&codes[i + 64]);
        lut = _mm512_load_si512(&LUT[i + 64]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    __m256i res0 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu0), _mm512_extracti64x4_epi64(accu0, 1)
    );
    __m256i res1 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu1), _mm512_extracti64x4_epi64(accu1, 1)
    );

    res0 = _mm256_sub_epi16(res0, _mm256_slli_epi16(res1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(res0, res1, 0x21), _mm256_blend_epi32(res0, res1, 0xF0)
    );
    _mm256_store_si256((__m256i*)result, dis0);

    __m256i res2 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu2), _mm512_extracti64x4_epi64(accu2, 1)
    );
    __m256i res3 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu3), _mm512_extracti64x4_epi64(accu3, 1)
    );

    res2 = _mm256_sub_epi16(res2, _mm256_slli_epi16(res3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(res2, res3, 0x21), _mm256_blend_epi32(res2, res3, 0xF0)
    );
    _mm256_store_si256((__m256i*)&result[16], dis1);

#elif defined(__AVX2__)
    __m256i c, lo, hi, lut, res_lo, res_hi;

    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (size_t i = 0; i < TOTAL; i += 64) {
        c = _mm256_load_si256(&codes[i]);
        lut = _mm256_load_si256(&LUT[i]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c = _mm256_load_si256(&codes[i + 32]);
        lut = _mm256_load_si256(&LUT[i + 32]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0)
    );
    _mm256_store_si256((__m256i*)result, dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0)
    );
    _mm256_store_si256((__m256i*)&result[16], dis1);
#else
    std::cerr << "NO AVX SIMD SUPPORTED!\n";
    abort();
#endif
}

FORCE_INLINE void accumulate_robust(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint32_t* __restrict__ result,
    size_t D
) {
    size_t TOTAL = D << 2;  // FAST_SIZE(32) * D / 8
#if defined(__AVX512F__)
    __m512i c, lo, hi, lut, res_lo, res_hi;

    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    for (size_t i = 0; i < TOTAL; i += 128) {
        c = _mm512_load_si512(&codes[i]);
        lut = _mm512_load_si512(&LUT[i]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));

        c = _mm512_load_si512(&codes[i + 64]);
        lut = _mm512_load_si512(&LUT[i + 64]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    __m256i res0 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu0), _mm512_extracti64x4_epi64(accu0, 1)
    );
    __m256i res1 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu1), _mm512_extracti64x4_epi64(accu1, 1)
    );

    res0 = _mm256_sub_epi16(res0, _mm256_slli_epi16(res1, 8));
    __m512i dis0 = _mm512_add_epi32(
        _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(res0, res1, 0x21)),
        _mm512_cvtepu16_epi32(_mm256_blend_epi32(res0, res1, 0xF0))
    );
    _mm512_store_si512(result, dis0);

    __m256i res2 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu2), _mm512_extracti64x4_epi64(accu2, 1)
    );
    __m256i res3 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu3), _mm512_extracti64x4_epi64(accu3, 1)
    );

    res2 = _mm256_sub_epi16(res2, _mm256_slli_epi16(res3, 8));

    __m512i dis1 = _mm512_add_epi32(
        _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(res2, res3, 0x21)),
        _mm512_cvtepu16_epi32(_mm256_blend_epi32(res2, res3, 0xF0))
    );
    _mm512_store_si512(&result[16], dis1);
#endif
}

inline uint32_t accumulate_one_block_high_acc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    float delta,
    int shift,
    float* __restrict__ ip_xb_qprime,
    size_t D
) {
    __m512i low_mask = _mm512_set1_epi8(0xf);
    __m512i accu[2][4];
    for (size_t _ = 0; _ < 2; _++)
        for (size_t i = 0; i < 4; i++)
            accu[_][i] = _mm512_setzero_si512();

    size_t M = D >> 2;

    // std::cerr << "FastScan YES!" << std::endl;
    for (size_t m = 0; m < M; m += 4) {
        __m512i c = _mm512_load_si512(codes);
        __m512i lo = _mm512_and_si512(c, low_mask);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);

        for (size_t _ = 0; _ < 2; _++) {
            __m512i lut = _mm512_load_si512(LUT);

            __m512i res_lo = _mm512_shuffle_epi8(lut, lo);
            __m512i res_hi = _mm512_shuffle_epi8(lut, hi);

            accu[_][0] = _mm512_add_epi16(accu[_][0], res_lo);
            accu[_][1] = _mm512_add_epi16(accu[_][1], _mm512_srli_epi16(res_lo, 8));

            accu[_][2] = _mm512_add_epi16(accu[_][2], res_hi);
            accu[_][3] = _mm512_add_epi16(accu[_][3], _mm512_srli_epi16(res_hi, 8));

            LUT += 64;
        }
        codes += 64;
    }

    __m512i res[2];
    __m512i dis0[2], dis1[2];

    for (size_t _ = 0; _ < 2; _++) {
        __m256i tmp0 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][0]), _mm512_extracti64x4_epi64(accu[_][0], 1)
        );
        __m256i tmp1 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][1]), _mm512_extracti64x4_epi64(accu[_][1], 1)
        );
        tmp0 = _mm256_sub_epi16(tmp0, _mm256_slli_epi16(tmp1, 8));

        dis0[_] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp0, tmp1, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp0, tmp1, 0xF0))
        );

        __m256i tmp2 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][2]), _mm512_extracti64x4_epi64(accu[_][2], 1)
        );
        __m256i tmp3 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][3]), _mm512_extracti64x4_epi64(accu[_][3], 1)
        );
        tmp2 = _mm256_sub_epi16(tmp2, _mm256_slli_epi16(tmp3, 8));

        dis1[_] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp2, tmp3, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp2, tmp3, 0xF0))
        );
    }
    // shift res of high, add res of low
    res[0] = _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8));
    res[1] = _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8));

    __m512i simd_shift = _mm512_set1_epi32(shift);
    __m512 simd_delta = _mm512_set1_ps(delta);
    for (size_t i = 0; i < 2; i++) {
        res[i] = _mm512_add_epi32(res[i], simd_shift);
        __m512 tmp = _mm512_cvtepi32_ps(res[i]);
        tmp = _mm512_mul_ps(tmp, simd_delta);
        _mm512_store_ps(ip_xb_qprime, tmp);
        ip_xb_qprime += 16;
    }
    return 0xFFFFFFFF;
}


inline float calculate_ip_xb_qprime_single(
    const int16_t* quant_query,
    const uint8_t* short_code, 
    size_t D,
    float delta
) {
    // Position mapping for 4-bit patterns (same as in the original code)
    constexpr int pos[16] = {
        3 /*0000*/, 3 /*0001*/, 2 /*0010*/, 3 /*0011*/,
        1 /*0100*/, 3 /*0101*/, 2 /*0110*/, 3 /*0111*/,
        0 /*1000*/, 3 /*1001*/, 2 /*1010*/, 3 /*1011*/,
        1 /*1100*/, 3 /*1101*/, 2 /*1110*/, 3 /*1111*/,
    };
    
    size_t M = D >> 2;  // Number of 4-dimensional blocks
    int accumulated_sum = 0;
    int total_shift = 0;
    
    const int16_t* quan_query_ptr = quant_query;
    size_t code_byte_idx = 0;
    int bit_offset = 0;
    
    // Process each 4-dimensional block
    for (size_t i = 0; i < M; i++) {
        // Build lookup table for current 4D block
        int LUT[16];
        int v_min = 0;
        
        LUT[0] = 0;
        for (int j = 1; j < 16; j++) {
            LUT[j] = LUT[j - lowbit(j)] + quan_query_ptr[pos[j]];
            v_min = (LUT[j] < v_min) ? LUT[j] : v_min;
        }
        
        // Extract 4-bit pattern from short_code
        uint8_t pattern_4bit = 0;
        for (int bit = 0; bit < 4; bit++) {
            // Get bit from packed short_code
            int global_bit_idx = i * 4 + bit;
            int byte_idx = global_bit_idx / 8;
            int bit_in_byte = global_bit_idx % 8;
            
            uint8_t bit_value = (short_code[byte_idx] >> (7 - bit_in_byte)) & 1;
            pattern_4bit |= (bit_value << (3 - bit));  // MSB first
        }
        
        // Look up the value and accumulate
        accumulated_sum += (LUT[pattern_4bit] - v_min);
        total_shift += v_min;
        
        quan_query_ptr += 4;  // Move to next 4D block
    }
    
    // Apply final scaling: (accumulated_sum + shift) * delta
    float result = (float)(accumulated_sum + total_shift) * delta;
    
    return result;
}

FORCE_INLINE uint32_t reverse_bits8(uint32_t x) {
    x = ((x & 0xF0U) >> 4) | ((x & 0x0FU) << 4);
    x = ((x & 0xCCU) >> 2) | ((x & 0x33U) << 2);
    x = ((x & 0xAAU) >> 1) | ((x & 0x55U) << 1);
    return x;
}

inline float calculate_ip_xb_qprime_single_scalar(
    const int16_t* quant_query,
    const uint8_t* short_code, 
    size_t D,
    float delta
) {
    int accumulated_sum = 0;

#if defined(__AVX512BW__) && defined(__AVX512F__)
    size_t d = 0;
    for (; d + 32 <= D; d += 32) {
        const uint32_t mask_bits =
            reverse_bits8(static_cast<uint32_t>(short_code[0])) |
            (reverse_bits8(static_cast<uint32_t>(short_code[1])) << 8) |
            (reverse_bits8(static_cast<uint32_t>(short_code[2])) << 16) |
            (reverse_bits8(static_cast<uint32_t>(short_code[3])) << 24);

        __m512i q16 = _mm512_loadu_si512(reinterpret_cast<const void*>(quant_query + d));
        __m512i selected16 = _mm512_maskz_mov_epi16(static_cast<__mmask32>(mask_bits), q16);

        __m256i selected16_lo = _mm512_castsi512_si256(selected16);
        __m256i selected16_hi = _mm512_extracti64x4_epi64(selected16, 1);
        __m512i selected32_lo = _mm512_cvtepi16_epi32(selected16_lo);
        __m512i selected32_hi = _mm512_cvtepi16_epi32(selected16_hi);

        accumulated_sum += _mm512_reduce_add_epi32(selected32_lo);
        accumulated_sum += _mm512_reduce_add_epi32(selected32_hi);
        short_code += 4;
    }

    if (d == D) {
        return float(accumulated_sum) * delta;
    }

    quant_query += d;
    D -= d;
#endif

    // D is assumed to be a multiple of 4 (same assumption as original)
    size_t M = D >> 2; // number of 4D blocks
    const int16_t* q = quant_query;

    for (size_t i = 0; i < M; ++i) {
        // extract nibble for this block: upper nibble for even i, lower for odd i
        const uint8_t byte_val = short_code[i >> 1];
        const uint8_t p = ( (i & 1) == 0 ) ? (byte_val >> 4) : (byte_val & 0x0F);

        // Bits: p[3] p[2] p[1] p[0] correspond to q[0], q[1], q[2], q[3]
        // Turn each bit into a mask of 0 or -1, then AND with the signed value.
        const int m0 = -int((p >> 3) & 1);
        const int m1 = -int((p >> 2) & 1);
        const int m2 = -int((p >> 1) & 1);
        const int m3 = -int((p >> 0) & 1);

        // Mask-and-add in 32-bit to avoid overflow
        accumulated_sum += (int(q[0]) & m0)
                         + (int(q[1]) & m1)
                         + (int(q[2]) & m2)
                         + (int(q[3]) & m3);

        q += 4;
    }

    return float(accumulated_sum) * delta;
}