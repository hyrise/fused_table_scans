#include <iostream>
#include <immintrin.h>
#include <bitset>
#include <random>
#include <algorithm>
#include <fstream>

#include "common.hpp"

#if REG == 128 && !AVX512
    const __m128 masks[] = {(__m128)(_mm_set_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff)),
                            (__m128)(_mm_set_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0x00000000)),
                            (__m128)(_mm_set_epi32(0xffffffff, 0xffffffff, 0x00000000, 0x00000000)),
                            (__m128)(_mm_set_epi32(0xffffffff, 0x00000000, 0x00000000, 0x00000000)),
                            (__m128)(_mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000))};

    #define __mXi __m128i
    #define _mmX_set1_epi32 _mm_set1_epi32
    #define _mmX_i32gather_epi32 _mm_i32gather_epi32
    #define _mmX_mask_cmpeq_epi32_mask(k, a, b) (_mm_movemask_ps((__m128)_mm_cmpeq_epi32(a, b)) & (k));
    #define _mmX_set_epi32(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) _mm_set_epi32(e3, e2, e1, e0)
    #define _mmX_loadu_siX _mm_loadu_si128
    #define _mmX_cmpeq_epi32_mask(a, b) _mm_movemask_ps((__m128)_mm_cmpeq_epi32(a, b))
    #define _mmX_permutexvar_epi32(a, b) [=](){switch((uint32_t)(a[1])) {\
                                                 case 0: return (__m128i)_mm_permute_ps((__m128)b, 0b11100100);\
                                                 case 1: return (__m128i)_mm_permute_ps((__m128)b, 0b10010000);\
                                                 case 2: return (__m128i)_mm_permute_ps((__m128)b, 0b01000000);\
                                                 case 3: return (__m128i)_mm_permute_ps((__m128)b, 0b00000000);\
                                                 default: throw std::runtime_error("unexpected case");\
                                               }}()
    #define _mmX_mask_compress_epi32(src, k, a) [=](){switch(k) {\
                                                        /* using no value from a */\
                                                        case 0b0000: return src;\
                                                        /* using one value from a */\
                                                        case 0b0001: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[1], (__m128)src), _mm_andnot_ps(masks[1], _mm_permute_ps((__m128)a, 0b00000000))));\
                                                        case 0b0010: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[1], (__m128)src), _mm_andnot_ps(masks[1], _mm_permute_ps((__m128)a, 0b00000001))));\
                                                        case 0b0100: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[1], (__m128)src), _mm_andnot_ps(masks[1], _mm_permute_ps((__m128)a, 0b00000010))));\
                                                        case 0b1000: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[1], (__m128)src), _mm_andnot_ps(masks[1], _mm_permute_ps((__m128)a, 0b00000011))));\
                                                        /* using two values from a */\
                                                        case 0b0011: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[2], (__m128)src), _mm_andnot_ps(masks[2], _mm_permute_ps((__m128)a, 0b00000100))));\
                                                        case 0b0101: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[2], (__m128)src), _mm_andnot_ps(masks[2], _mm_permute_ps((__m128)a, 0b00001000))));\
                                                        case 0b1001: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[2], (__m128)src), _mm_andnot_ps(masks[2], _mm_permute_ps((__m128)a, 0b00001100))));\
                                                        case 0b0110: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[2], (__m128)src), _mm_andnot_ps(masks[2], _mm_permute_ps((__m128)a, 0b00001001))));\
                                                        case 0b1010: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[2], (__m128)src), _mm_andnot_ps(masks[2], _mm_permute_ps((__m128)a, 0b00001101))));\
                                                        case 0b1100: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[2], (__m128)src), _mm_andnot_ps(masks[2], _mm_permute_ps((__m128)a, 0b00001110))));\
                                                        /* using three values from a */\
                                                        case 0b1110: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[3], (__m128)src), _mm_andnot_ps(masks[3], _mm_permute_ps((__m128)a, 0b00111001))));\
                                                        case 0b1101: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[3], (__m128)src), _mm_andnot_ps(masks[3], _mm_permute_ps((__m128)a, 0b00111000))));\
                                                        case 0b1011: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[3], (__m128)src), _mm_andnot_ps(masks[3], _mm_permute_ps((__m128)a, 0b00110100))));\
                                                        case 0b0111: return (__m128i)(_mm_or_ps(_mm_and_ps(masks[3], (__m128)src), _mm_andnot_ps(masks[3], _mm_permute_ps((__m128)a, 0b00100100))));\
                                                        /* using four values from a */\
                                                        case 0b1111: return a;\
                                                        default: throw std::runtime_error("unexpected case");\
                                                      }}()

    #define _mmX_add_epi32 _mm_add_epi32
    constexpr int pos_per_reg = 4;
#endif

#if REG == 128 && AVX512
    #define __mXi __m128i
    #define _mmX_set1_epi32 _mm_set1_epi32
    #define _mmX_i32gather_epi32 _mm_i32gather_epi32
    #define _mmX_mask_cmpeq_epi32_mask _mm_mask_cmpeq_epi32_mask
    #define _mmX_set_epi32(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) _mm_set_epi32(e3, e2, e1, e0)
    #define _mmX_loadu_siX _mm_loadu_si128
    #define _mmX_cmpeq_epi32_mask _mm_cmpeq_epi32_mask
    #define _mmX_permutexvar_epi32(a, b) _mm_permutex2var_epi32(b, a, b)
    #define _mmX_mask_compress_epi32 _mm_mask_compress_epi32
    #define _mmX_add_epi32 _mm_add_epi32
    constexpr int pos_per_reg = 4;
#endif

#if REG == 256 && !AVX512
    #error "256-bit registers are not supported without AVX-512 because translating 128-bit was already enough of a pain"
#endif

#if REG == 256 && AVX512
    #define __mXi __m256i
    #define _mmX_set1_epi32 _mm256_set1_epi32
    #define _mmX_i32gather_epi32 _mm256_i32gather_epi32
    #define _mmX_mask_cmpeq_epi32_mask _mm256_mask_cmpeq_epi32_mask
    #define _mmX_set_epi32(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0)
    #define _mmX_loadu_siX _mm256_loadu_si256
    #define _mmX_cmpeq_epi32_mask _mm256_cmpeq_epi32_mask
    #define _mmX_permutexvar_epi32(a, b) _mm256_permutevar8x32_epi32(b, a)
    #define _mmX_mask_compress_epi32 _mm256_mask_compress_epi32
    #define _mmX_add_epi32 _mm256_add_epi32
    constexpr int pos_per_reg = 8;
#endif

#if REG == 512 && !AVX512
    #error "512-bit registers are not supported without AVX-512"
#endif

#if REG == 512 && AVX512
    #define __mXi __m512i
    #define _mmX_set1_epi32 _mm512_set1_epi32
    #define _mmX_i32gather_epi32(a,b,c) _mm512_i32gather_epi32(b,a,c)
    #define _mmX_mask_cmpeq_epi32_mask _mm512_mask_cmpeq_epi32_mask
    #define _mmX_set_epi32(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) _mm512_set_epi32(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
    #define _mmX_loadu_siX _mm512_loadu_si512
    #define _mmX_cmpeq_epi32_mask _mm512_cmpeq_epi32_mask
    #define _mmX_permutexvar_epi32 _mm512_permutexvar_epi32
    #define _mmX_mask_compress_epi32 _mm512_mask_compress_epi32
    #define _mmX_add_epi32 _mm512_add_epi32
    constexpr int pos_per_reg = 16;
#endif

std::ostream& operator<<(std::ostream& o, const __m512i& v) {
#if AVX512
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 0), 0) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 0), 1) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 0), 2) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 0), 3) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 1), 0) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 1), 1) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 1), 2) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 1), 3) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 2), 0) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 2), 1) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 2), 2) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 2), 3) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 3), 0) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 3), 1) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 3), 2) << " ";
     o << _mm_extract_epi32(_mm512_extracti32x4_epi32(v, 3), 3) << " ";
#endif
    (void)v;
    return o;
}

int avx_second_scan(const __mXi pos_list, const int num_entries) {
    static auto scan_values = _mmX_set1_epi32(second_scan_value);

    auto values = _mmX_i32gather_epi32(&(col_b[0]), pos_list, sizeof(value_id_t));

    auto scan_result = _mmX_mask_cmpeq_epi32_mask((1 << num_entries) - 1, values, scan_values);

    return _mm_popcnt_u32(scan_result);
}

__mXi shift_shuffles[] = {
    _mmX_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mmX_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0),
    _mmX_set_epi32(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0),
    _mmX_set_epi32(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0),
    _mmX_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0),
    _mmX_set_epi32(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    _mmX_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
};

int avx_scan() {
    int total_results = 0;


    __mXi indexes = _mmX_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __mXi index_add = _mmX_set1_epi32(pos_per_reg);
    const auto scan_values = _mmX_set1_epi32(first_scan_value);

    __mXi pos_list = _mmX_set1_epi32(0);
    constexpr uint max_entries_in_pos_list = sizeof(pos_list) / sizeof(pos_t);
    uint free_entries_in_pos_list = max_entries_in_pos_list;

    for(size_t start_index = 0; start_index < col_a.size(); start_index += pos_per_reg) {
        auto values = _mmX_loadu_siX((__mXi*)&(col_a[start_index]));

        // do scan
        auto scan_result = _mmX_cmpeq_epi32_mask(values, scan_values);

        uint num_results = _mm_popcnt_u32(scan_result);

        if(num_results > free_entries_in_pos_list) {
            // doesn't fit into current pos_list anymore, perform scan in next column

            total_results += avx_second_scan(pos_list, max_entries_in_pos_list - free_entries_in_pos_list);

            free_entries_in_pos_list = max_entries_in_pos_list;
        }

        // move as many positions into current pos list as possible
        pos_list = _mmX_permutexvar_epi32(shift_shuffles[num_results], pos_list);
        pos_list = _mmX_mask_compress_epi32(pos_list, scan_result, indexes);
        free_entries_in_pos_list -= num_results;

        indexes = _mmX_add_epi32(indexes, index_add);
    }

    total_results += avx_second_scan(pos_list, max_entries_in_pos_list - free_entries_in_pos_list);

    return total_results;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage ./program TABLE_SIZE DISTINCT_VALUES";
        return -1;
    }

    PAPI_library_init(PAPI_VER_CURRENT);

    const std::string program_name  = std::string(argv[0]);
    const size_t table_size         = std::stol(argv[1]);
    const size_t distinct_values    = std::stol(argv[2]);

    if (distinct_values > table_size) {
        std::string empty_papi_string = "";
        for (size_t i = 0; i < num_events; ++i) {
            empty_papi_string.append("-1,");
        }

        std::cout << "AVX" << REG << "," << table_size << "," << distinct_values << "," << -1 << "," << empty_papi_string << 0 << "," << 0 << std::endl;
        return 0;
    }

    col_a = std::vector<value_id_t>(table_size);
    col_b = std::vector<value_id_t>(table_size);

    fill_vector(col_a, distinct_values);
    fill_vector(col_b, distinct_values);

    auto ptr_a = std::make_shared<std::vector<value_id_t>>(col_a);
    auto ptr_b = std::make_shared<std::vector<value_id_t>>(col_b);

    auto run_info = run_n_times(avx_scan, {ptr_a, ptr_b});

    std::string papi_string = "";
    for (auto papi_median : run_info.papi_medians) {
        papi_string.append(std::to_string(papi_median));
        papi_string.append(",");
    }

    std::cout << "AVX" << REG << "," << table_size << "," << distinct_values << "," << run_info.duration << "," << papi_string << run_info.qualifying_rows << "," << run_info.result_sum << std::endl;

    return 0;
}
