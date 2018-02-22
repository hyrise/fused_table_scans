#include <iostream>
#include <immintrin.h>
#include <bitset>
#include <random>
#include <algorithm>
#include <fstream>

#include "common.hpp"

#ifdef AUTO_VECTORIZATION
#define ALGORITHM_NAME "SISD_AUTO_VEC"
#else
#define ALGORITHM_NAME "SISD_NO_AUTO_VEC"
#endif

#ifndef AUTO_VECTORIZATION
__attribute__((optimize("no-tree-vectorize")))
#endif
int sisd_scan() {
    volatile int total_results = 0;
    for(pos_t i = 0; i < col_a.size(); ++i) {
        if(col_a[i] == first_scan_value && col_b[i] == second_scan_value) {
            ++total_results;
        }
    }
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

        std::cout << ALGORITHM_NAME << "," << table_size << "," << distinct_values << "," << -1 << "," << empty_papi_string << 0 << "," << 0 << std::endl;
        return 0;
    }

    col_a = std::vector<value_id_t>(table_size);
    col_b = std::vector<value_id_t>(table_size);

    fill_vector(col_a, distinct_values);
    fill_vector(col_b, distinct_values);

    auto ptr_a = std::make_shared<std::vector<value_id_t>>(col_a);
    auto ptr_b = std::make_shared<std::vector<value_id_t>>(col_b);

    auto run_info = run_n_times(sisd_scan, {ptr_a, ptr_b});

    std::string papi_string = "";
    for (auto papi_median : run_info.papi_medians) {
        papi_string.append(std::to_string(papi_median));
        papi_string.append(",");
    }

    std::cout << ALGORITHM_NAME << "," << table_size << "," << distinct_values << "," << run_info.duration << "," << papi_string << run_info.qualifying_rows << "," << run_info.result_sum << std::endl;

    return 0;
}
