#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <tuple>
#include <vector>
#include <papi.h>

using pos_t = uint32_t;
using value_id_t = int32_t;

const size_t CACHE_SIZE = 40'370'176;

struct RunInfo {
    size_t duration;
    int result_sum;
    int qualifying_rows;
    std::vector<long long> papi_medians;
};

// flushes a given memory range from cache
void mem_flush(const void *p, size_t allocation_size) {
    const size_t cache_line = 64;
    const char *cp = (const char *)p;
    size_t i = 0;

    if (p == NULL || allocation_size <= 0) {
        std::cout << "Cache Flush Fail" << std::endl;
        return;
    }

    for (i = 0; i < allocation_size; i += cache_line) {
            asm volatile("clflush (%0)\n\t"
                         :
                         : "r"(&cp[i])
                         : "memory");
    }

    asm volatile("sfence\n\t"
                 :
                 :
                 : "memory");
}

// fills the vector with random values
void fill_vector(std::vector<value_id_t>& vec, size_t distinct_values, size_t random_init = 5) {
    std::mt19937 mt(random_init);
    std::uniform_int_distribution<value_id_t> dist(1, distinct_values);
    std::generate(vec.begin(), vec.end(), [&dist, &mt]{return dist(mt);});
}

int USELESS_HWPF = 0;
int events[] = {USELESS_HWPF, PAPI_BR_MSP, PAPI_PRF_DM, PAPI_RES_STL};
constexpr size_t num_events = sizeof(events) / sizeof(PAPI_INT_INS);

RunInfo run_n_times(int(*func)(), std::vector<std::shared_ptr<std::vector<value_id_t>>> data_vectors, int n = 51) {
    int result_sum = 0;

    std::vector<size_t> durations_us(n);

    if(PAPI_event_name_to_code("L2_LINES_OUT:USELESS_HWPF", &USELESS_HWPF) != PAPI_OK) {
        throw std::runtime_error("PAPI_event_name_to_code failed");
    }
    events[0] = USELESS_HWPF;

    std::vector<std::array<long long, num_events>> vec_of_values(n);

    for (int run = 0; run < n; ++run) {
        if (PAPI_start_counters(events, num_events) != PAPI_OK) {
            throw std::runtime_error("PAPI_start_counters failed");
        }

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        int result = func();
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        if (PAPI_stop_counters(vec_of_values[run].data(), num_events) != PAPI_OK) {
            throw std::runtime_error("PAPI_stop_counters failed");
        }

        durations_us[run] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        result_sum += result;
        for (const auto data_vector : data_vectors) {
            mem_flush(data_vector->data(), data_vector->size() * sizeof(value_id_t));
        }
    }

    std::sort(durations_us.begin(), durations_us.end());

    // Calculate median values for PAPI counters
    std::vector<std::vector<long long>> vec_of_papi_event_counts(num_events);
    for (auto& values : vec_of_values) {
        for (size_t event_id = 0; event_id < num_events; ++event_id) {
            vec_of_papi_event_counts[event_id].push_back(values[event_id]);
        }
    }
    std::vector<long long> papi_medians;
    for (auto& papi_event_counts : vec_of_papi_event_counts) {
        std::sort(papi_event_counts.begin(), papi_event_counts.end());
        papi_medians.push_back(papi_event_counts[papi_event_counts.size() / 2]);
    }

    RunInfo run_info = {durations_us[durations_us.size() / 2], result_sum, result_sum / n, papi_medians};

    return run_info;
}
