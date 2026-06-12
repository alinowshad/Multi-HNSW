#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/define.hpp"
#include "../../hnswlib/IO.hpp"
#include "hnswlib/Quantizer_hnsw.h"
#include "hnswlib/Rotator.hpp"
#include "hnswlib/ScalarQuantizer.hpp"

#include <thread>
#include <array>
#include <cstring>
#include <filesystem>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <functional>
#include <atomic>
#include <exception>
#include <tuple>
#include <numeric>
#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <fstream>
#include <sstream>
#include <limits>
#include <immintrin.h>

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

class ThreadPool {
 public:
    ThreadPool(size_t numThreads) : stop(false), tasks_in_flight(0) {
        workers.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    task();

                    if (--tasks_in_flight == 0) {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv_done.notify_one();
                    }
                }
            });
        }
    }

    void enqueue(std::function<void()> func) {
        tasks_in_flight++;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.push(std::move(func));
        }
        cv.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_done.wait(lock, [this]{ return tasks_in_flight.load() == 0; });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        cv.notify_all();
        for (auto &thread : workers) thread.join();
    }

 private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::condition_variable cv_done;
    std::atomic<size_t> tasks_in_flight;
    bool stop;
};

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);
                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

float MinDist(const float* query_chunk_vectors, const FloatRowMat &data_vectors, int dim,
              size_t query_chunk_size, size_t data_chunk_size,
              size_t data_offset,
              hnswlib::SpaceInterface<float> *space) {
    float total_min_dist = 0.0f;
    hnswlib::DISTFUNC<float> dist_func = space->get_dist_func();
    void* dist_func_param = space->get_dist_func_param();

#ifdef ENABLE_AVX2_IP
    auto dot4_avx2 = [&](const float* q_ptr,
                         const float* d0,
                         const float* d1,
                         const float* d2,
                         const float* d3,
                         int d) -> std::array<float,4> {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= d; i += 8) {
            __m256 qv = _mm256_loadu_ps(q_ptr + i);
            __m256 v0 = _mm256_loadu_ps(d0 + i);
            __m256 v1 = _mm256_loadu_ps(d1 + i);
            __m256 v2 = _mm256_loadu_ps(d2 + i);
            __m256 v3 = _mm256_loadu_ps(d3 + i);
#if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(qv, v0, acc0);
            acc1 = _mm256_fmadd_ps(qv, v1, acc1);
            acc2 = _mm256_fmadd_ps(qv, v2, acc2);
            acc3 = _mm256_fmadd_ps(qv, v3, acc3);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(qv, v0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(qv, v1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(qv, v2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(qv, v3));
#endif
        }
        auto hsum256 = [](__m256 v) -> float {
            __m128 vlow  = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            vlow = _mm_add_ps(vlow, vhigh);
            __m128 shuf = _mm_movehdup_ps(vlow);
            __m128 sums = _mm_add_ps(vlow, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        };
        float s0 = hsum256(acc0);
        float s1 = hsum256(acc1);
        float s2 = hsum256(acc2);
        float s3 = hsum256(acc3);
        for (; i < d; ++i) {
            float qv = q_ptr[i];
            s0 += qv * d0[i];
            s1 += qv * d1[i];
            s2 += qv * d2[i];
            s3 += qv * d3[i];
        }
        return {s0, s1, s2, s3};
    };
#endif

    for (size_t q_idx = 0; q_idx < query_chunk_size; ++q_idx) {
        const float* q_vec = query_chunk_vectors + q_idx * dim;
        float min_dist_for_q = std::numeric_limits<float>::infinity();
        size_t d_idx = 0;

#ifdef ENABLE_AVX2_IP
        for (; d_idx + 4 <= data_chunk_size; d_idx += 4) {
            const float* d0 = data_vectors.row(data_offset + d_idx + 0).data();
            const float* d1 = data_vectors.row(data_offset + d_idx + 1).data();
            const float* d2 = data_vectors.row(data_offset + d_idx + 2).data();
            const float* d3 = data_vectors.row(data_offset + d_idx + 3).data();
            if (d_idx + 16 < data_chunk_size) {
                __builtin_prefetch(data_vectors.row(data_offset + d_idx + 16).data(), 0, 1);
            }
            auto s = dot4_avx2(q_vec, d0, d1, d2, d3, dim);
            min_dist_for_q = std::min(min_dist_for_q, -s[0]);
            min_dist_for_q = std::min(min_dist_for_q, -s[1]);
            min_dist_for_q = std::min(min_dist_for_q, -s[2]);
            min_dist_for_q = std::min(min_dist_for_q, -s[3]);
        }
        for (; d_idx < data_chunk_size; ++d_idx) {
            const float* d_vec = data_vectors.row(data_offset + d_idx).data();
            float acc = 0.0f;
            int i = 0;
            __m256 acc_ps = _mm256_setzero_ps();
            for (; i + 16 <= dim; i += 16) {
                __m256 qv0 = _mm256_loadu_ps(q_vec + i);
                __m256 dv0 = _mm256_loadu_ps(d_vec + i);
                __m256 qv1 = _mm256_loadu_ps(q_vec + i + 8);
                __m256 dv1 = _mm256_loadu_ps(d_vec + i + 8);
#if defined(__FMA__)
                acc_ps = _mm256_fmadd_ps(qv0, dv0, acc_ps);
                acc_ps = _mm256_fmadd_ps(qv1, dv1, acc_ps);
#else
                acc_ps = _mm256_add_ps(acc_ps, _mm256_mul_ps(qv0, dv0));
                acc_ps = _mm256_add_ps(acc_ps, _mm256_mul_ps(qv1, dv1));
#endif
            }
            if (i + 8 <= dim) {
                __m256 qv = _mm256_loadu_ps(q_vec + i);
                __m256 dv = _mm256_loadu_ps(d_vec + i);
#if defined(__FMA__)
                acc_ps = _mm256_fmadd_ps(qv, dv, acc_ps);
#else
                acc_ps = _mm256_add_ps(acc_ps, _mm256_mul_ps(qv, dv));
#endif
                i += 8;
            }
            __m128 low = _mm256_castps256_ps128(acc_ps);
            __m128 high = _mm256_extractf128_ps(acc_ps, 1);
            __m128 sum = _mm_add_ps(low, high);
            __m128 shuf = _mm_movehdup_ps(sum);
            sum = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sum);
            sum = _mm_add_ss(sum, shuf);
            acc += _mm_cvtss_f32(sum);
            for (; i < dim; ++i) {
                acc += q_vec[i] * d_vec[i];
            }
            const float dist = -acc;
            if (dist < min_dist_for_q) {
                min_dist_for_q = dist;
            }
        }
#else
        for (; d_idx + 2 < data_chunk_size; d_idx += 2) {
            const float* d_vec0 = data_vectors.row(data_offset + d_idx).data();
            const float* d_vec1 = data_vectors.row(data_offset + d_idx + 1).data();
            if (d_idx + 8 < data_chunk_size) {
                __builtin_prefetch(data_vectors.row(data_offset + d_idx + 8).data(), 0, 1);
            }
            float dist0 = dist_func(q_vec, d_vec0, dist_func_param);
            float dist1 = dist_func(q_vec, d_vec1, dist_func_param);
            min_dist_for_q = std::min(min_dist_for_q, dist0);
            min_dist_for_q = std::min(min_dist_for_q, dist1);
        }
        for (; d_idx < data_chunk_size; ++d_idx) {
            const float* d_vec = data_vectors.row(data_offset + d_idx).data();
            float dist = dist_func(q_vec, d_vec, dist_func_param);
            if (dist < min_dist_for_q) min_dist_for_q = dist;
        }
#endif
        total_min_dist += min_dist_for_q;
    }
    return total_min_dist / query_chunk_size;
}

float MinDistQuantized(
    const std::vector<hnswlib::HierarchicalNSW<float>::PreprocessedQuantizedQuery>& prepared_query_queries,
    size_t query_offset,
    size_t query_chunk_size,
    size_t data_chunk_size,
    size_t data_offset,
    hnswlib::HierarchicalNSW<float>* alg_hnsw
) {
    float total_min_dist = 0.0f;
    const size_t base_block_idx = data_offset / FAST_SIZE;
    const size_t end_block_idx = (data_offset + data_chunk_size + FAST_SIZE - 1) / FAST_SIZE;
    const size_t local_block_count = end_block_idx - base_block_idx;
    static thread_local hnswlib::HierarchicalNSW<float>::FastScanScratch scratch;

    for (size_t q_idx = 0; q_idx < query_chunk_size; ++q_idx) {
        scratch.prepareRange(base_block_idx, local_block_count);
        const auto& prepared_query = prepared_query_queries[query_offset + q_idx];
        float min_dist_for_q = std::numeric_limits<float>::infinity();

        for (size_t d_idx = 0; d_idx < data_chunk_size; ++d_idx) {
            const hnswlib::tableint internal_id =
                static_cast<hnswlib::tableint>(data_offset + d_idx);
            const float dist = alg_hnsw->computeQuantizedDistance(prepared_query, internal_id, scratch);
            if (dist < min_dist_for_q) {
                min_dist_for_q = dist;
            }
        }
        total_min_dist += min_dist_for_q;
    }

    return total_min_dist / query_chunk_size;
}

namespace {

hnswlib::HierarchicalNSW<float>::ConstructionGraphMode parseConstructionGraphMode(
    const std::string& mode
) {
    using Mode = hnswlib::HierarchicalNSW<float>::ConstructionGraphMode;
    if (mode == "raw") return Mode::Raw;
    if (mode == "hybrid") return Mode::Hybrid;
    if (mode == "auto") return Mode::Auto;
    throw std::runtime_error("Unsupported construction_graph_mode: " + mode);
}

std::string constructionGraphModeName(
    hnswlib::HierarchicalNSW<float>::ConstructionGraphMode mode
) {
    using Mode = hnswlib::HierarchicalNSW<float>::ConstructionGraphMode;
    switch (mode) {
        case Mode::Raw: return "raw";
        case Mode::Hybrid: return "hybrid";
        case Mode::Auto: return "auto";
    }
    return "unknown";
}

std::string resolveCompressedIndexPath(
    const std::string& dataset,
    const std::string& quantization_mode,
    hnswlib::HierarchicalNSW<float>::ConstructionGraphMode construction_graph_mode,
    bool load_existing_index,
    bool exact_load_mode
) {
    const std::string base =
        "/data/ali/indexing/" + dataset + "_" + quantization_mode +
        "_index_colbert_quantized_complete_compressed_ids";

    if (!load_existing_index) {
        return base + "_" + constructionGraphModeName(construction_graph_mode) + ".bin";
    }

    if (exact_load_mode) {
        return base + "_" + constructionGraphModeName(construction_graph_mode) + ".bin";
    }

    const std::vector<std::string> candidates = {
        base + ".bin",
        base + "_auto.bin",
        base + "_raw.bin",
        base + "_hybrid.bin"
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return base + ".bin";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <DATASET> [exrabitq|scalar_int8] [load [raw|hybrid|auto] | raw|hybrid|auto [construction_graph_ram_limit_gb]]"
                  << std::endl;
        return 1;
    }

    int dim;
    int M = 128;
    int ef_construction = 128;
    int num_threads =80;
    const char* DATASET = argv[1];
    std::string quantization_mode = argc >= 3 ? argv[2] : "exrabitq";
    bool explicit_load_command = false;
    bool exact_load_mode = false;
    hnswlib::HierarchicalNSW<float>::ConstructionGraphMode construction_graph_mode =
        hnswlib::HierarchicalNSW<float>::ConstructionGraphMode::Raw;
    if (argc >= 4) {
        const std::string command_or_mode = argv[3];
        if (command_or_mode == "load") {
            explicit_load_command = true;
            if (argc >= 5) {
                construction_graph_mode = parseConstructionGraphMode(argv[4]);
                exact_load_mode = true;
            }
        } else {
            construction_graph_mode = parseConstructionGraphMode(command_or_mode);
        }
    }
    const double construction_graph_ram_limit_gb =
        (!explicit_load_command && argc >= 5) ? std::stod(argv[4]) : 0.0;
    const bool force_rebuild = argc >= 4 && !explicit_load_command;
    const bool load_existing_index = !force_rebuild;
    // Reuse the already-loaded base matrix during construction to avoid duplicating
    // hundreds of GB of raw vectors inside the index on large datasets.
    const bool use_external_build_raw_data = true;
    const bool use_maxsim_approximation = true;
    const bool use_raw_vectors_for_final_stage = false;
    const bool compress_ids_for_search = true;
    const std::string adjacency_mode = "compressed_ids";
    const bool need_base_raw_vectors =
        !load_existing_index || use_raw_vectors_for_final_stage;

    char data_file[500];
    char query_file[500];
    char gt_file[500];
    char result_file[500];
    char data_file_chunks[500];
    char query_file_chunks[500];
    const std::string results_dir = "/home/ali/hnsw-skipping-construction-quantizer/results";

    sprintf(data_file, "/data/ali/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(data_file_chunks, "/data/ali/%s/%s_base_chunks.fvecs", DATASET, DATASET);
    sprintf(query_file, "/data/ali/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(query_file_chunks, "/data/ali/%s/%s_query_chunks.fvecs", DATASET, DATASET);
    sprintf(gt_file, "/data/ali/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    sprintf(
        result_file,
        "/home/ali/hnsw-skipping-construction-quantizer/results/%s_%s_%s_hnsw-quantized-complete_k.csv",
        DATASET,
        quantization_mode.c_str(),
        adjacency_mode.c_str()
    );
    std::filesystem::create_directories(results_dir);

    FloatRowMat data_vecs;
    FloatRowMat data_chunks;
    FloatRowMat query_vecs;
    FloatRowMat query_chunks;
    UintRowMat gt_vecs;
    const bool has_gt = std::filesystem::exists(gt_file);

    if (need_base_raw_vectors) {
        load_vecs<float, FloatRowMat>(data_file, data_vecs);
    } else {
        std::cout << "Skipping base raw-vector load because the existing index and quantized final stage do not need it\n";
    }
    load_vecs<float, FloatRowMat>(data_file_chunks, data_chunks);
    load_vecs<float, FloatRowMat>(query_file, query_vecs);
    load_vecs<float, FloatRowMat>(query_file_chunks, query_chunks);
    if (has_gt) {
        load_vecs<PID, UintRowMat>(gt_file, gt_vecs);
        std::cout << "Ground truth loaded from: " << gt_file << '\n';
    } else {
        std::cout << "Ground truth not found, running in save-results-only mode\n";
    }

    size_t N = 0;
    for (int i = 0; i < data_chunks.rows(); ++i) {
        N += static_cast<size_t>(data_chunks(i, 0));
    }
    dim = query_vecs.cols();
    size_t NQ = query_vecs.rows();
    size_t NQ_chunks = query_chunks.rows();
    std::vector<int> k_values = {10, 100};
    std::vector<int> target_k_values = {1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100, 200};

    std::cout << "data loaded\n";
    std::cout << "\tN: " << N <<  '\n';
    std::cout << "query loaded\n";
    std::cout << "\tNQ: " << NQ << '\n';
    std::cout << "Use raw vectors for final stage: "
              << (use_raw_vectors_for_final_stage ? "true" : "false") << '\n';
    std::cout << "Adjacency mode: " << adjacency_mode << '\n';
    std::cout << "Construction graph mode request: "
              << constructionGraphModeName(construction_graph_mode) << '\n';
    std::cout << "Construction graph RAM limit GB: "
              << construction_graph_ram_limit_gb << '\n';
    std::cout << "Load existing index: " << (load_existing_index ? "true" : "false") << '\n';
    std::cout << "Exact load mode: " << (exact_load_mode ? "true" : "false") << '\n';

    hnswlib::InnerProductSpace space(dim);
    int b = 4;
    DataQuantizer quantizer(dim, b);
    Rotator rotator(dim);
    ScalarInt8Quantizer scalar_quantizer(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = nullptr;
    hnswlib::HierarchicalNSW<float>::ConstructionGraphOptions graph_options;
    graph_options.mode = construction_graph_mode;
    graph_options.ram_limit_gb = construction_graph_ram_limit_gb;

    std::string index_path =
        "/data/ali/indexing/" + std::string(DATASET) + "_" + quantization_mode + "_index_colbert_quantized_complete.bin";
    std::string compressed_index_path = resolveCompressedIndexPath(
        std::string(DATASET),
        quantization_mode,
        construction_graph_mode,
        load_existing_index,
        exact_load_mode
    );

    std::vector<std::pair<size_t, size_t>> data_parent_info;
    size_t current_offset = 0;
    for (int i = 0; i < data_chunks.rows(); ++i) {
        size_t chunk_size = data_chunks(i, 0);
        data_parent_info.push_back({current_offset, chunk_size});
        current_offset += chunk_size;
    }
    std::vector<size_t> parent_offsets(data_parent_info.size());
    std::vector<size_t> parent_chunk_sizes(data_parent_info.size());
    for (size_t parent_id = 0; parent_id < data_parent_info.size(); ++parent_id) {
        parent_offsets[parent_id] = data_parent_info[parent_id].first;
        parent_chunk_sizes[parent_id] = data_parent_info[parent_id].second;
    }

    std::vector<std::pair<size_t, size_t>> query_parent_info;
    current_offset = 0;
    for (int i = 0; i < query_chunks.rows(); ++i) {
        size_t chunk_size = query_chunks(i, 0);
        query_parent_info.push_back({current_offset, chunk_size});
        current_offset += chunk_size;
    }
    if (current_offset != NQ) {
        throw std::runtime_error(
            "Sum of query chunk sizes does not match the number of query vectors"
        );
    }
    if (has_gt && static_cast<size_t>(gt_vecs.rows()) != NQ_chunks) {
        throw std::runtime_error(
            "Ground truth row count does not match the number of query chunks"
        );
    }

    std::vector<hnswlib::labeltype> vector_to_parent_map(N);
    size_t vector_offset = 0;
    for (int i = 0; i < data_chunks.rows(); ++i) {
        size_t chunk_size = data_chunks(i, 0);
        for (size_t j = 0; j < chunk_size; ++j) {
            size_t vector_index = vector_offset + j;
            if (vector_index >= N) {
                throw std::runtime_error("Sum of chunks exceeds the total number of vectors.");
            }
            vector_to_parent_map[vector_index] = i;
        }
        vector_offset += chunk_size;
    }
    if (vector_offset != N) {
        throw std::runtime_error("Sum of chunks does not match the total number of vectors.");
    }
    std::cout << "Offsets are computed" << std::endl;

    if (load_existing_index) {
        std::cout << "Loading compressed index from: " << compressed_index_path << std::endl;
        if (!std::filesystem::exists(compressed_index_path)) {
            throw std::runtime_error(
                "Compressed index not found. Expected one of the legacy or mode-specific compressed index filenames."
            );
        }
        StopW stopw_loading;
        if (quantization_mode == "exrabitq") {
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                &space, quantizer.short_code_length(), quantizer.long_code_length(), b, compressed_index_path
            );
        } else if (quantization_mode == "scalar_int8") {
            (void)scalar_quantizer;
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 0, 0, 0, compressed_index_path);
        } else {
            std::cerr << "Unsupported quantization mode: " << quantization_mode << std::endl;
            return 1;
        }
        float loading_time_s = stopw_loading.getElapsedTimeMicro() / 1e6;
        std::cout << "Compressed index loading time: " << loading_time_s << " s" << std::endl;
        std::cout << "Compressed index loaded successfully" << std::endl;
    } else {
        std::cout << "Building normal index in memory and saving only compressed output to: "
                  << compressed_index_path << std::endl;
        if (quantization_mode == "exrabitq") {
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                &space,
                quantizer.short_code_length(),
                quantizer.long_code_length(),
                b,
                N,
                M,
                ef_construction,
                100,
                false,
                !use_external_build_raw_data,
                graph_options
            );
            alg_hnsw->setQuantizationModel(quantizer, rotator);
        } else if (quantization_mode == "scalar_int8") {
            scalar_quantizer.train(data_vecs.data(), N);
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                &space, 0, 0, 0, N, M, ef_construction, 100, false, !use_external_build_raw_data, graph_options
            );
            alg_hnsw->setScalarInt8Model(scalar_quantizer);
        } else {
            std::cerr << "Unsupported quantization mode: " << quantization_mode << std::endl;
            return 1;
        }

        if (use_external_build_raw_data) {
            std::cout << "Using external raw data backing during build" << std::endl;
            alg_hnsw->setExternalBuildRawData(
                data_vecs.data(),
                static_cast<size_t>(data_vecs.outerStride()) * sizeof(float),
                N
            );
        }

        StopW stopw_indexing;
        ParallelFor(0, N, num_threads, [&](size_t row, size_t threadId) {
            (void)threadId;
            hnswlib::labeltype parent_id = vector_to_parent_map[row];
            alg_hnsw->addPoint((void*)(data_vecs.row(row).data()), row, parent_id);
        });
        float indexing_time_s = stopw_indexing.getElapsedTimeMicro() / 1e6f;
        std::cout << "Indexing time: " << indexing_time_s << " s" << std::endl;
        std::cout << "Effective construction graph mode: "
                  << constructionGraphModeName(alg_hnsw->effectiveConstructionGraphMode()) << std::endl;
        std::cout << "Estimated raw adjacency bytes: "
                  << alg_hnsw->estimatedRawConstructionAdjacencyBytes() << std::endl;
        std::cout << "Construction graph RAM limit bytes: "
                  << alg_hnsw->constructionGraphRamLimitBytes() << std::endl;
        std::cout << "Construction compressed base bytes: "
                  << alg_hnsw->constructionCompressedBaseBytes() << std::endl;
        std::cout << "Construction delta overlay bytes: "
                  << alg_hnsw->constructionDeltaOverlayBytes() << std::endl;
        std::cout << "Construction total adjacency bytes: "
                  << alg_hnsw->constructionAdjacencyBytes() << std::endl;
        std::cout << "Construction merge count: "
                  << alg_hnsw->constructionMergeCount() << std::endl;
        std::cout << "Construction merge seconds: "
                  << alg_hnsw->constructionMergeSeconds() << std::endl;
        alg_hnsw->finalizeQuantizedIndex();
    }

    if (!load_existing_index && compress_ids_for_search) {
        const size_t raw_neighbor_reserved_bytes = alg_hnsw->rawNeighborIdBytesReserved();
        StopW stopw_compress_ids;
        alg_hnsw->compressNeighborIdsSortedVarint(true);
        const float compress_ids_time_s = stopw_compress_ids.getElapsedTimeMicro() / 1e6f;
        const size_t compressed_neighbor_bytes = alg_hnsw->compressedNeighborIdBytes();
        const double compression_ratio =
            raw_neighbor_reserved_bytes == 0
                ? 0.0
                : static_cast<double>(compressed_neighbor_bytes) /
                      static_cast<double>(raw_neighbor_reserved_bytes);
        std::cout << "Compressed adjacency IDs in " << compress_ids_time_s << " s" << std::endl;
        std::cout << "Raw reserved neighbor bytes: " << raw_neighbor_reserved_bytes << std::endl;
        std::cout << "Compressed neighbor bytes: " << compressed_neighbor_bytes << std::endl;
        std::cout << "Neighbor compression ratio: " << compression_ratio << std::endl;
        std::cout << "Neighbor savings percent: " << (1.0 - compression_ratio) * 100.0 << std::endl;
        std::cout << "Saving compressed adjacency index to: " << compressed_index_path << std::endl;
        alg_hnsw->saveIndex(compressed_index_path);
        std::cout << "Compressed adjacency index saved successfully" << std::endl;
        std::cout << "Reloading compressed adjacency index for search/evaluation" << std::endl;
        delete alg_hnsw;
        alg_hnsw = nullptr;
        StopW stopw_reload;
        if (quantization_mode == "exrabitq") {
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                &space, quantizer.short_code_length(), quantizer.long_code_length(), b, compressed_index_path
            );
        } else if (quantization_mode == "scalar_int8") {
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 0, 0, 0, compressed_index_path);
        } else {
            throw std::runtime_error("Unsupported quantization mode during reload");
        }
        std::cout << "Reloaded compressed index in "
                  << stopw_reload.getElapsedTimeMicro() / 1e6f << " s" << std::endl;
    }

    std::cout << "Starting quantized query preprocessing" << std::endl;
    std::vector<hnswlib::HierarchicalNSW<float>::PreprocessedQuantizedQuery> prepared_query_queries(NQ);
    ParallelFor(0, NQ, num_threads, [&](size_t row, size_t threadId) {
        (void)threadId;
        prepared_query_queries[row] = alg_hnsw->preprocessQueryQuantization(query_vecs.row(row).data());
    });
    std::cout << "Finished quantized query preprocessing" << std::endl;

    std::ofstream result_output(result_file);
    if (!result_output.is_open()) {
        std::cerr << "Failed to open result file: " << result_file << std::endl;
        return 1;
    }
    result_output << "tk,k,Recall,MRR,nDCG,QPS" << std::endl;

    float epsilon = 0.01f;
    const size_t search_threads = std::max(1, num_threads / 2);
    const size_t rerank_threads = std::max(1, num_threads - static_cast<int>(search_threads));
    ThreadPool search_pool(search_threads);
    ThreadPool rerank_pool(rerank_threads);
    auto reciprocal_rank_at_k =
        [&](const std::vector<std::pair<hnswlib::labeltype, float>>& topk_rows,
            hnswlib::labeltype gt_label) -> double {
            for (size_t rank = 0; rank < topk_rows.size(); ++rank) {
                if (topk_rows[rank].first == gt_label) {
                    return 1.0 / static_cast<double>(rank + 1);
                }
            }
            return 0.0;
        };
    auto ndcg_at_k =
        [&](const std::vector<std::pair<hnswlib::labeltype, float>>& topk_rows,
            hnswlib::labeltype gt_label) -> double {
            for (size_t rank = 0; rank < topk_rows.size(); ++rank) {
                if (topk_rows[rank].first == gt_label) {
                    return 1.0 / std::log2(static_cast<double>(rank + 2));
                }
            }
            return 0.0;
        };
    auto collect_topk_predictions =
        [&](int query_idx,
            int k,
            int tk,
            const std::vector<std::pair<float, hnswlib::labeltype>>& reranked_parents,
            hnswlib::labeltype gt_label,
            std::vector<std::pair<hnswlib::labeltype, float>>& topk_rows) -> bool {
            (void)query_idx;
            (void)tk;
            bool found_gt = false;
            topk_rows.clear();
            topk_rows.reserve(std::min<int>(k, static_cast<int>(reranked_parents.size())));
            for (int j = 0; j < k && j < (int)reranked_parents.size(); ++j) {
                topk_rows.emplace_back(reranked_parents[j].second, reranked_parents[j].first);
                found_gt = found_gt || (reranked_parents[j].second == gt_label);
            }
            return found_gt;
        };

    using SearchResult = std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>;
    struct DocumentStageSlot {
        size_t query_offset{0};
        size_t query_chunk_size{0};
        std::vector<std::vector<SearchResult>> all_search_results;
        std::vector<float> query_chunk_buffer;
        std::mutex ready_mutex;
        std::condition_variable ready_cv;
        std::atomic<size_t> pending_tasks{0};
        bool ready{false};
    };

    std::array<DocumentStageSlot, 2> stage_slots;
    std::vector<float> kth_distances;
    std::vector<hnswlib::labeltype> parent_ids;
    std::vector<int> parent_id_to_index(data_parent_info.size(), -1);
    std::vector<size_t> touched_parent_slots;
    std::vector<float> best_dist_per_parent_query;
    std::vector<float> optimistic_dist;
    std::vector<size_t> top_candidate_indices;
    std::vector<std::pair<float, hnswlib::labeltype>> reranked_parents;

    auto enqueue_search_stage = [&](DocumentStageSlot& slot, int query_chunk_id, int tk) {
        slot.query_offset = query_parent_info[query_chunk_id].first;
        slot.query_chunk_size = query_parent_info[query_chunk_id].second;
        slot.query_chunk_buffer.resize(slot.query_chunk_size * dim);
        for (size_t j = 0; j < slot.query_chunk_size; ++j) {
            std::memcpy(
                slot.query_chunk_buffer.data() + j * dim,
                query_vecs.row(slot.query_offset + j).data(),
                static_cast<size_t>(dim) * sizeof(float)
            );
        }

        slot.all_search_results.resize(slot.query_chunk_size);
        for (auto& result_vec : slot.all_search_results) {
            result_vec.clear();
            if (result_vec.capacity() < static_cast<size_t>(tk)) {
                result_vec.reserve(tk);
            }
        }

        {
            std::lock_guard<std::mutex> lock(slot.ready_mutex);
            slot.ready = false;
        }
        slot.pending_tasks.store(slot.query_chunk_size, std::memory_order_relaxed);
        if (slot.query_chunk_size == 0) {
            std::lock_guard<std::mutex> lock(slot.ready_mutex);
            slot.ready = true;
            slot.ready_cv.notify_one();
            return;
        }

        DocumentStageSlot* slot_ptr = &slot;
        const size_t query_offset_local = slot.query_offset;
        const int tk_local = tk;
        for (size_t j = 0; j < slot.query_chunk_size; ++j) {
            search_pool.enqueue([&, j, slot_ptr, query_offset_local, tk_local] {
                auto results = alg_hnsw->searchKnnSkippingDuplicates_quantized(
                    prepared_query_queries[query_offset_local + j], tk_local
                );

                auto& result_vec = slot_ptr->all_search_results[j];
                while (!results.empty()) {
                    result_vec.push_back(results.top());
                    results.pop();
                }
                std::reverse(result_vec.begin(), result_vec.end());
                if (slot_ptr->pending_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lock(slot_ptr->ready_mutex);
                    slot_ptr->ready = true;
                    slot_ptr->ready_cv.notify_one();
                }
            });
        }
    };

    auto wait_for_search_stage = [&](DocumentStageSlot& slot) {
        std::unique_lock<std::mutex> lock(slot.ready_mutex);
        slot.ready_cv.wait(lock, [&slot] { return slot.ready; });
    };

    for (int tk : target_k_values) {
        for (int k : k_values) {
            const std::string runfile_path =
                results_dir + "/" + std::string(DATASET) + "_" +
                quantization_mode + "_" + adjacency_mode +
                (use_maxsim_approximation ? "_maxsim_approx" : "_full") +
                "_k" + std::to_string(k) + "_tk" + std::to_string(tk) + "_beir.tsv";
            std::vector<std::vector<std::pair<hnswlib::labeltype, float>>> topk_predictions(NQ_chunks);
            std::vector<uint8_t> gt_hits(NQ_chunks, 0);
            double sum_mrr = 0.0;
            double sum_ndcg = 0.0;
            if (NQ_chunks == 0) {
                continue;
            }

            float total_search_time_us = 0.0f;
            size_t processed_query_count = 0;
            enqueue_search_stage(stage_slots[0], 0, tk);
            for (int i = 0; i < NQ_chunks; i++) {
                if ( i == 1000)
                    break;
                StopW query_search_stopw;
                DocumentStageSlot& current_slot = stage_slots[i % 2];
                wait_for_search_stage(current_slot);
                const size_t query_offset = current_slot.query_offset;
                const size_t query_chunk_size = current_slot.query_chunk_size;
                if (i + 1 < static_cast<int>(NQ_chunks)) {
                    DocumentStageSlot& next_slot = stage_slots[(i + 1) % 2];
                    enqueue_search_stage(next_slot, i + 1, tk);
                }

                kth_distances.assign(query_chunk_size, 1.0f);
                parent_ids.clear();
                touched_parent_slots.clear();
                best_dist_per_parent_query.clear();
                for (size_t j = 0; j < query_chunk_size; ++j) {
                    const auto& search_results = current_slot.all_search_results[j];
                    if (!search_results.empty()) {
                        const size_t kth_index =
                            std::min(static_cast<size_t>(tk - 1), search_results.size() - 1);
                        kth_distances[j] = std::get<0>(search_results[kth_index]);

                        for (const auto& result : search_results) {
                            const float approx_dist = std::get<0>(result);
                            const hnswlib::labeltype parent_id = std::get<2>(result);
                            const size_t parent_slot = static_cast<size_t>(parent_id);
                            int parent_index = parent_id_to_index[parent_slot];
                            if (parent_index < 0) {
                                parent_index = static_cast<int>(parent_ids.size());
                                parent_id_to_index[parent_slot] = parent_index;
                                touched_parent_slots.push_back(parent_slot);
                                parent_ids.push_back(parent_id);
                                best_dist_per_parent_query.insert(
                                    best_dist_per_parent_query.end(),
                                    query_chunk_size,
                                    std::numeric_limits<float>::infinity()
                                );
                            }
                            float& best_dist =
                                best_dist_per_parent_query[static_cast<size_t>(parent_index) * query_chunk_size + j];
                            if (approx_dist < best_dist) {
                                best_dist = approx_dist;
                            }
                        }
                    }
                }

                reranked_parents.clear();
                if (use_maxsim_approximation) {
                    if (!parent_ids.empty()) {
                        optimistic_dist.assign(parent_ids.size(), 0.0f);
                        for (size_t idx = 0; idx < parent_ids.size(); ++idx) {
                            const size_t base = idx * query_chunk_size;
                            float optimistic = 0.0f;
                            for (size_t j = 0; j < query_chunk_size; ++j) {
                                const float best_dist = best_dist_per_parent_query[base + j];
                                if (std::isfinite(best_dist)) {
                                    optimistic += best_dist;
                                } else {
                                    optimistic += kth_distances[j] + epsilon;
                                }
                            }
                            optimistic_dist[idx] = optimistic / static_cast<float>(query_chunk_size);
                        }

                        top_candidate_indices.resize(parent_ids.size());
                        std::iota(top_candidate_indices.begin(), top_candidate_indices.end(), 0);
                        const size_t approx_rerank_pool =
                            std::max<size_t>(
                                static_cast<size_t>(k),
                                static_cast<size_t>(tk) * query_chunk_size
                            );
                        const size_t current_k =
                            std::min<size_t>(approx_rerank_pool, top_candidate_indices.size());
                        if (current_k < top_candidate_indices.size()) {
                            std::nth_element(
                                top_candidate_indices.begin(),
                                top_candidate_indices.begin() + current_k,
                                top_candidate_indices.end(),
                                [&](size_t lhs, size_t rhs) {
                                    return optimistic_dist[lhs] < optimistic_dist[rhs];
                                }
                            );
                            top_candidate_indices.resize(current_k);
                        }

                        std::vector<hnswlib::labeltype> top_candidate_parent_ids(top_candidate_indices.size());
                        for (size_t ci = 0; ci < top_candidate_indices.size(); ++ci) {
                            top_candidate_parent_ids[ci] = parent_ids[top_candidate_indices[ci]];
                        }

                        if (use_raw_vectors_for_final_stage) {
                            reranked_parents.resize(top_candidate_indices.size());
                            for (size_t ci = 0; ci < top_candidate_indices.size(); ++ci) {
                                rerank_pool.enqueue([&, ci] {
                                    const hnswlib::labeltype parent_id = top_candidate_parent_ids[ci];
                                    const size_t data_offset = parent_offsets[parent_id];
                                    const size_t data_chunk_size = parent_chunk_sizes[parent_id];
                                    const float distance = MinDist(
                                        current_slot.query_chunk_buffer.data(), data_vecs, dim,
                                        query_chunk_size, data_chunk_size,
                                        data_offset, &space
                                    );
                                    reranked_parents[ci] = {distance, parent_id};
                                });
                            }
                            rerank_pool.wait();
                        } else {
                            reranked_parents.resize(top_candidate_indices.size());
                            for (size_t ci = 0; ci < top_candidate_indices.size(); ++ci) {
                                rerank_pool.enqueue([&, ci] {
                                    const hnswlib::labeltype parent_id = top_candidate_parent_ids[ci];
                                    const size_t data_offset = parent_offsets[parent_id];
                                    const size_t data_chunk_size = parent_chunk_sizes[parent_id];
                                    const float distance = MinDistQuantized(
                                        prepared_query_queries,
                                        query_offset,
                                        query_chunk_size,
                                        data_chunk_size,
                                        data_offset,
                                        alg_hnsw
                                    );
                                    reranked_parents[ci] = {distance, parent_id};
                                });
                            }
                            rerank_pool.wait();
                        }
                        std::sort(reranked_parents.begin(), reranked_parents.end());
                    }
                } else {
                    if (use_raw_vectors_for_final_stage) {
                        reranked_parents.resize(parent_ids.size());
                        for (size_t ci = 0; ci < parent_ids.size(); ++ci) {
                            rerank_pool.enqueue([&, ci] {
                                const hnswlib::labeltype parent_id = parent_ids[ci];
                                const size_t data_offset = parent_offsets[parent_id];
                                const size_t data_chunk_size = parent_chunk_sizes[parent_id];
                                const float distance = MinDist(
                                    current_slot.query_chunk_buffer.data(), data_vecs, dim,
                                    query_chunk_size, data_chunk_size,
                                    data_offset, &space
                                );
                                reranked_parents[ci] = {distance, parent_id};
                            });
                        }
                        rerank_pool.wait();
                    } else {
                        reranked_parents.resize(parent_ids.size());
                        for (size_t ci = 0; ci < parent_ids.size(); ++ci) {
                            rerank_pool.enqueue([&, ci] {
                                const hnswlib::labeltype parent_id = parent_ids[ci];
                                const size_t data_offset = parent_offsets[parent_id];
                                const size_t data_chunk_size = parent_chunk_sizes[parent_id];
                                const float distance = MinDistQuantized(
                                    prepared_query_queries,
                                    query_offset,
                                    query_chunk_size,
                                    data_chunk_size,
                                    data_offset,
                                    alg_hnsw
                                );
                                reranked_parents[ci] = {distance, parent_id};
                            });
                        }
                        rerank_pool.wait();
                    }
                    std::sort(reranked_parents.begin(), reranked_parents.end());
                }

                // QPS timing stops after rerank/sort; prediction materialization and metrics stay outside.
                total_search_time_us += query_search_stopw.getElapsedTimeMicro();
                const hnswlib::labeltype gt_vector_label = has_gt ? gt_vecs(i, 0) : hnswlib::labeltype();
                gt_hits[i] = collect_topk_predictions(i, k, tk, reranked_parents, gt_vector_label, topk_predictions[i]) && has_gt;
                ++processed_query_count;
                if (has_gt) {
                    sum_mrr += reciprocal_rank_at_k(topk_predictions[i], gt_vector_label);
                    sum_ndcg += ndcg_at_k(topk_predictions[i], gt_vector_label);
                }
                for (size_t parent_slot : touched_parent_slots) {
                    parent_id_to_index[parent_slot] = -1;
                }
            }

            float qps = processed_query_count == 0 ? 0.0f : processed_query_count / (total_search_time_us / 1e6);
            float correct_knn = 0.0f;
            for (uint8_t hit : gt_hits) {
                correct_knn += static_cast<float>(hit);
            }
            float recall_knn =
                has_gt && processed_query_count > 0 ? (correct_knn / static_cast<float>(processed_query_count)) : -1.0f;
            const double mean_mrr =
                has_gt && processed_query_count > 0 ? (sum_mrr / static_cast<double>(processed_query_count)) : -1.0;
            const double mean_ndcg =
                has_gt && processed_query_count > 0 ? (sum_ndcg / static_cast<double>(processed_query_count)) : -1.0;
            std::cout << "tk" << tk << " k-NN Document Recall@" << k << ": " << recall_knn
                      << ", MRR: " << mean_mrr
                      << ", nDCG: " << mean_ndcg
                      << ", QPS: " << qps << "\n";
            result_output << tk << "," << k << "," << recall_knn << "," << mean_mrr << "," << mean_ndcg << "," << qps << std::endl;

            std::ofstream csv_out(runfile_path, std::ios::out | std::ios::trunc);
            for (int query_idx = 0; query_idx < static_cast<int>(NQ_chunks); ++query_idx) {
                for (const auto& row : topk_predictions[query_idx]) {
                    csv_out << query_idx << "\t" << row.first << "\t" << (-row.second) << "\n";
                }
            }
        }
    }

    result_output.close();
    delete alg_hnsw;
    return 0;
}
