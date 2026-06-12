#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "defines.hpp"
#include "Quantizer_hnsw.h"
#include "ScalarQuantizer.hpp"
#include "utils/space.hpp"
#include "utils/memory.hpp"
#include "utils/tools.hpp"
#include "fastscan/FastScan.hpp"
#include <atomic>
#include <cstdint>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <unordered_map>
#include <tuple>
#include <chrono>
#include <fstream>
#include <limits>
#include <string>
#include <unistd.h>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;
    static constexpr uint32_t SERIALIZATION_MAGIC = 0x51484E53U;
    static constexpr uint32_t SERIALIZATION_VERSION = 4U;
    using aligned_float_vector = std::vector<float, memory::align_allocator<float>>;
    using aligned_int16_vector = std::vector<int16_t, memory::align_allocator<int16_t>>;
    using aligned_byte_vector = std::vector<uint8_t, memory::align_allocator<uint8_t>>;
    enum class QuantizationMethod : uint32_t {
        None = 0,
        ExRaBitQ = 1,
        ScalarInt8 = 2
    };
    enum class ConstructionGraphMode : uint32_t {
        Raw = 0,
        Hybrid = 1,
        Auto = 2
    };

    struct ConstructionGraphOptions {
        ConstructionGraphMode mode{ConstructionGraphMode::Raw};
        double ram_limit_gb{0.0};
        size_t merge_trigger_bytes{128ULL << 20};
        double compaction_ratio{2.0};
        bool log_stats{true};
    };

    struct PreprocessedQuantizedQuery {
        aligned_float_vector unit_q;
        aligned_int16_vector quant_query;
        aligned_float_vector scaled_query;
        aligned_byte_vector fastscan_lut;
        float sumq{0.0f};
        float delta{0.0f};
        int short_shift{0};
        size_t D{0};

        void resize(size_t dim) {
            D = dim;
            unit_q.resize(dim);
            quant_query.resize(dim);
        }

        void resize_scaled(size_t dim) {
            D = dim;
            scaled_query.resize(dim);
        }

        void resize_fastscan_lut(size_t bytes) {
            fastscan_lut.resize(bytes);
        }
    };

    struct FastScanScratch {
        aligned_float_vector block_scores;
        std::vector<uint32_t> block_epochs;
        uint32_t current_epoch{1};
        size_t block_base{0};

        void prepare(size_t block_count) {
            prepareRange(0, block_count);
        }

        void prepareRange(size_t base_block_idx, size_t block_count) {
            block_base = base_block_idx;
            const size_t score_count = block_count * FAST_SIZE;
            if (block_scores.size() < score_count) {
                block_scores.resize(score_count);
            }
            if (block_epochs.size() < block_count) {
                block_epochs.resize(block_count, 0);
            }

            ++current_epoch;
            if (current_epoch == 0) {
                std::fill(block_epochs.begin(), block_epochs.end(), 0);
                current_epoch = 1;
            }
        }
    };

    static constexpr uint32_t INVALID_EXTERNAL_RAW_ROW = std::numeric_limits<uint32_t>::max();

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    size_t short_code_length_{0};
    size_t long_code_length_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };
    const float* data_{nullptr};
    int ex_bits_{0};
    int FAC_RESCALE{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    mutable std::mutex parent_id_lookup_lock;
    std::unordered_map<labeltype, std::vector<labeltype>> parent_id_to_labels_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 }, parent_id_offset_{ 0 };
    size_t ex_factor_offset_{0};

    char *data_level0_memory_{nullptr};
    char *raw_data_memory_{nullptr};
    const char* external_raw_data_view_{nullptr};
    size_t external_raw_data_stride_{0};
    size_t external_raw_data_count_{0};
    std::vector<uint32_t> external_raw_row_ids_;
    char **linkLists_{nullptr};
    uint8_t* short_codes_{nullptr};
    uint8_t* long_codes_{nullptr};
    mutable uint8_t* packed_short_codes_{nullptr};
    int8_t* scalar_codes_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};
    size_t vector_dim_{0};
    size_t quantized_dim_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions
    bool persist_raw_vectors_{true};
    bool quantization_enabled_{false};
    bool raw_data_compacted_{false};
    bool compressed_ids_enabled_{false};
    bool compact_level0_storage_enabled_{false};
    QuantizationMethod quantization_method_{QuantizationMethod::None};
    ConstructionGraphOptions construction_graph_options_{};
    ConstructionGraphMode effective_construction_graph_mode_{ConstructionGraphMode::Raw};

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements
    DataQuantizer quantizer_;
    ScalarInt8Quantizer scalar_quantizer_;
    Rotator rotator_;
    struct CompressedNeighborListMeta {
        uint64_t offset{0};
        uint32_t byte_size{0};
        uint32_t count{0};
    };
    struct SerializedAdjacencySnapshot {
        std::vector<CompressedNeighborListMeta> level0_neighbor_meta;
        std::vector<uint8_t> level0_neighbor_blob;
        std::vector<std::vector<CompressedNeighborListMeta>> upper_neighbor_meta;
        std::vector<uint8_t> upper_neighbor_blob;
    };
    char* compressed_level0_meta_memory_{nullptr};
    size_t compressed_level0_meta_memory_bytes_{0};
    unsigned char* compressed_deleted_marks_{nullptr};
    size_t compressed_deleted_marks_count_{0};
    CompressedNeighborListMeta* compressed_level0_neighbor_meta_{nullptr};
    size_t compressed_level0_neighbor_meta_count_{0};
    std::vector<std::vector<CompressedNeighborListMeta>> compressed_upper_neighbor_meta_;
    std::vector<uint8_t> compressed_level0_neighbor_blob_;
    std::vector<uint8_t> compressed_upper_neighbor_blob_;
    bool hybrid_level0_neighbors_node_local_{false};
    size_t hybrid_delta_bytes_{0};
    size_t hybrid_delta_peak_bytes_{0};
    size_t hybrid_dirty_node_count_{0};
    size_t hybrid_level0_active_base_bytes_{0};
    size_t hybrid_level0_merge_count_{0};
    double hybrid_level0_merge_seconds_{0.0};
    size_t construction_estimated_raw_adjacency_bytes_{0};
    size_t construction_ram_limit_bytes_{0};

    void configureStorageLayout() {
        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        offsetLevel0_ = 0;
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_;
        parent_id_offset_ = label_offset_ + sizeof(labeltype);
        ex_factor_offset_ = parent_id_offset_ + sizeof(labeltype);
        size_data_per_element_ = size_links_level0_ + sizeof(labeltype) + sizeof(labeltype) + sizeof(ExFactor);
    }

    size_t compactLevel0MetaBytes() const {
        return size_data_per_element_ - offsetData_;
    }

    size_t compressedLevel0MetaMemoryBytes() const {
        return compressed_level0_meta_memory_bytes_;
    }

    const uint8_t* hybridLevel0CompressedData(tableint internal_id) const {
        if (!hybrid_level0_neighbors_node_local_ ||
            compressed_level0_neighbor_meta_ == nullptr ||
            internal_id >= compressed_level0_neighbor_meta_count_) {
            return nullptr;
        }
        return reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(compressed_level0_neighbor_meta_[internal_id].offset)
        );
    }

    uint8_t* hybridLevel0CompressedData(tableint internal_id) {
        return const_cast<uint8_t*>(
            static_cast<const HierarchicalNSW<dist_t>*>(this)->hybridLevel0CompressedData(internal_id)
        );
    }

    void freeCompressedLevel0Storage() {
        if (hybrid_level0_neighbors_node_local_ && compressed_level0_neighbor_meta_ != nullptr) {
            const size_t active_count = std::min<size_t>(compressed_level0_neighbor_meta_count_, cur_element_count);
            for (size_t i = 0; i < active_count; ++i) {
                free(hybridLevel0CompressedData(static_cast<tableint>(i)));
            }
            hybrid_level0_neighbors_node_local_ = false;
        }
        free(compressed_level0_meta_memory_);
        compressed_level0_meta_memory_ = nullptr;
        compressed_level0_meta_memory_bytes_ = 0;

        free(compressed_deleted_marks_);
        compressed_deleted_marks_ = nullptr;
        compressed_deleted_marks_count_ = 0;

        free(compressed_level0_neighbor_meta_);
        compressed_level0_neighbor_meta_ = nullptr;
        compressed_level0_neighbor_meta_count_ = 0;
    }

    void allocateCompressedLevel0Storage(size_t element_count, bool zero_initialize) {
        freeCompressedLevel0Storage();
        if (element_count == 0) {
            return;
        }

        const size_t meta_bytes = compactLevel0MetaBytes();
        const size_t total_meta_bytes = element_count * meta_bytes;
        compressed_level0_meta_memory_ = static_cast<char*>(
            zero_initialize ? calloc(total_meta_bytes, 1) : malloc(total_meta_bytes)
        );
        if (compressed_level0_meta_memory_ == nullptr) {
            throw std::runtime_error("Not enough memory for compressed level-0 metadata");
        }
        compressed_level0_meta_memory_bytes_ = total_meta_bytes;

        compressed_deleted_marks_ = static_cast<unsigned char*>(calloc(element_count, sizeof(unsigned char)));
        if (compressed_deleted_marks_ == nullptr) {
            freeCompressedLevel0Storage();
            throw std::runtime_error("Not enough memory for compressed deleted marks");
        }
        compressed_deleted_marks_count_ = element_count;

        compressed_level0_neighbor_meta_ = static_cast<CompressedNeighborListMeta*>(
            calloc(element_count, sizeof(CompressedNeighborListMeta))
        );
        if (compressed_level0_neighbor_meta_ == nullptr) {
            freeCompressedLevel0Storage();
            throw std::runtime_error("Not enough memory for compressed level-0 neighbor metadata");
        }
        compressed_level0_neighbor_meta_count_ = element_count;
    }

    void freeHybridLevel0DeltaStorage() {
        hybrid_level0_neighbors_node_local_ = false;
    }

    void allocateHybridLevel0DeltaStorage(size_t element_count) {
        (void)element_count;
        hybrid_level0_neighbors_node_local_ = true;
    }

    bool hasCompressedLevel0Meta() const {
        return compact_level0_storage_enabled_ && compressed_level0_meta_memory_ != nullptr;
    }

    bool hasCompressedDeletedMarks() const {
        return compact_level0_storage_enabled_ && compressed_deleted_marks_ != nullptr;
    }

    bool hybridConstructionActive() const {
        return effective_construction_graph_mode_ == ConstructionGraphMode::Hybrid &&
               compact_level0_storage_enabled_ &&
               !compressed_ids_enabled_;
    }

    bool useCompactLevel0Adjacency() const {
        return compact_level0_storage_enabled_;
    }

    bool levelHasCompressedAdjacency(tableint internal_id, int level) const {
        if (level == 0) {
            return useCompactLevel0Adjacency();
        }
        return compressed_ids_enabled_ &&
               internal_id < compressed_upper_neighbor_meta_.size() &&
               static_cast<size_t>(level - 1) < compressed_upper_neighbor_meta_[internal_id].size();
    }

    size_t labelStorageOffset() const {
        return hasCompressedLevel0Meta() ? (label_offset_ - offsetData_) : label_offset_;
    }

    size_t parentIdStorageOffset() const {
        return hasCompressedLevel0Meta() ? (parent_id_offset_ - offsetData_) : parent_id_offset_;
    }

    size_t exFactorStorageOffset() const {
        return hasCompressedLevel0Meta() ? (ex_factor_offset_ - offsetData_) : ex_factor_offset_;
    }

    char* level0MetaBase(tableint internal_id) const {
        if (hasCompressedLevel0Meta()) {
            return compressed_level0_meta_memory_ + internal_id * compactLevel0MetaBytes();
        }
        return data_level0_memory_ + internal_id * size_data_per_element_;
    }

    static inline void appendVarUint(std::vector<uint8_t>& blob, uint32_t value) {
        while (value >= 0x80U) {
            blob.push_back(static_cast<uint8_t>((value & 0x7FU) | 0x80U));
            value >>= 7U;
        }
        blob.push_back(static_cast<uint8_t>(value));
    }

    static CompressedNeighborListMeta encodeCompressedNeighborList(
        tableint internal_id,
        const std::vector<tableint>& input_neighbors,
        bool sort_and_dedup,
        std::vector<uint8_t>& blob
    ) {
        std::vector<tableint> neighbors = input_neighbors;
        neighbors.erase(
            std::remove(neighbors.begin(), neighbors.end(), internal_id),
            neighbors.end()
        );
        if (sort_and_dedup) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }

        CompressedNeighborListMeta meta;
        meta.offset = blob.size();
        meta.count = static_cast<uint32_t>(neighbors.size());
        uint32_t prev = 0;
        for (tableint neighbor : neighbors) {
            const uint32_t delta = static_cast<uint32_t>(neighbor) - prev;
            appendVarUint(blob, delta);
            prev = static_cast<uint32_t>(neighbor);
        }
        meta.byte_size = static_cast<uint32_t>(blob.size() - meta.offset);
        return meta;
    }

    SerializedAdjacencySnapshot buildSerializedHybridAdjacencySnapshot(bool sort_and_dedup) const {
        SerializedAdjacencySnapshot snapshot;
        const size_t element_count = cur_element_count;

        snapshot.level0_neighbor_meta.resize(element_count);
        snapshot.upper_neighbor_meta.assign(element_count, {});
        snapshot.level0_neighbor_blob.reserve(compressed_level0_neighbor_blob_.size() + hybrid_delta_bytes_);

        size_t upper_raw_bytes = 0;
        for (size_t internal_id = 0; internal_id < element_count; ++internal_id) {
            if (element_levels_[internal_id] > 0) {
                upper_raw_bytes += static_cast<size_t>(element_levels_[internal_id]) * size_links_per_element_;
            }
            std::vector<tableint> level0_neighbors = getConnectionsNoLock(static_cast<tableint>(internal_id), 0);
            snapshot.level0_neighbor_meta[internal_id] = encodeCompressedNeighborList(
                static_cast<tableint>(internal_id),
                level0_neighbors,
                sort_and_dedup,
                snapshot.level0_neighbor_blob
            );
            snapshot.upper_neighbor_meta[internal_id].reserve(static_cast<size_t>(element_levels_[internal_id]));
        }
        snapshot.upper_neighbor_blob.reserve(upper_raw_bytes);

        for (size_t internal_id = 0; internal_id < element_count; ++internal_id) {
            for (int level = 1; level <= element_levels_[internal_id]; ++level) {
                std::vector<tableint> neighbors = getConnectionsNoLock(static_cast<tableint>(internal_id), level);
                snapshot.upper_neighbor_meta[internal_id].push_back(
                    encodeCompressedNeighborList(
                        static_cast<tableint>(internal_id),
                        neighbors,
                        sort_and_dedup,
                        snapshot.upper_neighbor_blob
                    )
                );
            }
        }
        return snapshot;
    }

    template <typename Function>
    inline void decodeCompressedNeighborSpan(
        const uint8_t* ptr,
        uint32_t byte_size,
        Function&& fn
    ) const {
        if (ptr == nullptr || byte_size == 0) {
            return;
        }
        const uint8_t* end = ptr + byte_size;
        uint32_t current = 0;
        uint32_t value = 0;
        int shift = 0;
        while (ptr < end) {
            const uint8_t byte = *ptr++;
            value |= static_cast<uint32_t>(byte & 0x7FU) << shift;
            if (byte & 0x80U) {
                shift += 7;
                continue;
            }
            current += value;
            fn(static_cast<tableint>(current));
            value = 0;
            shift = 0;
        }
    }

    template <typename Function>
    inline void decodeCompressedNeighborBlob(
        const std::vector<uint8_t>& blob,
        const CompressedNeighborListMeta& meta,
        Function&& fn
    ) const {
        if (meta.byte_size == 0) {
            return;
        }
        decodeCompressedNeighborSpan(
            blob.data() + meta.offset,
            meta.byte_size,
            std::forward<Function>(fn)
        );
    }

    inline const tableint* tryGetRawNeighbors(tableint internal_id, int level, size_t& size) const {
        if (levelHasCompressedAdjacency(internal_id, level)) {
            if (level == 0) {
                size = internal_id < compressed_level0_neighbor_meta_count_
                    ? compressed_level0_neighbor_meta_[internal_id].count
                    : 0;
            } else {
                size = (internal_id < compressed_upper_neighbor_meta_.size() &&
                        static_cast<size_t>(level - 1) < compressed_upper_neighbor_meta_[internal_id].size())
                    ? compressed_upper_neighbor_meta_[internal_id][static_cast<size_t>(level - 1)].count
                    : 0;
            }
            return nullptr;
        }

        int* data = (level == 0) ? (int*)get_linklist0(internal_id) : (int*)get_linklist(internal_id, level);
        size = getListCount((linklistsizeint*)data);
        return reinterpret_cast<tableint*>(data + 1);
    }

    template <typename Function>
    inline void forEachNeighbor(tableint internal_id, int level, Function&& fn) const {
        size_t size = 0;
        const tableint* raw_neighbors = tryGetRawNeighbors(internal_id, level, size);
        if (raw_neighbors != nullptr) {
            for (size_t i = 0; i < size; ++i) {
                fn(raw_neighbors[i]);
            }
            return;
        }

        if (level == 0 && hybrid_level0_neighbors_node_local_) {
            if (internal_id < compressed_level0_neighbor_meta_count_) {
                const CompressedNeighborListMeta& meta = compressed_level0_neighbor_meta_[internal_id];
                decodeCompressedNeighborSpan(
                    hybridLevel0CompressedData(internal_id),
                    meta.byte_size,
                    std::forward<Function>(fn)
                );
            }
        } else if (level == 0) {
            decodeCompressedNeighborBlob(
                compressed_level0_neighbor_blob_,
                compressed_level0_neighbor_meta_[internal_id],
                std::forward<Function>(fn)
            );
        } else {
            decodeCompressedNeighborBlob(
                compressed_upper_neighbor_blob_,
                compressed_upper_neighbor_meta_[internal_id][static_cast<size_t>(level - 1)],
                std::forward<Function>(fn)
            );
        }
    }

    void configureQuantizedDistance() {
        if (ex_bits_ != 8 &&
            ex_bits_ != 4 &&
            ex_bits_ != 6 &&
            ex_bits_ != 2 &&
            ex_bits_ != 3 &&
            ex_bits_ != 7) {
            throw std::runtime_error("Unsupported number of quantization bits");
        }
        FAC_RESCALE = 1 << ex_bits_;
    }

    void allocateScalarCodeStorage() {
        if (scalar_codes_ != nullptr || vector_dim_ == 0) {
            return;
        }
        scalar_codes_ = (int8_t*)malloc(max_elements_ * vector_dim_ * sizeof(int8_t));
        if (scalar_codes_ == nullptr) {
            throw std::runtime_error("Not enough memory for scalar int8 codes");
        }
    }

    size_t packedShortCodeBytesPerBlock() const {
        return short_code_length_ * FAST_SIZE;
    }

    size_t packedShortCodeBlockCount(size_t elements) const {
        return div_rd_up(elements, FAST_SIZE);
    }

    void freePackedShortCodeStorage() const {
        free(packed_short_codes_);
        packed_short_codes_ = nullptr;
    }

    void allocatePackedShortCodeStorage() const {
        if (quantization_method_ != QuantizationMethod::ExRaBitQ || short_code_length_ == 0) {
            return;
        }
        size_t total_blocks = packedShortCodeBlockCount(max_elements_);
        if (total_blocks == 0) {
            return;
        }
        freePackedShortCodeStorage();
        packed_short_codes_ = (uint8_t*)malloc(total_blocks * packedShortCodeBytesPerBlock());
        if (packed_short_codes_ == nullptr) {
            throw std::runtime_error("Not enough memory for packed FastScan short codes");
        }
        std::memset(packed_short_codes_, 0, total_blocks * packedShortCodeBytesPerBlock());
    }

    void rebuildPackedShortCodeBlocks() const {
        if (quantization_method_ != QuantizationMethod::ExRaBitQ || short_code_length_ == 0) {
            return;
        }
        if (packed_short_codes_ == nullptr) {
            allocatePackedShortCodeStorage();
        }

        const size_t blocks = packedShortCodeBlockCount(cur_element_count);
        const size_t codes_per_vector_u64 = quantized_dim_ / 64;
        std::vector<uint64_t> block_binary(FAST_SIZE * codes_per_vector_u64, 0);
        for (size_t block_idx = 0; block_idx < blocks; ++block_idx) {
            std::fill(block_binary.begin(), block_binary.end(), 0);
            const size_t block_start = block_idx * FAST_SIZE;
            const size_t block_count = std::min<size_t>(FAST_SIZE, cur_element_count - block_start);
            for (size_t slot = 0; slot < block_count; ++slot) {
                const uint8_t* short_code = short_codes_ + (block_start + slot) * short_code_length_;
                for (size_t chunk = 0; chunk < codes_per_vector_u64; ++chunk) {
                    uint64_t cur = 0;
                    const size_t bit_base = chunk * 64;
                    for (size_t bit = 0; bit < 64; ++bit) {
                        const size_t global_bit = bit_base + bit;
                        const size_t byte_idx = global_bit / 8;
                        const size_t bit_idx = global_bit % 8;
                        const uint64_t bit_value = (short_code[byte_idx] >> (7 - bit_idx)) & 1U;
                        cur |= (bit_value << (63 - bit));
                    }
                    block_binary[slot * codes_per_vector_u64 + chunk] = cur;
                }
            }
            pack_codes(
                quantized_dim_,
                block_binary.data(),
                block_count,
                packed_short_codes_ + block_idx * packedShortCodeBytesPerBlock()
            );
        }
    }

    void allocateNodeStorage(size_t max_elements, bool allocate_raw_data = true, bool allocate_level0_links = true) {
        if (allocate_level0_links) {
            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");
        } else {
            data_level0_memory_ = nullptr;
        }

        if (allocate_raw_data) {
            raw_data_memory_ = (char *) malloc(max_elements * data_size_);
            if (raw_data_memory_ == nullptr)
                throw std::runtime_error("Not enough memory for raw data");
        } else {
            raw_data_memory_ = nullptr;
        }

        if (short_code_length_ > 0) {
            short_codes_ = (uint8_t*)malloc(max_elements * short_code_length_ * sizeof(uint8_t));
            if (short_codes_ == nullptr) {
                throw std::runtime_error("Not enough memory for short codes");
            }
        } else {
            short_codes_ = nullptr;
        }

        if (long_code_length_ > 0) {
            long_codes_ = (uint8_t*)malloc(max_elements * long_code_length_ * sizeof(uint8_t));
            if (long_codes_ == nullptr) {
                throw std::runtime_error("Not enough memory for long codes");
            }
        } else {
            long_codes_ = nullptr;
        }
    }

    static size_t bytesFromGigabytes(double gib) {
        if (gib <= 0.0) {
            return 0;
        }
        return static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
    }

    static size_t readMemAvailableBytes() {
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open()) {
            return 0;
        }

        std::string key;
        size_t value_kb = 0;
        std::string unit;
        while (meminfo >> key >> value_kb >> unit) {
            if (key == "MemAvailable:") {
                return value_kb * 1024ULL;
            }
        }
        return 0;
    }

    static size_t readPhysicalMemoryBytes() {
        const long page_count = sysconf(_SC_PHYS_PAGES);
        const long page_size = sysconf(_SC_PAGE_SIZE);
        if (page_count <= 0 || page_size <= 0) {
            return 0;
        }
        return static_cast<size_t>(page_count) * static_cast<size_t>(page_size);
    }

    static size_t resolveConstructionRamLimitBytes(const ConstructionGraphOptions& options) {
        const size_t configured_bytes = bytesFromGigabytes(options.ram_limit_gb);
        if (configured_bytes > 0) {
            return configured_bytes;
        }

        const size_t reserve_bytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
        size_t available_bytes = readMemAvailableBytes();
        if (available_bytes == 0) {
            available_bytes = readPhysicalMemoryBytes();
        }

        if (available_bytes <= reserve_bytes) {
            return 0;
        }
        return available_bytes - reserve_bytes;
    }

    static const char* constructionModeName(ConstructionGraphMode mode) {
        switch (mode) {
            case ConstructionGraphMode::Raw: return "raw";
            case ConstructionGraphMode::Hybrid: return "hybrid";
            case ConstructionGraphMode::Auto: return "auto";
        }
        return "unknown";
    }

    static ConstructionGraphMode resolveEffectiveConstructionGraphMode(
        ConstructionGraphMode requested_mode,
        size_t estimated_raw_adjacency_bytes,
        size_t ram_limit_bytes
    ) {
        if (requested_mode != ConstructionGraphMode::Auto) {
            return requested_mode;
        }
        return estimated_raw_adjacency_bytes > ram_limit_bytes
            ? ConstructionGraphMode::Hybrid
            : ConstructionGraphMode::Raw;
    }

    size_t estimateRawConstructionAdjacencyBytes(size_t element_count) const {
        if (element_count == 0) {
            return 0;
        }
        const double upper_level_multiplier = (M_ > 1) ? (1.0 / static_cast<double>(M_ - 1)) : 1.0;
        const double estimated =
            static_cast<double>(element_count) * static_cast<double>(size_links_level0_) +
            static_cast<double>(element_count) * upper_level_multiplier * static_cast<double>(size_links_per_element_);
        return static_cast<size_t>(estimated);
    }

    void configureConstructionGraphMode(size_t max_elements) {
        const ConstructionGraphMode requested_mode = construction_graph_options_.mode;
        construction_ram_limit_bytes_ = resolveConstructionRamLimitBytes(construction_graph_options_);
        construction_estimated_raw_adjacency_bytes_ = estimateRawConstructionAdjacencyBytes(max_elements);
        effective_construction_graph_mode_ = resolveEffectiveConstructionGraphMode(
            construction_graph_options_.mode,
            construction_estimated_raw_adjacency_bytes_,
            construction_ram_limit_bytes_
        );
        if (construction_graph_options_.log_stats) {
            HNSWERR << "construction_graph_mode_requested=" << constructionModeName(requested_mode)
                    << " effective=" << constructionModeName(effective_construction_graph_mode_)
                    << " estimated_raw_adjacency_bytes=" << construction_estimated_raw_adjacency_bytes_
                    << " ram_limit_bytes=" << construction_ram_limit_bytes_
                    << " ram_limit_source=" << (construction_graph_options_.ram_limit_gb > 0.0 ? "configured" : "available_minus_10g")
                    << std::endl;
        }
    }

    void enableHybridLevel0Storage() {
        compact_level0_storage_enabled_ = true;
        allocateCompressedLevel0Storage(max_elements_, true);
        allocateHybridLevel0DeltaStorage(max_elements_);
        hybrid_delta_bytes_ = 0;
        hybrid_delta_peak_bytes_ = 0;
        hybrid_dirty_node_count_ = 0;
        hybrid_level0_active_base_bytes_ =
            compressedLevel0MetaMemoryBytes() +
            compressed_deleted_marks_count_ * sizeof(unsigned char);
    }

    bool hasExternalRawDataView() const {
        return external_raw_data_view_ != nullptr;
    }

    bool rawDataAvailable() const {
        return raw_data_memory_ != nullptr || hasExternalRawDataView();
    }

    void requireRawData(const char* operation) const {
        if (!rawDataAvailable()) {
            throw std::runtime_error(std::string(operation) + " requires raw vectors, but this index was compacted to quantized-only storage");
        }
    }

    void requireOwnedRawData(const char* operation) const {
        if (raw_data_memory_ == nullptr) {
            if (hasExternalRawDataView()) {
                throw std::runtime_error(
                    std::string(operation) +
                    " requires writable index-owned raw vectors, but external build raw backing is active"
                );
            }
            throw std::runtime_error(
                std::string(operation) +
                " requires raw vectors, but this index was compacted to quantized-only storage"
            );
        }
    }

    void clearExternalRawDataView() {
        external_raw_data_view_ = nullptr;
        external_raw_data_stride_ = 0;
        external_raw_data_count_ = 0;
        external_raw_row_ids_.clear();
    }

    void assignExternalRawRow(tableint internal_id, const void* data_point) {
        if (!hasExternalRawDataView()) {
            return;
        }
        const char* raw_ptr = static_cast<const char*>(data_point);
        if (raw_ptr < external_raw_data_view_) {
            throw std::runtime_error("External raw data pointer is outside the registered source matrix");
        }
        const size_t byte_offset = static_cast<size_t>(raw_ptr - external_raw_data_view_);
        if (external_raw_data_stride_ == 0 || (byte_offset % external_raw_data_stride_) != 0) {
            throw std::runtime_error("External raw data pointer does not align with the registered source matrix stride");
        }
        const size_t row = byte_offset / external_raw_data_stride_;
        if (row >= external_raw_data_count_) {
            throw std::runtime_error("External raw data pointer exceeds the registered source matrix");
        }
        if (internal_id >= external_raw_row_ids_.size()) {
            throw std::runtime_error("Internal id exceeds external raw row mapping capacity");
        }
        external_raw_row_ids_[internal_id] = static_cast<uint32_t>(row);
    }

    void prepareQuantizedQuery(
        const float* query_data,
        float* unit_q,
        int16_t* quant_query,
        float& sumq,
        float& delta,
        size_t& D
    ) const {
        D = quantized_dim_ != 0 ? quantized_dim_ : short_code_length_ * 8;

        if (quantization_enabled_) {
            FloatRowMat padded_query(1, D);
            padded_query.setZero();
            std::memcpy(padded_query.data(), query_data, sizeof(float) * vector_dim_);
            FloatRowMat rotated_query(1, D);
            rotator_.rotate(padded_query, rotated_query);
            sumq = normalize_query16(unit_q, rotated_query.data(), D);
        } else {
            sumq = normalize_query16(unit_q, query_data, D);
        }

        high_acc_quantize16(quant_query, unit_q, delta, D);
    }

    void prepareQuantizedQuery(
        const float* query_data,
        PreprocessedQuantizedQuery& prepared_query
    ) const {
        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            prepared_query.resize_scaled(vector_dim_);
            scalar_quantizer_.prepareQuery(query_data, prepared_query.scaled_query.data());
            prepared_query.sumq = 0.0f;
            prepared_query.delta = 0.0f;
            return;
        }

        size_t D = quantized_dim_ != 0 ? quantized_dim_ : short_code_length_ * 8;
        prepared_query.resize(D);
        prepareQuantizedQuery(
            query_data,
            prepared_query.unit_q.data(),
            prepared_query.quant_query.data(),
            prepared_query.sumq,
            prepared_query.delta,
            prepared_query.D
        );
        prepared_query.short_shift = 0;
        prepared_query.resize_fastscan_lut(D * 8);
        pack_high_acc_LUT(
            prepared_query.quant_query.data(),
            D,
            prepared_query.fastscan_lut.data(),
            prepared_query.short_shift
        );
    }

    PreprocessedQuantizedQuery preprocessQueryQuantization(const void *query_data) const {
        PreprocessedQuantizedQuery prepared_query;
        prepareQuantizedQuery(static_cast<const float*>(query_data), prepared_query);
        return prepared_query;
    }

    inline const uint8_t* getPackedShortCodesBase() const {
        return packed_short_codes_;
    }

    inline int8_t* getScalarCode(tableint internal_id) const {
        return scalar_codes_ + internal_id * vector_dim_;
    }

    inline void setScalarCode(tableint internal_id, const int8_t* scalar_code) const {
        memcpy(getScalarCode(internal_id), scalar_code, vector_dim_ * sizeof(int8_t));
    }

    dist_t computeScalarQuantizedDistance(const float* scaled_query, tableint internal_id) const {
        float raw_dot = scalar_int8_inner_product(scaled_query, getScalarCode(internal_id), vector_dim_);
        return 1 - raw_dot;
    }

    float getPackedShortScore(
        const PreprocessedQuantizedQuery& prepared_query,
        tableint internal_id,
        FastScanScratch& scratch
    ) const {
        const size_t row = static_cast<size_t>(internal_id);
        const size_t block_idx = row / FAST_SIZE;
        const size_t local_block_idx = block_idx - scratch.block_base;
        const size_t slot = row % FAST_SIZE;
        const uint8_t* packed_short_codes_base = getPackedShortCodesBase();
        if (scratch.block_epochs[local_block_idx] != scratch.current_epoch) {
            accumulate_one_block_high_acc(
                packed_short_codes_base + block_idx * packedShortCodeBytesPerBlock(),
                prepared_query.fastscan_lut.data(),
                prepared_query.delta,
                prepared_query.short_shift,
                scratch.block_scores.data() + local_block_idx * FAST_SIZE,
                prepared_query.D
            );
            scratch.block_epochs[local_block_idx] = scratch.current_epoch;
        }
        return scratch.block_scores[local_block_idx * FAST_SIZE + slot];
    }

    template <int ex_bits>
    static inline float computeLongCodeInnerProduct(const float* unit_q, const uint8_t* long_code, size_t D) {
        if constexpr (ex_bits == 8) {
            return IP16_fxu8(unit_q, long_code, D);
        } else if constexpr (ex_bits == 4) {
            return IP32_fxu4(unit_q, long_code, D);
        } else if constexpr (ex_bits == 6) {
            return IP64_fxu6(unit_q, long_code, D);
        } else if constexpr (ex_bits == 2) {
            return IP64_fxu2(unit_q, long_code, D);
        } else if constexpr (ex_bits == 3) {
            return IP64_fxu3(unit_q, long_code, D);
        } else if constexpr (ex_bits == 7) {
            return IP64_fxu7(unit_q, long_code, D);
        } else {
            static_assert(
                ex_bits == 8 || ex_bits == 4 || ex_bits == 6 ||
                ex_bits == 2 || ex_bits == 3 || ex_bits == 7,
                "Unsupported ExRaBitQ bit width"
            );
            return 0.0f;
        }
    }

    template <int ex_bits>
    dist_t computeExRaBitQApproxDistance(
        const PreprocessedQuantizedQuery& prepared_query,
        tableint internal_id,
        FastScanScratch& scratch
    ) const {
        return computeExRaBitQRefinedDistance<ex_bits>(prepared_query, internal_id, scratch);
    }

    template <int ex_bits>
    dist_t computeExRaBitQRefinedDistance(
        const PreprocessedQuantizedQuery& prepared_query,
        tableint internal_id,
        FastScanScratch& scratch
    ) const {
        constexpr float fac_rescale = static_cast<float>(1 << ex_bits);
        float short_score = getPackedShortScore(prepared_query, internal_id, scratch);
        uint8_t* long_code = getLongCode(internal_id);
        ExFactor* ex_factor = getExFactor(internal_id);
        float raw_dot = fac_rescale * short_score +
                        computeLongCodeInnerProduct<ex_bits>(prepared_query.unit_q.data(), long_code, prepared_query.D) -
                        fac_rescale * prepared_query.sumq;
        float ex_dist = ex_factor->xipnorm * raw_dot;
        return 1 - ex_dist;
    }

    dist_t computeQuantizedDistance(
        const PreprocessedQuantizedQuery& prepared_query,
        tableint internal_id,
        FastScanScratch& scratch
    ) const {
        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            return computeScalarQuantizedDistance(prepared_query.scaled_query.data(), internal_id);
        }

        switch (ex_bits_) {
            case 8:
                return computeExRaBitQApproxDistance<8>(prepared_query, internal_id, scratch);
            case 7:
                return computeExRaBitQApproxDistance<7>(prepared_query, internal_id, scratch);
            case 6:
                return computeExRaBitQApproxDistance<6>(prepared_query, internal_id, scratch);
            case 4:
                return computeExRaBitQApproxDistance<4>(prepared_query, internal_id, scratch);
            case 3:
                return computeExRaBitQApproxDistance<3>(prepared_query, internal_id, scratch);
            case 2:
                return computeExRaBitQApproxDistance<2>(prepared_query, internal_id, scratch);
            default:
                throw std::runtime_error("Unsupported quantized distance configuration");
        }
    }

    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t short_code_length,
        size_t long_code_length,
        int ex_bits,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted),
          short_code_length_(short_code_length),
          long_code_length_(long_code_length),
          ex_bits_(ex_bits) {
        if (ex_bits_ > 0) {
            configureQuantizedDistance();
        }
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t short_code_length,
        size_t long_code_length,
        int ex_bits,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false,
        bool allocate_raw_data = true,
        const ConstructionGraphOptions& construction_graph_options = ConstructionGraphOptions())
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted),
            construction_graph_options_(construction_graph_options) {
        max_elements_ = max_elements;
        short_code_length_ = short_code_length;
        long_code_length_ = long_code_length;
        ex_bits_ = ex_bits;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        vector_dim_ = *((size_t*)s->get_dist_func_param());
        quantized_dim_ = short_code_length_ * 8;
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if (ex_bits_ > 0) {
            configureQuantizedDistance();
        }
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        configureStorageLayout();
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
        configureConstructionGraphMode(max_elements_);
        allocateNodeStorage(
            max_elements_,
            allocate_raw_data,
            effective_construction_graph_mode_ != ConstructionGraphMode::Hybrid
        );

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        std::memset(linkLists_, 0, sizeof(void *) * max_elements_);
        if (effective_construction_graph_mode_ == ConstructionGraphMode::Hybrid) {
            if (allow_replace_deleted_) {
                throw std::runtime_error("Hybrid construction mode does not support replace_deleted");
            }
            enableHybridLevel0Storage();
        }
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        free(raw_data_memory_);
        raw_data_memory_ = nullptr;
        clearExternalRawDataView();
        free(short_codes_);
        short_codes_ = nullptr;
        free(long_codes_);
        long_codes_ = nullptr;
        freePackedShortCodeStorage();
        free(scalar_codes_);
        scalar_codes_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (linkLists_ != nullptr && linkLists_[i] != nullptr)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        visited_list_pool_.reset(nullptr);
        parent_id_to_labels_.clear();
        compressed_ids_enabled_ = false;
        freeCompressedLevel0Storage();
        compressed_upper_neighbor_meta_.clear();
        compressed_level0_neighbor_blob_.clear();
        compressed_upper_neighbor_blob_.clear();
        freeHybridLevel0DeltaStorage();
        compact_level0_storage_enabled_ = false;
        cur_element_count = 0;
        hybrid_delta_bytes_ = 0;
        hybrid_delta_peak_bytes_ = 0;
        hybrid_dirty_node_count_ = 0;
        hybrid_level0_active_base_bytes_ = 0;
        hybrid_level0_merge_count_ = 0;
        hybrid_level0_merge_seconds_ = 0.0;
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, level0MetaBase(internal_id) + labelStorageOffset(), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy(level0MetaBase(internal_id) + labelStorageOffset(), &label, sizeof(labeltype));
    }


    inline labeltype getParentId(tableint internal_id) const {
        labeltype return_id;
        memcpy(&return_id, level0MetaBase(internal_id) + parentIdStorageOffset(), sizeof(labeltype));
        return return_id;
    }

    inline void setParentId(tableint internal_id, labeltype parent_id) const {
        memcpy(level0MetaBase(internal_id) + parentIdStorageOffset(), &parent_id, sizeof(labeltype));
    }


    inline uint8_t* getShortCode(tableint internal_id) const {
        return short_codes_ + internal_id * short_code_length_;
    }

    inline void setShortCode(tableint internal_id, const uint8_t* short_code) const {
        memcpy(short_codes_ + internal_id * short_code_length_, short_code, short_code_length_ * sizeof(uint8_t));
    }

    inline uint8_t* getLongCode(tableint internal_id) const {
        return long_codes_ + internal_id * long_code_length_;
    }

    inline void setLongCode(tableint internal_id, const uint8_t* long_code) const {
        memcpy(long_codes_ + internal_id * long_code_length_, long_code, long_code_length_ * sizeof(uint8_t));
    }

    inline ExFactor* getExFactor(tableint internal_id) const {
        return (ExFactor*)(level0MetaBase(internal_id) + exFactorStorageOffset());
    }

    inline void setExFactor(tableint internal_id, const ExFactor* ex_factor) const {
        memcpy(level0MetaBase(internal_id) + exFactorStorageOffset(), ex_factor, sizeof(ExFactor));
    }

    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (level0MetaBase(internal_id) + labelStorageOffset());
    }


    inline char *tryGetDataByInternalId(tableint internal_id) const {
        if (raw_data_memory_) {
            if (internal_id >= max_elements_) {
                return nullptr;
            }
            return raw_data_memory_ + internal_id * data_size_;
        }
        if (!hasExternalRawDataView()) {
            return nullptr;
        }
        if (internal_id >= external_raw_row_ids_.size()) {
            return nullptr;
        }
        const size_t row = external_raw_row_ids_[internal_id];
        if (row == INVALID_EXTERNAL_RAW_ROW || row >= external_raw_data_count_) {
            return nullptr;
        }
        return const_cast<char*>(external_raw_data_view_ + row * external_raw_data_stride_);
    }

    inline char *getDataByInternalId(tableint internal_id) const {
        char* data_ptr = tryGetDataByInternalId(internal_id);
        assert(data_ptr != nullptr);
        return data_ptr;
    }

    inline void prefetchDataByInternalId(tableint internal_id) const {
#ifdef USE_SSE
        char* data_ptr = tryGetDataByInternalId(internal_id);
        if (data_ptr != nullptr) {
            _mm_prefetch(data_ptr, _MM_HINT_T0);
        }
#else
        (void)internal_id;
#endif
    }

    void setQuantizationModel(
        const DataQuantizer& quantizer,
        const Rotator& rotator,
        bool persist_raw_vectors = false
    ) {
        if (cur_element_count != 0) {
            throw std::runtime_error("Quantization model must be configured before inserting elements");
        }

        if (quantizer.short_code_length() != short_code_length_ ||
            quantizer.long_code_length() != long_code_length_ ||
            static_cast<int>(quantizer.ex_bits()) != ex_bits_) {
            throw std::runtime_error("Quantizer configuration does not match the index layout");
        }

        if (quantizer.dim() * sizeof(float) != data_size_) {
            throw std::runtime_error("Quantizer dimension does not match the space dimension");
        }

        quantizer_ = quantizer;
        quantization_method_ = QuantizationMethod::ExRaBitQ;
        rotator_ = rotator;
        vector_dim_ = quantizer.dim();
        quantized_dim_ = quantizer.padded_dim();
        ex_bits_ = quantizer.ex_bits();
        configureQuantizedDistance();
        quantization_enabled_ = true;
        raw_data_compacted_ = false;
        persist_raw_vectors_ = persist_raw_vectors;
        allocatePackedShortCodeStorage();
    }

    void setScalarInt8Model(
        const ScalarInt8Quantizer& quantizer,
        bool persist_raw_vectors = false
    ) {
        if (cur_element_count != 0) {
            throw std::runtime_error("Scalar quantization model must be configured before inserting elements");
        }
        if (quantizer.dim() * sizeof(float) != data_size_) {
            throw std::runtime_error("Scalar quantizer dimension does not match the space dimension");
        }

        scalar_quantizer_ = quantizer;
        quantization_method_ = QuantizationMethod::ScalarInt8;
        quantization_enabled_ = true;
        raw_data_compacted_ = false;
        persist_raw_vectors_ = persist_raw_vectors;
        vector_dim_ = quantizer.dim();
        quantized_dim_ = vector_dim_;
        short_code_length_ = 0;
        long_code_length_ = 0;
        ex_bits_ = 0;
        allocateScalarCodeStorage();
    }

    void setExternalBuildRawData(const void* raw_data_base, size_t row_stride_bytes, size_t row_count) {
        if (cur_element_count != 0) {
            throw std::runtime_error("External build raw data must be configured before inserting elements");
        }
        if (raw_data_compacted_) {
            throw std::runtime_error("Cannot configure external build raw data after raw vectors were compacted");
        }
        if (persist_raw_vectors_) {
            throw std::runtime_error("External build raw data requires persist_raw_vectors=false");
        }
        if (raw_data_base == nullptr) {
            throw std::runtime_error("External build raw data pointer cannot be null");
        }
        if (row_stride_bytes < data_size_) {
            throw std::runtime_error("External build raw data stride is smaller than the vector size");
        }
        if (row_count == 0) {
            throw std::runtime_error("External build raw data must contain at least one row");
        }
        if (row_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("External build raw data row count exceeds uint32_t capacity");
        }

        free(raw_data_memory_);
        raw_data_memory_ = nullptr;
        external_raw_data_view_ = static_cast<const char*>(raw_data_base);
        external_raw_data_stride_ = row_stride_bytes;
        external_raw_data_count_ = row_count;
        external_raw_row_ids_.assign(max_elements_, INVALID_EXTERNAL_RAW_ROW);
    }

    void finalizeQuantizedIndex() {
        if (!quantization_enabled_) {
            throw std::runtime_error("Cannot finalize a quantized index before configuring quantization");
        }
        if (quantization_method_ == QuantizationMethod::ExRaBitQ) {
            rebuildPackedShortCodeBlocks();
        }

        free(raw_data_memory_);
        raw_data_memory_ = nullptr;
        clearExternalRawDataView();
        raw_data_compacted_ = true;
        persist_raw_vectors_ = false;
    }

    bool isQuantizedOnly() const {
        return raw_data_compacted_;
    }

    void replaceHybridLevel0Neighbors(tableint internal_id, const std::vector<tableint>& input_neighbors) {
        if (!hybridConstructionActive() || !hybrid_level0_neighbors_node_local_) {
            throw std::runtime_error("Hybrid level-0 node-local storage is not active");
        }

        std::vector<uint8_t> encoded_blob;
        const CompressedNeighborListMeta encoded_meta = encodeCompressedNeighborList(
            internal_id,
            input_neighbors,
            true,
            encoded_blob
        );

        uint8_t* new_data = nullptr;
        if (!encoded_blob.empty()) {
            new_data = static_cast<uint8_t*>(malloc(encoded_blob.size()));
            if (new_data == nullptr) {
                throw std::runtime_error("Not enough memory for hybrid level-0 neighbor update");
            }
            std::memcpy(new_data, encoded_blob.data(), encoded_blob.size());
        }

        CompressedNeighborListMeta& stored_meta = compressed_level0_neighbor_meta_[internal_id];
        uint8_t* old_data = hybridLevel0CompressedData(internal_id);
        hybrid_delta_bytes_ -= stored_meta.byte_size;
        hybrid_delta_bytes_ += encoded_meta.byte_size;
        hybrid_delta_peak_bytes_ = std::max(hybrid_delta_peak_bytes_, hybrid_delta_bytes_);
        stored_meta.offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(new_data));
        stored_meta.byte_size = encoded_meta.byte_size;
        stored_meta.count = encoded_meta.count;
        free(old_data);

        hybrid_level0_active_base_bytes_ =
            compressedLevel0MetaMemoryBytes() +
            compressed_deleted_marks_count_ * sizeof(unsigned char) +
            hybrid_delta_bytes_;
    }

    void materializeHybridLevel0NodeLocalStorage() {
        if (!hybrid_level0_neighbors_node_local_) {
            return;
        }

        std::vector<uint8_t> new_blob;
        new_blob.reserve(hybrid_delta_bytes_);
        for (size_t internal_id = 0; internal_id < cur_element_count; ++internal_id) {
            CompressedNeighborListMeta& meta = compressed_level0_neighbor_meta_[internal_id];
            const uint8_t* node_data = hybridLevel0CompressedData(static_cast<tableint>(internal_id));
            const uint64_t new_offset = new_blob.size();
            if (node_data != nullptr && meta.byte_size > 0) {
                new_blob.insert(new_blob.end(), node_data, node_data + meta.byte_size);
            }
            free(const_cast<uint8_t*>(node_data));
            meta.offset = new_offset;
        }
        for (size_t internal_id = cur_element_count; internal_id < compressed_level0_neighbor_meta_count_; ++internal_id) {
            free(hybridLevel0CompressedData(static_cast<tableint>(internal_id)));
            compressed_level0_neighbor_meta_[internal_id] = {};
        }

        compressed_level0_neighbor_blob_.swap(new_blob);
        compressed_level0_neighbor_meta_count_ = cur_element_count;
        hybrid_level0_neighbors_node_local_ = false;
        hybrid_delta_bytes_ = 0;
        hybrid_dirty_node_count_ = 0;
        hybrid_level0_active_base_bytes_ =
            compressedLevel0MetaMemoryBytes() +
            compressed_deleted_marks_count_ * sizeof(unsigned char) +
            compressed_level0_neighbor_blob_.size();
    }

    void maybeMergeHybridLevel0Delta() {
    }

    void finalizeHybridAdjacency(bool sort_and_dedup) {
        if (!compact_level0_storage_enabled_ || compressed_ids_enabled_) {
            return;
        }

        (void)sort_and_dedup;
        materializeHybridLevel0NodeLocalStorage();
        compressed_upper_neighbor_meta_.assign(cur_element_count, {});
        compressed_upper_neighbor_blob_.clear();

        std::vector<tableint> neighbors;
        for (size_t internal_id = 0; internal_id < cur_element_count; ++internal_id) {
            for (int level = 1; level <= element_levels_[internal_id]; ++level) {
                size_t size = 0;
                const tableint* raw_neighbors = tryGetRawNeighbors(static_cast<tableint>(internal_id), level, size);
                neighbors.assign(raw_neighbors, raw_neighbors + size);
                neighbors.erase(
                    std::remove(neighbors.begin(), neighbors.end(), static_cast<tableint>(internal_id)),
                    neighbors.end()
                );
                if (sort_and_dedup) {
                    std::sort(neighbors.begin(), neighbors.end());
                    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                }

                CompressedNeighborListMeta meta;
                meta.offset = compressed_upper_neighbor_blob_.size();
                meta.count = static_cast<uint32_t>(neighbors.size());
                uint32_t prev = 0;
                for (tableint neighbor : neighbors) {
                    const uint32_t delta = static_cast<uint32_t>(neighbor) - prev;
                    appendVarUint(compressed_upper_neighbor_blob_, delta);
                    prev = static_cast<uint32_t>(neighbor);
                }
                meta.byte_size = static_cast<uint32_t>(compressed_upper_neighbor_blob_.size() - meta.offset);
                compressed_upper_neighbor_meta_[internal_id].push_back(meta);
            }
        }

        for (size_t i = 0; i < cur_element_count; ++i) {
            if (linkLists_[i] != nullptr) {
                free(linkLists_[i]);
                linkLists_[i] = nullptr;
            }
        }
        compressed_ids_enabled_ = true;
        if (construction_graph_options_.log_stats) {
            HNSWERR << "construction_graph_mode_finalized=hybrid"
                    << " compressed_base_bytes=" << constructionCompressedBaseBytes()
                    << " delta_overlay_bytes=" << constructionDeltaOverlayBytes()
                    << " total_adjacency_bytes=" << constructionAdjacencyBytes()
                    << " merge_count=" << constructionMergeCount()
                    << " merge_seconds=" << constructionMergeSeconds()
                    << std::endl;
        }
    }

    void compressNeighborIdsSortedVarint(bool sort_and_dedup = true) {
        if (compressed_ids_enabled_) {
            return;
        }
        if (compact_level0_storage_enabled_) {
            finalizeHybridAdjacency(sort_and_dedup);
            return;
        }
        if (cur_element_count == 0) {
            compact_level0_storage_enabled_ = true;
            compressed_ids_enabled_ = true;
            return;
        }

        const size_t meta_bytes = compactLevel0MetaBytes();
        allocateCompressedLevel0Storage(cur_element_count, false);
        compressed_upper_neighbor_meta_.assign(cur_element_count, {});
        compressed_level0_neighbor_blob_.clear();
        compressed_upper_neighbor_blob_.clear();

        std::vector<tableint> neighbors;
        for (size_t internal_id = 0; internal_id < cur_element_count; ++internal_id) {
            std::memcpy(
                compressed_level0_meta_memory_ + internal_id * meta_bytes,
                data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_,
                meta_bytes
            );
            if (isMarkedDeleted(static_cast<tableint>(internal_id))) {
                compressed_deleted_marks_[internal_id] = DELETE_MARK;
            }

            for (int level = 0; level <= element_levels_[internal_id]; ++level) {
                size_t size = 0;
                const tableint* raw_neighbors = tryGetRawNeighbors(static_cast<tableint>(internal_id), level, size);
                neighbors.assign(raw_neighbors, raw_neighbors + size);
                neighbors.erase(
                    std::remove(neighbors.begin(), neighbors.end(), static_cast<tableint>(internal_id)),
                    neighbors.end()
                );
                if (sort_and_dedup) {
                    std::sort(neighbors.begin(), neighbors.end());
                    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                }

                CompressedNeighborListMeta meta;
                std::vector<uint8_t>& blob = (level == 0) ? compressed_level0_neighbor_blob_ : compressed_upper_neighbor_blob_;
                meta.offset = blob.size();
                meta.count = static_cast<uint32_t>(neighbors.size());

                uint32_t prev = 0;
                for (tableint neighbor : neighbors) {
                    const uint32_t delta = static_cast<uint32_t>(neighbor) - prev;
                    appendVarUint(blob, delta);
                    prev = static_cast<uint32_t>(neighbor);
                }
                meta.byte_size = static_cast<uint32_t>(blob.size() - meta.offset);

                if (level == 0) {
                    compressed_level0_neighbor_meta_[internal_id] = meta;
                } else {
                    compressed_upper_neighbor_meta_[internal_id].push_back(meta);
                }
            }
        }

        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (size_t i = 0; i < cur_element_count; ++i) {
            if (linkLists_[i] != nullptr) {
                free(linkLists_[i]);
                linkLists_[i] = nullptr;
            }
        }
        compact_level0_storage_enabled_ = true;
        compressed_ids_enabled_ = true;
    }

    size_t rawNeighborIdBytesReserved() const {
        size_t bytes = cur_element_count * size_links_level0_;
        for (size_t i = 0; i < cur_element_count; ++i) {
            if (element_levels_[i] > 0) {
                bytes += static_cast<size_t>(element_levels_[i]) * size_links_per_element_;
            }
        }
        return bytes;
    }

    size_t compressedNeighborIdBytes() const {
        size_t upper_meta_bytes = 0;
        for (size_t i = 0; i < cur_element_count && i < compressed_upper_neighbor_meta_.size(); ++i) {
            upper_meta_bytes += compressed_upper_neighbor_meta_[i].size() * sizeof(CompressedNeighborListMeta);
        }
        return compressed_level0_neighbor_blob_.size() +
               compressed_upper_neighbor_blob_.size() +
               compressed_level0_neighbor_meta_count_ * sizeof(CompressedNeighborListMeta) +
               compressed_deleted_marks_count_ * sizeof(unsigned char) +
               upper_meta_bytes;
    }

    ConstructionGraphMode effectiveConstructionGraphMode() const {
        return effective_construction_graph_mode_;
    }

    size_t estimatedRawConstructionAdjacencyBytes() const {
        return construction_estimated_raw_adjacency_bytes_;
    }

    size_t constructionGraphRamLimitBytes() const {
        return construction_ram_limit_bytes_;
    }

    size_t constructionCompressedBaseBytes() const {
        if (hybrid_level0_neighbors_node_local_) {
            return compressedLevel0MetaMemoryBytes() +
                   compressed_deleted_marks_count_ * sizeof(unsigned char) +
                   hybrid_delta_bytes_;
        }
        return compressedLevel0MetaMemoryBytes() +
               compressed_deleted_marks_count_ * sizeof(unsigned char) +
               compressed_level0_neighbor_meta_count_ * sizeof(CompressedNeighborListMeta) +
               compressed_level0_neighbor_blob_.size();
    }

    size_t constructionDeltaOverlayBytes() const {
        return hybrid_level0_neighbors_node_local_ ? 0 : hybrid_delta_bytes_;
    }

    size_t constructionAdjacencyBytes() const {
        if (hybridConstructionActive() || compressed_ids_enabled_) {
            size_t upper_raw_bytes = 0;
            if (!compressed_ids_enabled_) {
                for (size_t i = 0; i < cur_element_count; ++i) {
                    if (element_levels_[i] > 0) {
                        upper_raw_bytes += static_cast<size_t>(element_levels_[i]) * size_links_per_element_;
                    }
                }
            }
            return constructionCompressedBaseBytes() + constructionDeltaOverlayBytes() + upper_raw_bytes;
        }
        return rawNeighborIdBytesReserved();
    }

    size_t constructionMergeCount() const {
        return hybrid_level0_merge_count_;
    }

    double constructionMergeSeconds() const {
        return hybrid_level0_merge_seconds_;
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            size_t size = 0;
            tryGetRawNeighbors(curNodeNum, layer, size);
            forEachNeighbor(curNodeNum, layer, [&](tableint candidate_id) {
                if (visited_array[candidate_id] == visited_array_tag) return;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    prefetchDataByInternalId(candidateSet.top().second);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            });
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerSkippingV2(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
        std::unordered_map<labeltype, std::pair<dist_t, tableint>> best_vector_per_parent;
        std::unordered_map<labeltype, int> parent_id_freq;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
            parent_id_freq[getParentId(ep_id)]++;
            best_vector_per_parent[getParentId(ep_id)] = {dist, ep_id};
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && best_vector_per_parent.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            size_t size = 0;
            tryGetRawNeighbors(curNodeNum, layer, size);
            forEachNeighbor(curNodeNum, layer, [&](tableint candidate_id) {
                if (visited_array[candidate_id] == visited_array_tag) return;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (best_vector_per_parent.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    prefetchDataByInternalId(candidateSet.top().second);
#endif

                    if (!isMarkedDeleted(candidate_id)) {
                        top_candidates.emplace(dist1, candidate_id);
                        labeltype parent_id = getParentId(candidate_id);
                        auto it = best_vector_per_parent.find(parent_id);
                        parent_id_freq[parent_id]++;
                        if (it == best_vector_per_parent.end() || dist1 < it->second.first) {
                            best_vector_per_parent[parent_id] = {dist1, candidate_id};
                        }
                    }

                    while (best_vector_per_parent.size() > ef_construction_) {
                        tableint id = top_candidates.top().second;
                        labeltype removed_parent_id = getParentId(id);
                        top_candidates.pop();
                        parent_id_freq[removed_parent_id]--;
                        if (parent_id_freq[removed_parent_id] == 0) {
                            best_vector_per_parent.erase(removed_parent_id);
                        }
                    }

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            });
        }
        visited_list_pool_->releaseVisitedList(vl);

        // std::cout << "top_candidates size: " << top_candidates.size() << std::endl;
        
        // return final_results;

        // std::cout << "final_results size: " << final_results.size() << std::endl;
        // std::cout << "top_candidates size: " << top_candidates.size() << std::endl;

        return top_candidates;

        // return final_results;
    }


    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerSkipping(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
        // Track best vector per parent ID for more efficient management
        std::unordered_map<labeltype, std::pair<dist_t, tableint>> best_vector_per_parent;
        // Track frequency of parent IDs for efficient removal
        std::unordered_map<labeltype, int> parent_id_freq;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
            parent_id_freq[getParentId(ep_id)]++;
            best_vector_per_parent[getParentId(ep_id)] = {dist, ep_id};
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            size_t size = 0;
            tryGetRawNeighbors(curNodeNum, layer, size);
            forEachNeighbor(curNodeNum, layer, [&](tableint candidate_id) {
                if (visited_array[candidate_id] == visited_array_tag) return;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    prefetchDataByInternalId(candidateSet.top().second);
#endif

                    if (!isMarkedDeleted(candidate_id)){
                        top_candidates.emplace(dist1, candidate_id);
                        labeltype parent_id = getParentId(candidate_id);
                        auto it = best_vector_per_parent.find(parent_id);
                        if (it == best_vector_per_parent.end() || dist1 < it->second.first) {
                            best_vector_per_parent[parent_id] = {dist1, candidate_id};
                            parent_id_freq[parent_id]++;
                        }
                    }

                    if (top_candidates.size() > ef_construction_){
                        tableint id = top_candidates.top().second;
                        labeltype removed_parent_id = getParentId(id);
                        top_candidates.pop();
                        parent_id_freq[removed_parent_id]--;
                        if (parent_id_freq[removed_parent_id] == 0) {
                            best_vector_per_parent.erase(removed_parent_id);
                        }
                    }

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            });
        }
        visited_list_pool_->releaseVisitedList(vl);

        // Reconstruct the priority queue with the best vector per parent ID
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> final_results;
        
        for (const auto& [parent_id, best_vector] : best_vector_per_parent) {
            final_results.emplace(best_vector.first, best_vector.second);
        }
        
        // return final_results;

        // std::cout << "final_results size: " << final_results.size() << std::endl;
        // std::cout << "top_candidates size: " << top_candidates.size() << std::endl;

        return final_results;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerSTSkippingDuplicates(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        int threshold_multiplier = 1.00;
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        // Track best vector per parent ID for more efficient management
        std::unordered_map<labeltype, std::pair<dist_t, tableint>> best_vector_per_parent;
        
        // Track frequency of parent IDs for efficient removal
        std::unordered_map<labeltype, int> parent_id_freq;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            parent_id_freq[getParentId(ep_id)]++;
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
            // Add the parent ID of the entry point with its best vector
            best_vector_per_parent[getParentId(ep_id)] = {dist, ep_id};
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                // Add threshold-based early termination: stop if candidate_dist exceeds lowerBound by threshold_multiplier
                dist_t threshold = lowerBound * threshold_multiplier;
                flag_stop_search = candidate_dist > threshold && best_vector_per_parent.size() >= ef;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    // Stop based on unique parent IDs instead of total size
                    // Add threshold-based early termination here too
                    dist_t threshold = lowerBound * threshold_multiplier;
                    flag_stop_search = candidate_dist > threshold && best_vector_per_parent.size() >= ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            size_t size = 0;
            const tableint* raw_neighbors = tryGetRawNeighbors(current_node_id, 0, size);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

            auto process_candidate = [&](tableint candidate_id) {
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        // Consider candidate based on unique parent IDs and quality
                        labeltype candidate_parent_id = getParentId(candidate_id);
                        auto it = best_vector_per_parent.find(candidate_parent_id);
                        
                        if (it == best_vector_per_parent.end()) {
                            // New parent ID - always consider if we haven't reached the limit
                            flag_consider_candidate = best_vector_per_parent.size() < ef || lowerBound > dist;
                        } else {
                            // Existing parent ID - only consider if this vector is better
                            flag_consider_candidate = dist < it->second.first || lowerBound > dist;
                        }
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        prefetchDataByInternalId(candidate_set.top().second);
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            // Update best vector for this parent ID if needed
                            labeltype parent_id = getParentId(candidate_id);
                            auto it = best_vector_per_parent.find(parent_id);
                            parent_id_freq[parent_id]++;
                            if (it == best_vector_per_parent.end() || dist < it->second.first) {
                                best_vector_per_parent[parent_id] = {dist, candidate_id};
                            }
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            // Remove extra based on unique parent IDs instead of total size
                            flag_remove_extra = best_vector_per_parent.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            labeltype removed_parent_id = getParentId(id);
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                // Decrement frequency and remove if this was the last occurrence
                                parent_id_freq[removed_parent_id]--;
                                if (parent_id_freq[removed_parent_id] == 0) {
                                    best_vector_per_parent.erase(removed_parent_id);
                                }
                                
                                flag_remove_extra = best_vector_per_parent.size() > ef;
                            }
                        }

                        if (!top_candidates.empty()){
                            auto it_max = std::max_element(
                                best_vector_per_parent.begin(),
                                best_vector_per_parent.end(),
                                [](const std::pair<const labeltype, std::pair<dist_t, tableint>>& a,
                                   const std::pair<const labeltype, std::pair<dist_t, tableint>>& b) {
                                  return a.second.first < b.second.first;
                                }
                            );
                            lowerBound = it_max->second.first; 
                            // lowerBound = top_candidates.top().first;
                        }
                    }
                }
            };

            if (raw_neighbors != nullptr) {
#ifdef USE_SSE
                if (size > 0) {
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0]), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0] + 64), _MM_HINT_T0);
                    prefetchDataByInternalId(raw_neighbors[0]);
                }
#endif
                for (size_t j = 0; j < size; ++j) {
#ifdef USE_SSE
                    if (j + 1 < size) {
                        _mm_prefetch((char *) (visited_array + raw_neighbors[j + 1]), _MM_HINT_T0);
                        prefetchDataByInternalId(raw_neighbors[j + 1]);
                    }
#endif
                    process_candidate(raw_neighbors[j]);
                }
            } else {
                forEachNeighbor(current_node_id, 0, process_candidate);
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        
        // Reconstruct the priority queue with the best vector per parent ID
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> final_results;
        
        for (const auto& [parent_id, best_vector] : best_vector_per_parent) {
            final_results.emplace(best_vector.first, best_vector.second);
        }
        
        return final_results;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            size_t size = 0;
            const tableint* raw_neighbors = tryGetRawNeighbors(current_node_id, 0, size);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

            auto process_candidate = [&](tableint candidate_id) {
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        prefetchDataByInternalId(candidate_set.top().second);
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            };

            if (raw_neighbors != nullptr) {
#ifdef USE_SSE
                if (size > 0) {
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0]), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0] + 64), _MM_HINT_T0);
                    prefetchDataByInternalId(raw_neighbors[0]);
                }
#endif
                for (size_t j = 0; j < size; ++j) {
#ifdef USE_SSE
                    if (j + 1 < size) {
                        _mm_prefetch((char *) (visited_array + raw_neighbors[j + 1]), _MM_HINT_T0);
                        prefetchDataByInternalId(raw_neighbors[j + 1]);
                    }
#endif
                    process_candidate(raw_neighbors[j]);
                }
            } else {
                forEachNeighbor(current_node_id, 0, process_candidate);
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <int ex_bits, bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST_quantized(
        tableint ep_id,
        const PreprocessedQuantizedQuery& prepared_query,
        FastScanScratch& scratch,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = (!bare_bone_search && stop_condition) ? getDataByInternalId(ep_id) : nullptr;
            dist_t dist = computeExRaBitQApproxDistance<ex_bits>(prepared_query, ep_id, scratch);

            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            size_t size = 0;
            const tableint* raw_neighbors = tryGetRawNeighbors(current_node_id, 0, size);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

            auto process_candidate = [&](tableint candidate_id) {
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (!bare_bone_search && stop_condition) ? getDataByInternalId(candidate_id) : nullptr;
                    dist_t dist = computeExRaBitQApproxDistance<ex_bits>(
                        prepared_query, candidate_id, scratch
                    );

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                packed_short_codes_ +
                                (candidate_set.top().second / FAST_SIZE) * packedShortCodeBytesPerBlock()
                            ),
                            _MM_HINT_T0
                        );
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(
                                    getExternalLabel(id),
                                    getDataByInternalId(id),
                                    dist
                                );
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            };

            if (raw_neighbors != nullptr) {
#ifdef USE_SSE
                if (size > 0) {
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0]), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0] + 64), _MM_HINT_T0);
                    _mm_prefetch(
                        reinterpret_cast<const char*>(
                            packed_short_codes_ + (raw_neighbors[0] / FAST_SIZE) * packedShortCodeBytesPerBlock()
                        ),
                        _MM_HINT_T0
                    );
                }
#endif
                for (size_t j = 0; j < size; ++j) {
#ifdef USE_SSE
                    if (j + 1 < size) {
                        _mm_prefetch((char *) (visited_array + raw_neighbors[j + 1]), _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                packed_short_codes_ + (raw_neighbors[j + 1] / FAST_SIZE) * packedShortCodeBytesPerBlock()
                            ),
                            _MM_HINT_T0
                        );
                    }
#endif
                    process_candidate(raw_neighbors[j]);
                }
            } else {
                forEachNeighbor(current_node_id, 0, process_candidate);
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    template <int ex_bits, bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerSTSkippingDuplicates_quantized(
        tableint ep_id,
        const PreprocessedQuantizedQuery& prepared_query,
        FastScanScratch& scratch,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        int threshold_multiplier = 1.00;
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
        std::unordered_map<labeltype, std::pair<dist_t, tableint>> best_vector_per_parent;
        std::unordered_map<labeltype, int> parent_id_freq;

        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = (!bare_bone_search && stop_condition) ? getDataByInternalId(ep_id) : nullptr;
            dist_t dist = computeExRaBitQApproxDistance<ex_bits>(prepared_query, ep_id, scratch);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            parent_id_freq[getParentId(ep_id)]++;
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
            best_vector_per_parent[getParentId(ep_id)] = {dist, ep_id};
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                dist_t threshold = lowerBound * threshold_multiplier;
                flag_stop_search = candidate_dist > threshold && best_vector_per_parent.size() >= ef;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    dist_t threshold = lowerBound * threshold_multiplier;
                    flag_stop_search = candidate_dist > threshold && best_vector_per_parent.size() >= ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            size_t size = 0;
            const tableint* raw_neighbors = tryGetRawNeighbors(current_node_id, 0, size);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

            auto process_candidate = [&](tableint candidate_id) {
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (!bare_bone_search && stop_condition) ? getDataByInternalId(candidate_id) : nullptr;
                    dist_t dist = computeExRaBitQApproxDistance<ex_bits>(
                        prepared_query, candidate_id, scratch
                    );

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        labeltype candidate_parent_id = getParentId(candidate_id);
                        auto it = best_vector_per_parent.find(candidate_parent_id);

                        if (it == best_vector_per_parent.end()) {
                            flag_consider_candidate = best_vector_per_parent.size() < ef || lowerBound > dist;
                        } else {
                            flag_consider_candidate = dist < it->second.first || lowerBound > dist;
                        }
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                packed_short_codes_ +
                                (candidate_set.top().second / FAST_SIZE) * packedShortCodeBytesPerBlock()
                            ),
                            _MM_HINT_T0
                        );
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            labeltype parent_id = getParentId(candidate_id);
                            auto it = best_vector_per_parent.find(parent_id);
                            parent_id_freq[parent_id]++;
                            if (it == best_vector_per_parent.end() || dist < it->second.first) {
                                best_vector_per_parent[parent_id] = {dist, candidate_id};
                            }
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = best_vector_per_parent.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            labeltype removed_parent_id = getParentId(id);
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                parent_id_freq[removed_parent_id]--;
                                if (parent_id_freq[removed_parent_id] == 0) {
                                    best_vector_per_parent.erase(removed_parent_id);
                                }
                                flag_remove_extra = best_vector_per_parent.size() > ef;
                            }
                        }

                        if (!top_candidates.empty()) {
                            auto it_max = std::max_element(
                                best_vector_per_parent.begin(),
                                best_vector_per_parent.end(),
                                [](const std::pair<const labeltype, std::pair<dist_t, tableint>>& a,
                                   const std::pair<const labeltype, std::pair<dist_t, tableint>>& b) {
                                    return a.second.first < b.second.first;
                                }
                            );
                            lowerBound = it_max->second.first;
                        }
                    }
                }
            };

            if (raw_neighbors != nullptr) {
#ifdef USE_SSE
                if (size > 0) {
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0]), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + raw_neighbors[0] + 64), _MM_HINT_T0);
                    _mm_prefetch(
                        reinterpret_cast<const char*>(
                            packed_short_codes_ + (raw_neighbors[0] / FAST_SIZE) * packedShortCodeBytesPerBlock()
                        ),
                        _MM_HINT_T0
                    );
                }
#endif
                for (size_t j = 0; j < size; ++j) {
#ifdef USE_SSE
                    if (j + 1 < size) {
                        _mm_prefetch((char *) (visited_array + raw_neighbors[j + 1]), _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                packed_short_codes_ + (raw_neighbors[j + 1] / FAST_SIZE) * packedShortCodeBytesPerBlock()
                            ),
                            _MM_HINT_T0
                        );
                    }
#endif
                    process_candidate(raw_neighbors[j]);
                }
            } else {
                forEachNeighbor(current_node_id, 0, process_candidate);
            }
        }

        visited_list_pool_->releaseVisitedList(vl);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> final_results;
        for (const auto& [parent_id, best_vector] : best_vector_per_parent) {
            final_results.emplace(best_vector.first, best_vector.second);
        }

        return final_results;
    }

    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST_scalar(
        tableint ep_id,
        const float* scaled_query,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = computeScalarQuantizedDistance(scaled_query, ep_id);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            size_t size = 0;
            const tableint* raw_neighbors = tryGetRawNeighbors(current_node_id, 0, size);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

            auto process_candidate = [&](tableint candidate_id) {
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = getDataByInternalId(candidate_id);
                    dist_t dist = computeScalarQuantizedDistance(scaled_query, candidate_id);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            };

            if (raw_neighbors != nullptr) {
                for (size_t j = 0; j < size; ++j) {
                    process_candidate(raw_neighbors[j]);
                }
            } else {
                forEachNeighbor(current_node_id, 0, process_candidate);
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    // void getNeighborsByHeuristic2(
    //     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    //     const size_t M,
    //     const labeltype parent_id) {
    //     // Always enforce heuristic (including parent-ID uniqueness) even if fewer than M candidates
    //     std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    //     std::vector<std::pair<dist_t, tableint>> return_list;
    //     std::unordered_set<labeltype> used_parents;  // Track used parent IDs
        
    //     while (top_candidates.size() > 0) {
    //         queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
    //         top_candidates.pop();
    //     }

    //     while (queue_closest.size()) {
    //         if (return_list.size() >= M)
    //             break;
    //         std::pair<dist_t, tableint> curent_pair = queue_closest.top();
    //         dist_t dist_to_query = -curent_pair.first;
    //         queue_closest.pop();
    //         bool good = true;

    //         // Check parent ID diversity
    //         labeltype current_parent = getParentId(curent_pair.second);
    //         if (used_parents.find(current_parent) != used_parents.end() || current_parent == parent_id) {
    //             good = false;  // Skip if parent already used
    //         }

    //         // Check geometric diversity (only if parent diversity check passed)
    //         if (good) {
    //             for (std::pair<dist_t, tableint> second_pair : return_list) {
    //                 dist_t curdist =
    //                         fstdistfunc_(getDataByInternalId(second_pair.second),
    //                                         getDataByInternalId(curent_pair.second),
    //                                         dist_func_param_);
    //                 if (curdist < dist_to_query) {
    //                     good = false;
    //                     break;
    //                 }
    //             }
    //         }
            
    //         if (good) {
    //             return_list.push_back(curent_pair);
    //             used_parents.insert(current_parent);  // Mark parent as used
    //         }
    //     }

    //     // Additional safety check for parent ID duplicates
    //     for (std::pair<dist_t, tableint> curent_pair : return_list) {
    //         top_candidates.emplace(-curent_pair.first, curent_pair.second);
    //     }
    // }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        std::unordered_map<labeltype, std::pair<dist_t, tableint>> vector_per_parent;

        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (vector_per_parent.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
                vector_per_parent[getParentId(curent_pair.second)] = curent_pair;
            }
        }

        // Sort return_list by distance (ascending) and take only top M
        std::sort(return_list.begin(), return_list.end(),
                  [](const std::pair<dist_t, tableint> &a, const std::pair<dist_t, tableint> &b) {
                      return a.first < b.first;
                  });
        
        size_t count = 0;
        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            if (count >= M) break;
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
            count++;
        }
    }

    // void getNeighborsByHeuristic2(
    //     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    //     const size_t M) {
    //     if (top_candidates.size() < M) {
    //         return;
    //     }

    //     std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    //     std::vector<std::pair<dist_t, tableint>> return_list;
    //     while (top_candidates.size() > 0) {
    //         queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
    //         top_candidates.pop();
    //     }

    //     while (queue_closest.size()) {
    //         if (return_list.size() >= M)
    //             break;
    //         std::pair<dist_t, tableint> curent_pair = queue_closest.top();
    //         dist_t dist_to_query = -curent_pair.first;
    //         queue_closest.pop();
    //         bool good = true;

    //         for (std::pair<dist_t, tableint> second_pair : return_list) {
    //             dist_t curdist =
    //                     fstdistfunc_(getDataByInternalId(second_pair.second),
    //                                     getDataByInternalId(curent_pair.second),
    //                                     dist_func_param_);
    //             if (curdist < dist_to_query) {
    //                 good = false;
    //                 break;
    //             }
    //         }
    //         if (good) {
    //             return_list.push_back(curent_pair);
    //         }
    //     }

    //     for (std::pair<dist_t, tableint> curent_pair : return_list) {
    //         top_candidates.emplace(-curent_pair.first, curent_pair.second);
    //     }
    // }

    void getNeighborsByHeuristicMultiStage(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) return;

        std::unordered_map<labeltype, std::vector<std::pair<dist_t, tableint>>> grouped;
        while (!top_candidates.empty()) {
            auto cand = top_candidates.top(); top_candidates.pop();
            labeltype pid = getParentId(cand.second);
            grouped[pid].push_back(cand);
        }

        std::unordered_map<labeltype, std::vector<std::pair<dist_t, tableint>>> parent_reps;
        for (auto &kv : grouped) {
            auto &cands = kv.second;
            // Sort by distance ascending
            std::sort(cands.begin(), cands.end(),
                      [](const std::pair<dist_t, tableint> &a, const std::pair<dist_t, tableint> &b){ return a.first < b.first; });
    
            std::vector<std::pair<dist_t, tableint>> local_sel;
            for (auto cand : cands) {
                bool good = true;
                for (auto sel : local_sel) {
                    dist_t curdist = fstdistfunc_(getDataByInternalId(sel.second),
                                                  getDataByInternalId(cand.second),
                                                  dist_func_param_);
                    if (curdist < cand.first) {
                        good = false;
                        break;
                    }
                }
                if (good) local_sel.push_back(cand);
            }
            parent_reps[kv.first] = local_sel;  // representative vectors per parent
        }

        // --- Step 2: Global diversification across parents ---
        std::vector<std::pair<dist_t, tableint>> all_reps;
        for (auto &kv : parent_reps)
            all_reps.insert(all_reps.end(), kv.second.begin(), kv.second.end());
        
        // Sort globally by distance ascending
        std::sort(all_reps.begin(), all_reps.end(),
            [](const std::pair<dist_t, tableint> &a, const std::pair<dist_t, tableint> &b){ return a.first < b.first; });
        
        // Apply global occlusion-like diversification
        std::vector<std::pair<dist_t, tableint>> global_list;
        for (auto cand : all_reps) {
            bool good = true;
            for (auto sel : global_list) {
                dist_t curdist = fstdistfunc_(getDataByInternalId(sel.second),
                                            getDataByInternalId(cand.second),
                                            dist_func_param_);
                if (curdist < cand.first) {
                    good = false;
                    break;
                }
            }
            if (good) global_list.push_back(cand);
        }

        std::unordered_map<labeltype, std::vector<std::pair<dist_t, tableint>>> rr_groups;
        for (auto &cand : global_list) {
            labeltype pid = getParentId(cand.second);
            rr_groups[pid].push_back(cand);
        }

        std::vector<std::pair<dist_t, tableint>> final_list;
        size_t total_selected = 0;

        while (total_selected < M) {
            bool added = false;
            for (auto &kv : rr_groups) {
                auto &group = kv.second;
                if (group.empty()) continue;

                final_list.push_back(group.front());
                group.erase(group.begin());
                total_selected++;
                added = true;

                if (total_selected >= M) break;
            }
            if (!added) break; // all groups exhausted
        }

        // Push final_list back into the priority queue
        for (auto &cand : final_list)
            top_candidates.emplace(cand.first, cand.second);
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        if (useCompactLevel0Adjacency()) {
            throw std::runtime_error("Raw level0 linklist access is unavailable after ID compression");
        }
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        if (useCompactLevel0Adjacency()) {
            throw std::runtime_error("Raw level0 linklist access is unavailable after ID compression");
        }
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        if (levelHasCompressedAdjacency(internal_id, level)) {
            throw std::runtime_error("Raw upper-layer linklist access is unavailable after ID compression");
        }
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        // getNeighborsByHeuristic2(top_candidates, M_);
        getNeighborsByHeuristicMultiStage(top_candidates, M_);
        // getNeighborsByHeuristic2(top_candidates, M_);
        // if (top_candidates.size() > M_)
        //     throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(top_candidates.size());
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            std::vector<tableint> existing_neighbors = getConnectionsNoLock(cur_c, level);
            if (!existing_neighbors.empty() && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");
            }
            setConnectionsNoLock(cur_c, level, selectedNeighbors);
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            std::vector<tableint> other_neighbors = getConnectionsNoLock(selectedNeighbors[idx], level);
            size_t sz_link_list_other = other_neighbors.size();

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (other_neighbors[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    other_neighbors.push_back(cur_c);
                    setConnectionsNoLock(selectedNeighbors[idx], level, other_neighbors);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(other_neighbors[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), other_neighbors[j]);
                    }

                    // getNeighborsByHeuristic2(candidates, Mcurmax);
                    getNeighborsByHeuristicMultiStage(candidates, Mcurmax);
                    // getNeighborsByHeuristic2(candidates, Mcurmax);

                    other_neighbors.clear();
                    while (candidates.size() > 0) {
                        other_neighbors.push_back(candidates.top().second);
                        candidates.pop();
                    }
                    setConnectionsNoLock(selectedNeighbors[idx], level, other_neighbors);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        if (level == 0) {
            maybeMergeHybridLevel0Delta();
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");
        if (hasExternalRawDataView()) {
            throw std::runtime_error("Cannot resize an index while external build raw backing is active");
        }
        if (compressed_ids_enabled_ || compact_level0_storage_enabled_) {
            throw std::runtime_error("Cannot resize after compressed construction adjacency is enabled");
        }

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        if (raw_data_memory_ != nullptr) {
            char * raw_data_memory_new = (char *) realloc(raw_data_memory_, new_max_elements * data_size_);
            if (raw_data_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate raw data");
            raw_data_memory_ = raw_data_memory_new;
        }

        if (scalar_codes_ != nullptr) {
            int8_t* new_scalar_codes = (int8_t*)realloc(scalar_codes_, new_max_elements * vector_dim_ * sizeof(int8_t));
            if (new_scalar_codes == nullptr) {
                throw std::runtime_error("Not enough memory: resizeIndex failed to reallocate scalar codes");
            }
            scalar_codes_ = new_scalar_codes;
        }

        if (packed_short_codes_ != nullptr && short_code_length_ > 0) {
            uint8_t* new_packed_short_codes =
                (uint8_t*)realloc(
                    packed_short_codes_,
                    packedShortCodeBlockCount(new_max_elements) * packedShortCodeBytesPerBlock()
                );
            if (new_packed_short_codes == nullptr) {
                throw std::runtime_error("Not enough memory: resizeIndex failed to reallocate packed short codes");
            }
            packed_short_codes_ = new_packed_short_codes;
        }

        // Reallocate short and long codes
        uint8_t* new_short_codes = (uint8_t*)realloc(short_codes_, new_max_elements * short_code_length_ * sizeof(uint8_t));
        if (new_short_codes == nullptr) {
            throw std::runtime_error("Not enough memory: resizeIndex failed to reallocate short codes");
        }
        short_codes_ = new_short_codes;

        uint8_t* new_long_codes = (uint8_t*)realloc(long_codes_, new_max_elements * long_code_length_ * sizeof(uint8_t));
        if (new_long_codes == nullptr) {
            throw std::runtime_error("Not enough memory: resizeIndex failed to reallocate long codes");
        }
        long_codes_ = new_long_codes;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;
        for (size_t i = max_elements_; i < new_max_elements; ++i) {
            linkLists_[i] = nullptr;
        }

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        const bool serialize_hybrid_snapshot = hybridConstructionActive();
        const bool serialized_level0_compact = compact_level0_storage_enabled_ || compressed_ids_enabled_;
        const bool serialized_compressed_ids = compressed_ids_enabled_ || serialize_hybrid_snapshot;
        const SerializedAdjacencySnapshot hybrid_snapshot = serialize_hybrid_snapshot
            ? buildSerializedHybridAdjacencySnapshot(true)
            : SerializedAdjacencySnapshot{};
        const auto& serialized_level0_neighbor_blob = serialize_hybrid_snapshot
            ? hybrid_snapshot.level0_neighbor_blob
            : compressed_level0_neighbor_blob_;
        const auto& serialized_upper_neighbor_meta = serialize_hybrid_snapshot
            ? hybrid_snapshot.upper_neighbor_meta
            : compressed_upper_neighbor_meta_;
        const auto& serialized_upper_neighbor_blob = serialize_hybrid_snapshot
            ? hybrid_snapshot.upper_neighbor_blob
            : compressed_upper_neighbor_blob_;
        const size_t element_count = cur_element_count;
        const size_t level0_meta_count = serialize_hybrid_snapshot
            ? hybrid_snapshot.level0_neighbor_meta.size()
            : compressed_level0_neighbor_meta_count_;
        size_t size = 0;
        size += sizeof(uint32_t);  // magic
        size += sizeof(uint32_t);  // version
        size += sizeof(persist_raw_vectors_);
        size += sizeof(quantization_enabled_);
        size += sizeof(vector_dim_);
        size += sizeof(quantized_dim_);
        size += sizeof(ex_bits_);
        size += sizeof(quantization_method_);
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(parent_id_offset_);
        size += sizeof(offsetData_);
        size += sizeof(ex_factor_offset_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);
        size += sizeof(short_code_length_);
        size += sizeof(long_code_length_);
        size += sizeof(serialized_compressed_ids);

        if (serialized_level0_compact) {
            size += sizeof(size_t) + element_count * compactLevel0MetaBytes();
        } else {
            size += element_count * size_data_per_element_;
        }
        size += element_count * short_code_length_ * sizeof(uint8_t);
        size += element_count * long_code_length_ * sizeof(uint8_t);
        if (quantization_method_ == QuantizationMethod::ExRaBitQ && short_code_length_ > 0) {
            size += packedShortCodeBlockCount(element_count) * packedShortCodeBytesPerBlock();
        }
        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            size += vector_dim_ * sizeof(float);
            size += element_count * vector_dim_ * sizeof(int8_t);
        }
        if (persist_raw_vectors_ && raw_data_memory_ != nullptr) {
            size += element_count * data_size_;
        }
        if (quantization_method_ == QuantizationMethod::ExRaBitQ) {
            size += quantized_dim_ * quantized_dim_ * sizeof(float);
        }

        if (serialized_level0_compact) {
            size += sizeof(size_t) + element_count * sizeof(unsigned char);
            size += sizeof(size_t) + level0_meta_count * sizeof(CompressedNeighborListMeta);
            size += sizeof(size_t) + serialized_level0_neighbor_blob.size();
        }
        if (serialized_compressed_ids) {
            for (size_t i = 0; i < element_count; ++i) {
                size += sizeof(size_t);
                size += serialized_upper_neighbor_meta[i].size() * sizeof(CompressedNeighborListMeta);
            }
            size += sizeof(size_t) + serialized_upper_neighbor_blob.size();
        } else {
            for (size_t i = 0; i < element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                size += sizeof(linkListSize);
                size += linkListSize;
            }
        }

        size += sizeof(size_t);  // map_size
        for (const auto& pair : parent_id_to_labels_) {
            size += sizeof(labeltype);  // parent_id
            size += sizeof(size_t);  // vec_size
            size += pair.second.size() * sizeof(labeltype);
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        const bool serialize_hybrid_snapshot = hybridConstructionActive();
        const bool serialized_level0_compact = compact_level0_storage_enabled_ || compressed_ids_enabled_;
        const bool serialized_compressed_ids = compressed_ids_enabled_ || serialize_hybrid_snapshot;
        const SerializedAdjacencySnapshot hybrid_snapshot = serialize_hybrid_snapshot
            ? buildSerializedHybridAdjacencySnapshot(true)
            : SerializedAdjacencySnapshot{};
        const auto& serialized_level0_neighbor_blob = serialize_hybrid_snapshot
            ? hybrid_snapshot.level0_neighbor_blob
            : compressed_level0_neighbor_blob_;
        const auto& serialized_upper_neighbor_meta = serialize_hybrid_snapshot
            ? hybrid_snapshot.upper_neighbor_meta
            : compressed_upper_neighbor_meta_;
        const auto& serialized_upper_neighbor_blob = serialize_hybrid_snapshot
            ? hybrid_snapshot.upper_neighbor_blob
            : compressed_upper_neighbor_blob_;
        const size_t element_count = cur_element_count;
        const CompressedNeighborListMeta* serialized_level0_neighbor_meta_data = serialize_hybrid_snapshot
            ? hybrid_snapshot.level0_neighbor_meta.data()
            : compressed_level0_neighbor_meta_;
        const size_t serialized_level0_neighbor_meta_count = serialize_hybrid_snapshot
            ? hybrid_snapshot.level0_neighbor_meta.size()
            : compressed_level0_neighbor_meta_count_;
        std::ofstream output(location, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + location);
        }
        writeBinaryPOD(output, SERIALIZATION_MAGIC);
        writeBinaryPOD(output, SERIALIZATION_VERSION);
        writeBinaryPOD(output, persist_raw_vectors_);
        writeBinaryPOD(output, quantization_enabled_);
        writeBinaryPOD(output, vector_dim_);
        writeBinaryPOD(output, quantized_dim_);
        writeBinaryPOD(output, ex_bits_);
        writeBinaryPOD(output, quantization_method_);

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, parent_id_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, ex_factor_offset_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);
        
        // Save code lengths for loading
        writeBinaryPOD(output, short_code_length_);
        writeBinaryPOD(output, long_code_length_);
        writeBinaryPOD(output, serialized_compressed_ids);

        if (serialized_level0_compact) {
            size_t compact_meta_bytes = element_count * compactLevel0MetaBytes();
            writeBinaryPOD(output, compact_meta_bytes);
            if (compact_meta_bytes > 0) {
                output.write(compressed_level0_meta_memory_, compact_meta_bytes);
            }
        } else {
            if (data_level0_memory_ == nullptr) {
                throw std::runtime_error("Cannot save raw level-0 storage because it is not resident in memory");
            }
            output.write(data_level0_memory_, element_count * size_data_per_element_);
        }
        if (short_code_length_ > 0) {
            output.write(reinterpret_cast<char*>(short_codes_), element_count * short_code_length_ * sizeof(uint8_t));
        }
        if (long_code_length_ > 0) {
            output.write(reinterpret_cast<char*>(long_codes_), element_count * long_code_length_ * sizeof(uint8_t));
        }
        if (quantization_method_ == QuantizationMethod::ExRaBitQ && short_code_length_ > 0) {
            if (packed_short_codes_ == nullptr) {
                rebuildPackedShortCodeBlocks();
            }
            output.write(
                reinterpret_cast<char*>(packed_short_codes_),
                packedShortCodeBlockCount(element_count) * packedShortCodeBytesPerBlock()
            );
        }
        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            output.write(reinterpret_cast<const char*>(scalar_quantizer_.scales().data()), vector_dim_ * sizeof(float));
            output.write(reinterpret_cast<char*>(scalar_codes_), element_count * vector_dim_ * sizeof(int8_t));
        }
        if (persist_raw_vectors_ && raw_data_memory_ != nullptr) {
            output.write(raw_data_memory_, element_count * data_size_);
        }
        if (quantization_method_ == QuantizationMethod::ExRaBitQ) {
            rotator_.save(output);
        }
        if (serialized_level0_compact) {
            size_t deleted_size = element_count * sizeof(unsigned char);
            writeBinaryPOD(output, deleted_size);
            if (deleted_size > 0) {
                output.write(reinterpret_cast<const char*>(compressed_deleted_marks_), deleted_size);
            }

            size_t level0_meta_count = serialized_level0_neighbor_meta_count;
            writeBinaryPOD(output, level0_meta_count);
            if (level0_meta_count > 0) {
                output.write(
                    reinterpret_cast<const char*>(serialized_level0_neighbor_meta_data),
                    level0_meta_count * sizeof(CompressedNeighborListMeta)
                );
            }

            size_t level0_blob_size = serialized_level0_neighbor_blob.size();
            writeBinaryPOD(output, level0_blob_size);
            if (level0_blob_size > 0) {
                output.write(reinterpret_cast<const char*>(serialized_level0_neighbor_blob.data()), level0_blob_size);
            }
        }

        if (serialized_compressed_ids) {
            for (size_t i = 0; i < element_count; ++i) {
                size_t upper_meta_count = serialized_upper_neighbor_meta[i].size();
                writeBinaryPOD(output, upper_meta_count);
                if (upper_meta_count > 0) {
                    output.write(
                        reinterpret_cast<const char*>(serialized_upper_neighbor_meta[i].data()),
                        upper_meta_count * sizeof(CompressedNeighborListMeta)
                    );
                }
            }

            size_t upper_blob_size = serialized_upper_neighbor_blob.size();
            writeBinaryPOD(output, upper_blob_size);
            if (upper_blob_size > 0) {
                output.write(reinterpret_cast<const char*>(serialized_upper_neighbor_blob.data()), upper_blob_size);
            }
        } else {
            for (size_t i = 0; i < element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
        }

        size_t map_size = parent_id_to_labels_.size();
        writeBinaryPOD(output, map_size);
        for (const auto& pair : parent_id_to_labels_) {
            writeBinaryPOD(output, pair.first);
            size_t vec_size = pair.second.size();
            writeBinaryPOD(output, vec_size);
            output.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(labeltype));
        }

        if (!output.good()) {
            throw std::runtime_error("Failed while writing index file: " + location);
        }
        output.close();
        if (!output) {
            throw std::runtime_error("Failed to finalize index file: " + location);
        }
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        auto require_input_ok = [&](const char* context) {
            if (!input) {
                throw std::runtime_error(
                    std::string("Index file is truncated or corrupted while reading ") +
                    context + ": " + location
                );
            }
        };

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        uint32_t magic = 0;
        uint32_t version = 0;
        readBinaryPOD(input, magic);
        readBinaryPOD(input, version);
        if (magic != SERIALIZATION_MAGIC || (version != 3U && version != SERIALIZATION_VERSION)) {
            throw std::runtime_error("Index format is unsupported");
        }
        require_input_ok("file header");

        readBinaryPOD(input, persist_raw_vectors_);
        readBinaryPOD(input, quantization_enabled_);
        readBinaryPOD(input, vector_dim_);
        readBinaryPOD(input, quantized_dim_);
        readBinaryPOD(input, ex_bits_);
        readBinaryPOD(input, quantization_method_);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, parent_id_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, ex_factor_offset_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);
        readBinaryPOD(input, short_code_length_);
        readBinaryPOD(input, long_code_length_);
        compressed_ids_enabled_ = false;
        if (version >= 4U) {
            readBinaryPOD(input, compressed_ids_enabled_);
        }
        compact_level0_storage_enabled_ = compressed_ids_enabled_;
        require_input_ok("index metadata");

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if (quantization_method_ == QuantizationMethod::ExRaBitQ && ex_bits_ > 0) {
            configureQuantizedDistance();
        }

        if (compressed_ids_enabled_) {
            size_t compact_meta_bytes = 0;
            readBinaryPOD(input, compact_meta_bytes);
            require_input_ok("compressed level-0 metadata size");
            freeCompressedLevel0Storage();
            compressed_level0_meta_memory_ = static_cast<char*>(malloc(compact_meta_bytes));
            if (compact_meta_bytes > 0 && compressed_level0_meta_memory_ == nullptr) {
                throw std::runtime_error("Not enough memory for compressed level-0 metadata");
            }
            compressed_level0_meta_memory_bytes_ = compact_meta_bytes;
            if (compact_meta_bytes > 0) {
                input.read(compressed_level0_meta_memory_, compact_meta_bytes);
                require_input_ok("compressed level-0 metadata");
            }
            data_level0_memory_ = nullptr;
        } else {
            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);
            require_input_ok("raw level-0 storage");
        }

        if (short_code_length_ > 0) {
            short_codes_ = (uint8_t*)malloc(max_elements * short_code_length_ * sizeof(uint8_t));
            if (short_codes_ == nullptr) {
                throw std::runtime_error("Not enough memory for short codes");
            }
            input.read(reinterpret_cast<char*>(short_codes_), cur_element_count * short_code_length_ * sizeof(uint8_t));
            require_input_ok("short codes");
        } else {
            short_codes_ = nullptr;
        }

        if (long_code_length_ > 0) {
            long_codes_ = (uint8_t*)malloc(max_elements * long_code_length_ * sizeof(uint8_t));
            if (long_codes_ == nullptr) {
                throw std::runtime_error("Not enough memory for long codes");
            }
            input.read(reinterpret_cast<char*>(long_codes_), cur_element_count * long_code_length_ * sizeof(uint8_t));
            require_input_ok("long codes");
        } else {
            long_codes_ = nullptr;
        }

        if (quantization_method_ == QuantizationMethod::ExRaBitQ && short_code_length_ > 0) {
            allocatePackedShortCodeStorage();
            input.read(
                reinterpret_cast<char*>(packed_short_codes_),
                packedShortCodeBlockCount(cur_element_count) * packedShortCodeBytesPerBlock()
            );
            require_input_ok("packed short codes");
        } else {
            packed_short_codes_ = nullptr;
        }

        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            std::vector<float> scales(vector_dim_);
            input.read(reinterpret_cast<char*>(scales.data()), vector_dim_ * sizeof(float));
            require_input_ok("scalar scales");
            scalar_quantizer_ = ScalarInt8Quantizer(vector_dim_);
            scalar_quantizer_.set_scales(scales);
            scalar_codes_ = (int8_t*)malloc(max_elements * vector_dim_ * sizeof(int8_t));
            if (scalar_codes_ == nullptr) {
                throw std::runtime_error("Not enough memory for scalar int8 codes");
            }
            input.read(reinterpret_cast<char*>(scalar_codes_), cur_element_count * vector_dim_ * sizeof(int8_t));
            require_input_ok("scalar codes");
        } else {
            scalar_codes_ = nullptr;
        }

        if (persist_raw_vectors_) {
            raw_data_memory_ = (char *) malloc(max_elements * data_size_);
            if (raw_data_memory_ == nullptr) {
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate raw data");
            }
            input.read(raw_data_memory_, cur_element_count * data_size_);
            require_input_ok("raw vectors");
            raw_data_compacted_ = false;
        } else {
            raw_data_memory_ = nullptr;
            raw_data_compacted_ = true;
        }
        if (quantization_method_ == QuantizationMethod::ExRaBitQ) {
            quantizer_ = DataQuantizer(vector_dim_, ex_bits_);
            rotator_.resize(quantized_dim_);
            rotator_.load(input);
            require_input_ok("rotation matrix");
        }

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        std::memset(linkLists_, 0, sizeof(void *) * max_elements);
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        if (compressed_ids_enabled_) {
            size_t deleted_size = 0;
            readBinaryPOD(input, deleted_size);
            require_input_ok("deleted marks size");
            compressed_deleted_marks_ = static_cast<unsigned char*>(calloc(deleted_size, sizeof(unsigned char)));
            if (deleted_size > 0 && compressed_deleted_marks_ == nullptr) {
                throw std::runtime_error("Not enough memory for deleted marks");
            }
            compressed_deleted_marks_count_ = deleted_size;
            if (deleted_size > 0) {
                input.read(reinterpret_cast<char*>(compressed_deleted_marks_), deleted_size);
                require_input_ok("deleted marks");
            }

            size_t level0_meta_count = 0;
            readBinaryPOD(input, level0_meta_count);
            require_input_ok("level-0 neighbor metadata size");
            compressed_level0_neighbor_meta_ = static_cast<CompressedNeighborListMeta*>(
                calloc(level0_meta_count, sizeof(CompressedNeighborListMeta))
            );
            if (level0_meta_count > 0 && compressed_level0_neighbor_meta_ == nullptr) {
                throw std::runtime_error("Not enough memory for level-0 neighbor metadata");
            }
            compressed_level0_neighbor_meta_count_ = level0_meta_count;
            if (level0_meta_count > 0) {
                input.read(
                    reinterpret_cast<char*>(compressed_level0_neighbor_meta_),
                    level0_meta_count * sizeof(CompressedNeighborListMeta)
                );
                require_input_ok("level-0 neighbor metadata");
            }

            size_t level0_blob_size = 0;
            readBinaryPOD(input, level0_blob_size);
            require_input_ok("level-0 neighbor blob size");
            compressed_level0_neighbor_blob_.resize(level0_blob_size);
            if (level0_blob_size > 0) {
                input.read(reinterpret_cast<char*>(compressed_level0_neighbor_blob_.data()), level0_blob_size);
                require_input_ok("level-0 neighbor blob");
            }

            compressed_upper_neighbor_meta_.assign(cur_element_count, {});
            for (size_t i = 0; i < cur_element_count; ++i) {
                label_lookup_[getExternalLabel(i)] = i;
                size_t upper_meta_count = 0;
                readBinaryPOD(input, upper_meta_count);
                require_input_ok("upper-level neighbor metadata size");
                compressed_upper_neighbor_meta_[i].resize(upper_meta_count);
                if (upper_meta_count > 0) {
                    input.read(
                        reinterpret_cast<char*>(compressed_upper_neighbor_meta_[i].data()),
                        upper_meta_count * sizeof(CompressedNeighborListMeta)
                    );
                    require_input_ok("upper-level neighbor metadata");
                }
                element_levels_[i] = static_cast<int>(upper_meta_count);
                linkLists_[i] = nullptr;
            }

            size_t upper_blob_size = 0;
            readBinaryPOD(input, upper_blob_size);
            require_input_ok("upper-level neighbor blob size");
            compressed_upper_neighbor_blob_.resize(upper_blob_size);
            if (upper_blob_size > 0) {
                input.read(reinterpret_cast<char*>(compressed_upper_neighbor_blob_.data()), upper_blob_size);
                require_input_ok("upper-level neighbor blob");
            }
        } else {
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                require_input_ok("raw upper-level adjacency size");
                if (linkListSize == 0) {
                    element_levels_[i] = 0;
                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                    require_input_ok("raw upper-level adjacency");
                }
            }
        }

        parent_id_to_labels_.clear();
        size_t map_size;
        readBinaryPOD(input, map_size);
        require_input_ok("parent label map size");
        for (size_t i = 0; i < map_size; ++i) {
            labeltype parent_id;
            readBinaryPOD(input, parent_id);
            size_t vec_size;
            readBinaryPOD(input, vec_size);
            require_input_ok("parent label map entry");
            std::vector<labeltype> children(vec_size);
            input.read(reinterpret_cast<char*>(children.data()), vec_size * sizeof(labeltype));
            require_input_ok("parent label map labels");
            parent_id_to_labels_[parent_id] = std::move(children);
        }

        if (input.tellg() != total_filesize) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        requireRawData("getDataByLabel");
        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    std::vector<labeltype> getSamplesByParentId(labeltype parent_id) const {
        std::unique_lock<std::mutex> lock(parent_id_lookup_lock);
        auto it = parent_id_to_labels_.find(parent_id);
        if (it != parent_id_to_labels_.end()) {
            return it->second;
        }
        return {};
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        if (hybridConstructionActive()) {
            throw std::runtime_error("Hybrid construction mode does not support deletes");
        }
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        
        labeltype parent_id = getParentId(internalId);
        {
            std::unique_lock<std::mutex> lock(parent_id_lookup_lock);
            auto it = parent_id_to_labels_.find(parent_id);
            if (it != parent_id_to_labels_.end()) {
                auto& children = it->second;
                auto child_it = std::find(children.begin(), children.end(), label);
                if (child_it != children.end()) {
                    children.erase(child_it);
                }
                if (children.empty()) {
                    parent_id_to_labels_.erase(it);
                }
            }
        }
        
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            if (hasCompressedDeletedMarks()) {
                compressed_deleted_marks_[internalId] |= DELETE_MARK;
            } else {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur |= DELETE_MARK;
            }
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        if (hybridConstructionActive()) {
            throw std::runtime_error("Hybrid construction mode does not support undeletion");
        }
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        
        labeltype parent_id = getParentId(internalId);
        {
            std::unique_lock<std::mutex> lock(parent_id_lookup_lock);
            parent_id_to_labels_[parent_id].push_back(label);
        }

        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            if (hasCompressedDeletedMarks()) {
                compressed_deleted_marks_[internalId] &= ~DELETE_MARK;
            } else {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur &= ~DELETE_MARK;
            }
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        if (hasCompressedDeletedMarks()) {
            return internalId < compressed_deleted_marks_count_ &&
                   (compressed_deleted_marks_[internalId] & DELETE_MARK);
        }
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }

    template <typename OutputIt>
    void copyNeighborsNoLock(tableint internal_id, int level, OutputIt out) const {
        forEachNeighbor(internal_id, level, [&](tableint neighbor) {
            *out++ = neighbor;
        });
    }

    std::vector<tableint> getConnectionsNoLock(tableint internalId, int level) const {
        size_t size = 0;
        tryGetRawNeighbors(internalId, level, size);
        std::vector<tableint> result;
        result.reserve(size);
        copyNeighborsNoLock(internalId, level, std::back_inserter(result));
        return result;
    }

    void setConnectionsNoLock(tableint internalId, int level, const std::vector<tableint>& neighbors) {
        if (level == 0 && hybridConstructionActive()) {
            replaceHybridLevel0Neighbors(internalId, neighbors);
            return;
        }

        linklistsizeint* ll_cur = get_linklist_at_level(internalId, level);
        setListCount(ll_cur, static_cast<unsigned short>(neighbors.size()));
        tableint* data = reinterpret_cast<tableint*>(ll_cur + 1);
        if (!neighbors.empty()) {
            std::memcpy(data, neighbors.data(), neighbors.size() * sizeof(tableint));
        }
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(
        const void *data_point,
        labeltype label,
        labeltype parent_id,
        bool replace_deleted = false
    ) override {
        if (compressed_ids_enabled_) {
            throw std::runtime_error("Cannot add points after adjacency IDs have been compressed");
        }
        if (hybridConstructionActive() && replace_deleted) {
            throw std::runtime_error("Hybrid construction mode does not support replace_deleted");
        }
        requireRawData("addPoint");
        if (replace_deleted && hasExternalRawDataView()) {
            throw std::runtime_error("Replacement is not supported while external build raw backing is active");
        }
        if (!quantization_enabled_) {
            throw std::runtime_error("Raw addPoint requires a configured quantization model");
        }

        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            std::vector<int8_t> scalar_code(vector_dim_);
            scalar_quantizer_.quantizeVector(static_cast<const float*>(data_point), scalar_code.data());
            addPointScalar(data_point, scalar_code.data(), label, parent_id, replace_deleted);
            return;
        }

        std::vector<PID> ids = {0};
        std::vector<uint8_t> short_data(quantizer_.block_bytes() * quantizer_.num_blocks(1));
        std::vector<uint8_t> short_code(short_code_length_);
        std::vector<uint8_t> long_code(long_code_length_);
        ExFactor ex_factor[1];

        quantizer_.quantize(
            static_cast<const float*>(data_point),
            ids,
            rotator_,
            short_data.data(),
            long_code.data(),
            short_code.data(),
            ex_factor
        );
        addPoint(
            data_point,
            short_code.data(),
            long_code.data(),
            ex_factor,
            label,
            parent_id,
            replace_deleted
        );
    }

    void addPoint(const void *data_point, uint8_t* short_code, uint8_t* long_code, ExFactor* ex_factor, labeltype label, labeltype parent_id, bool replace_deleted = false) {
        if (compressed_ids_enabled_) {
            throw std::runtime_error("Cannot add points after adjacency IDs have been compressed");
        }
        if (hybridConstructionActive() && replace_deleted) {
            throw std::runtime_error("Hybrid construction mode does not support replace_deleted");
        }
        if (quantization_method_ != QuantizationMethod::ExRaBitQ) {
            throw std::runtime_error("ExRaBitQ addPoint overload is only valid in ExRaBitQ mode");
        }
        requireRawData("addPoint");
        if (replace_deleted && hasExternalRawDataView()) {
            throw std::runtime_error("Replacement is not supported while external build raw backing is active");
        }
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, short_code, long_code, ex_factor, label, parent_id, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, short_code, long_code, ex_factor, label, parent_id, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);

            {
                std::unique_lock<std::mutex> lock(parent_id_lookup_lock);
                parent_id_to_labels_[parent_id].push_back(label);
            }

            setExternalLabel(internal_id_replaced, label);
            setParentId(internal_id_replaced, parent_id);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            setShortCode(internal_id_replaced, short_code);
            setLongCode(internal_id_replaced, long_code);
            setExFactor(internal_id_replaced, ex_factor);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void addPointScalar(
        const void *data_point,
        int8_t* scalar_code,
        labeltype label,
        labeltype parent_id,
        bool replace_deleted = false
    ) {
        if (compressed_ids_enabled_) {
            throw std::runtime_error("Cannot add points after adjacency IDs have been compressed");
        }
        if (hybridConstructionActive() && replace_deleted) {
            throw std::runtime_error("Hybrid construction mode does not support replace_deleted");
        }
        requireRawData("addPoint");
        if (replace_deleted && hasExternalRawDataView()) {
            throw std::runtime_error("Replacement is not supported while external build raw backing is active");
        }
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPointScalar(data_point, scalar_code, label, parent_id, -1);
            return;
        }

        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        if (!is_vacant_place) {
            addPointScalar(data_point, scalar_code, label, parent_id, -1);
        } else {
            labeltype label_replaced = getExternalLabel(internal_id_replaced);

            {
                std::unique_lock<std::mutex> lock(parent_id_lookup_lock);
                parent_id_to_labels_[parent_id].push_back(label);
            }

            setExternalLabel(internal_id_replaced, label);
            setParentId(internal_id_replaced, parent_id);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            setScalarCode(internal_id_replaced, scalar_code);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        if (compressed_ids_enabled_ || hybridConstructionActive()) {
            throw std::runtime_error("Hybrid/compressed construction mode does not support updatePoint");
        }
        requireOwnedRawData("updatePoint");
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                // getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);
                getNeighborsByHeuristicMultiStage(candidates, layer == 0 ? maxM0_ : maxM_);
                // getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    size_t candSize = candidates.size();
                    std::vector<tableint> new_neighbors;
                    new_neighbors.reserve(candSize);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        new_neighbors.push_back(candidates.top().second);
                        candidates.pop();
                    }
                    setConnectionsNoLock(neigh, layer, new_neighbors);
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    forEachNeighbor(currObj, level, [&](tableint cand) {
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    });
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayerSkippingV2(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        return getConnectionsNoLock(internalId, level);
    }


    tableint addPoint(const void *data_point, uint8_t* short_code, uint8_t* long_code, ExFactor* ex_factor, labeltype label, labeltype parent_id, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                if (hybridConstructionActive()) {
                    throw std::runtime_error("Hybrid construction mode does not support duplicate-label overwrite/update");
                }
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                if (hasExternalRawDataView()) {
                    throw std::runtime_error("Updating an existing label is not supported while external build raw backing is active");
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                setShortCode(existingInternalId, short_code);
                setLongCode(existingInternalId, long_code);
                setExFactor(existingInternalId, ex_factor);
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            linkLists_[cur_c] = nullptr;
            if (hasExternalRawDataView()) {
                assignExternalRawRow(cur_c, data_point);
            }
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        {
            std::unique_lock <std::mutex> lock(parent_id_lookup_lock);
            parent_id_to_labels_[parent_id].push_back(label);
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        if (data_level0_memory_ != nullptr) {
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        }

        // Initialisation of the data and label
        setExternalLabel(cur_c, label);
        setParentId(cur_c, parent_id);
        if (raw_data_memory_ != nullptr) {
            memcpy(raw_data_memory_ + cur_c * data_size_, data_point, data_size_);
        }
        setShortCode(cur_c, short_code);
        setLongCode(cur_c, long_code);
        setExFactor(cur_c, ex_factor);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand >= max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    tableint addPointScalar(const void *data_point, int8_t* scalar_code, labeltype label, labeltype parent_id, int level) {
        tableint cur_c = 0;
        {
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                if (hybridConstructionActive()) {
                    throw std::runtime_error("Hybrid construction mode does not support duplicate-label overwrite/update");
                }
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                if (hasExternalRawDataView()) {
                    throw std::runtime_error("Updating an existing label is not supported while external build raw backing is active");
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                setScalarCode(existingInternalId, scalar_code);
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            linkLists_[cur_c] = nullptr;
            if (hasExternalRawDataView()) {
                assignExternalRawRow(cur_c, data_point);
            }
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        {
            std::unique_lock <std::mutex> lock(parent_id_lookup_lock);
            parent_id_to_labels_[parent_id].push_back(label);
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        if (data_level0_memory_ != nullptr) {
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        }

        setExternalLabel(cur_c, label);
        setParentId(cur_c, parent_id);
        if (raw_data_memory_ != nullptr) {
            memcpy(raw_data_memory_ + cur_c * data_size_, data_point, data_size_);
        }
        setScalarCode(cur_c, scalar_code);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand >= max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnnSkippingDuplicates(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        requireRawData("searchKnnSkippingDuplicates");
        std::priority_queue<std::tuple<dist_t, labeltype, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t size = 0;
                const tableint* raw_neighbors = tryGetRawNeighbors(currObj, level, size);
                metric_hops++;
                metric_distance_computations+=size;
                forEachNeighbor(currObj, level, [&](tableint cand) {
                    if (cand >= max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                });
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerSTSkippingDuplicates<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerSTSkippingDuplicates<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        // // Save top candidates to CSV file
        // std::ofstream csv_file("top_candidates_skippingnodes.csv");
        // csv_file << "Distance,External_ID,Parent_ID\n"; // CSV header

        // auto temp_queue = top_candidates; // Create a copy to avoid destroying the original
        // while (!temp_queue.empty()) {
        //     auto candidate = temp_queue.top();
        //     csv_file << candidate.first << "," 
        //              << getExternalLabel(candidate.second) << "," 
        //              << getParentId(candidate.second) << "\n";
        //     temp_queue.pop();
        // }
        // csv_file.close();

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::make_tuple(rez.first, getExternalLabel(rez.second), getParentId(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    template <int ex_bits>
    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnnSkippingDuplicates_quantized_exbits(
        const PreprocessedQuantizedQuery& prepared_query,
        size_t k,
        BaseFilterFunctor* isIdAllowed
    ) const {
        std::priority_queue<std::tuple<dist_t, labeltype, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        if (getPackedShortCodesBase() == nullptr || !raw_data_compacted_) {
            rebuildPackedShortCodeBlocks();
        }

        static thread_local FastScanScratch fastscan_scratch;
        const size_t packed_block_count = packedShortCodeBlockCount(cur_element_count);
        fastscan_scratch.prepare(packed_block_count);

        dist_t curdist = computeExRaBitQApproxDistance<ex_bits>(
            prepared_query, enterpoint_node_, fastscan_scratch
        );

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t size = 0;
                tryGetRawNeighbors(currObj, level, size);
                metric_hops++;
                metric_distance_computations += size;
                forEachNeighbor(currObj, level, [&](tableint cand) {
                    if (cand >= max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = computeExRaBitQApproxDistance<ex_bits>(
                        prepared_query, cand, fastscan_scratch
                    );

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                });
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerSTSkippingDuplicates_quantized<ex_bits, true>(
                currObj, prepared_query, fastscan_scratch, std::max(ef_, k), isIdAllowed
            );
        } else {
            top_candidates = searchBaseLayerSTSkippingDuplicates_quantized<ex_bits, false>(
                currObj, prepared_query, fastscan_scratch, std::max(ef_, k), isIdAllowed
            );
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::make_tuple(rez.first, getExternalLabel(rez.second), getParentId(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnnSkippingDuplicates_quantized(
        const PreprocessedQuantizedQuery& prepared_query,
        size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr
    ) const {
        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            throw std::runtime_error("searchKnnSkippingDuplicates_quantized is only implemented for ExRaBitQ mode");
        }

        switch (ex_bits_) {
            case 8:
                return searchKnnSkippingDuplicates_quantized_exbits<8>(prepared_query, k, isIdAllowed);
            case 7:
                return searchKnnSkippingDuplicates_quantized_exbits<7>(prepared_query, k, isIdAllowed);
            case 6:
                return searchKnnSkippingDuplicates_quantized_exbits<6>(prepared_query, k, isIdAllowed);
            case 4:
                return searchKnnSkippingDuplicates_quantized_exbits<4>(prepared_query, k, isIdAllowed);
            case 3:
                return searchKnnSkippingDuplicates_quantized_exbits<3>(prepared_query, k, isIdAllowed);
            case 2:
                return searchKnnSkippingDuplicates_quantized_exbits<2>(prepared_query, k, isIdAllowed);
            default:
                throw std::runtime_error("Unsupported ExRaBitQ bit width");
        }
    }

    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnnSkippingDuplicates_quantized(
        const void *query_data,
        size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr
    ) const {
        PreprocessedQuantizedQuery prepared_query = preprocessQueryQuantization(query_data);
        return searchKnnSkippingDuplicates_quantized(prepared_query, k, isIdAllowed);
    }

    inline void pack_high_acc_LUT(int16_t* quant_query, size_t D, uint8_t* HC_LUT, int& shift) const {
        size_t M = D >> 2;
        constexpr int pos[16] = {
            3 /*0000*/,
            3 /*0001*/,
            2 /*0010*/,
            3 /*0011*/,
            1 /*0100*/,
            3 /*0101*/,
            2 /*0110*/,
            3 /*0111*/,
            0 /*1000*/,
            3 /*1001*/,
            2 /*1010*/,
            3 /*1011*/,
            1 /*1100*/,
            3 /*1101*/,
            2 /*1110*/,
            3 /*1111*/,
        };
    
        int16_t* quan_query = quant_query;
        for (size_t i = 0; i < M; i++) {
            int PORTABLE_ALIGN64 LUT[16];
            int v_min = 0;
    
            LUT[0] = 0;
            for (int j = 1; j < 16; j++) {
                LUT[j] = LUT[j - lowbit(j)] + quan_query[pos[j]];
                v_min = (LUT[j] < v_min) ? LUT[j] : v_min;
            }
    
            // avx2 - 256, avx512 - 512
            constexpr size_t B_regi = 512;
            constexpr size_t B_lane = 128;
            constexpr size_t B_byte = 8;
    
            constexpr size_t n_lut_per_iter = B_regi / B_lane;
            constexpr size_t n_code_per_iter = 2 * B_regi / B_byte;
            constexpr size_t n_code_per_lane = B_lane / B_byte;
    
            uint8_t* fill_lo = HC_LUT + i / n_lut_per_iter * n_code_per_iter +
                               (i % n_lut_per_iter) * n_code_per_lane;
            uint8_t* fill_hi = fill_lo + B_regi / B_byte;
    
            /* shift all the elements in LUT such that they become unsigned integer */
            __m512i lut = _mm512_load_epi32(LUT);
            __m512i tmp = _mm512_sub_epi32(lut, _mm512_set1_epi32(v_min));
            __m128i lo = _mm512_cvtepi32_epi8(tmp);
            __m128i hi = _mm512_cvtepi32_epi8(_mm512_srli_epi32(tmp, 8));
            _mm_store_si128((__m128i*)fill_lo, lo);
            _mm_store_si128((__m128i*)fill_hi, hi);
    
            // the shifted valued will be finally added back
            shift += v_min;
            quan_query += 4;
        }
    }

    template <int ex_bits>
    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnn_quantized_exbits(
        const PreprocessedQuantizedQuery& prepared_query,
        size_t k,
        BaseFilterFunctor* isIdAllowed
    ) const {
        std::priority_queue<std::tuple<dist_t, labeltype, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        if (getPackedShortCodesBase() == nullptr || !raw_data_compacted_) {
            rebuildPackedShortCodeBlocks();
        }

        static thread_local FastScanScratch fastscan_scratch;
        const size_t packed_block_count = packedShortCodeBlockCount(cur_element_count);
        fastscan_scratch.prepare(packed_block_count);

        dist_t curdist = computeExRaBitQApproxDistance<ex_bits>(
            prepared_query, enterpoint_node_, fastscan_scratch
        );

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t size = 0;
                tryGetRawNeighbors(currObj, level, size);
                metric_hops++;
                metric_distance_computations += size;
                forEachNeighbor(currObj, level, [&](tableint cand) {
                    if (cand >= max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = computeExRaBitQApproxDistance<ex_bits>(
                        prepared_query, cand, fastscan_scratch
                    );

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                });
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST_quantized<ex_bits, true>(
                currObj, prepared_query, fastscan_scratch, std::max(ef_, k), isIdAllowed
            );
        } else {
            top_candidates = searchBaseLayerST_quantized<ex_bits, false>(
                currObj, prepared_query, fastscan_scratch, std::max(ef_, k), isIdAllowed
            );
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::make_tuple(rez.first, getExternalLabel(rez.second), getParentId(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnn_quantized(
        const PreprocessedQuantizedQuery& prepared_query,
        size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr
    ) const {
        std::priority_queue<std::tuple<dist_t, labeltype, labeltype>> result;
        if (cur_element_count == 0) return result;

        if (quantization_method_ == QuantizationMethod::ScalarInt8) {
            const float* scaled_query = prepared_query.scaled_query.data();
            tableint currObj = enterpoint_node_;
            dist_t curdist = computeScalarQuantizedDistance(scaled_query, enterpoint_node_);

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    size_t size = 0;
                    const tableint* raw_neighbors = tryGetRawNeighbors(currObj, level, size);
                    metric_hops++;
                    metric_distance_computations += size;
                    forEachNeighbor(currObj, level, [&](tableint cand) {
                        if (cand >= max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = computeScalarQuantizedDistance(scaled_query, cand);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    });
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            bool bare_bone_search = !num_deleted_ && !isIdAllowed;
            if (bare_bone_search) {
                top_candidates = searchBaseLayerST_scalar<true>(
                        currObj, scaled_query, std::max(ef_, k), isIdAllowed);
            } else {
                top_candidates = searchBaseLayerST_scalar<false>(
                        currObj, scaled_query, std::max(ef_, k), isIdAllowed);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::make_tuple(rez.first, getExternalLabel(rez.second), getParentId(rez.second)));
                top_candidates.pop();
            }
            return result;
        }

        switch (ex_bits_) {
            case 8:
                return searchKnn_quantized_exbits<8>(prepared_query, k, isIdAllowed);
            case 7:
                return searchKnn_quantized_exbits<7>(prepared_query, k, isIdAllowed);
            case 6:
                return searchKnn_quantized_exbits<6>(prepared_query, k, isIdAllowed);
            case 4:
                return searchKnn_quantized_exbits<4>(prepared_query, k, isIdAllowed);
            case 3:
                return searchKnn_quantized_exbits<3>(prepared_query, k, isIdAllowed);
            case 2:
                return searchKnn_quantized_exbits<2>(prepared_query, k, isIdAllowed);
            default:
                throw std::runtime_error("Unsupported ExRaBitQ bit width");
        }
    }

    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnn_quantized(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        PreprocessedQuantizedQuery prepared_query = preprocessQueryQuantization(query_data);
        return searchKnn_quantized(prepared_query, k, isIdAllowed);
    }


    std::priority_queue<std::tuple<dist_t, labeltype, labeltype>>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        requireRawData("searchKnn");
        std::priority_queue<std::tuple<dist_t, labeltype, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t size = 0;
                const tableint* raw_neighbors = tryGetRawNeighbors(currObj, level, size);
                metric_hops++;
                metric_distance_computations+=size;
                forEachNeighbor(currObj, level, [&](tableint cand) {
                    if (cand >= max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                });
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::make_tuple(rez.first, getExternalLabel(rez.second), getParentId(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        requireRawData("searchStopConditionClosest");
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t size = 0;
                const tableint* raw_neighbors = tryGetRawNeighbors(currObj, level, size);
                metric_hops++;
                metric_distance_computations+=size;
                forEachNeighbor(currObj, level, [&](tableint cand) {
                    if (cand >= max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                });
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                std::unordered_set<tableint> s;
                int size = 0;
                forEachNeighbor(i, l, [&](tableint candidate) {
                    assert(candidate < cur_element_count);
                    assert(candidate != static_cast<tableint>(i));
                    inbound_connections_num[candidate]++;
                    s.insert(candidate);
                    connections_checked++;
                    ++size;
                });
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
