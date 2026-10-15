#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;

namespace sycl_flow {

struct WorkConfig {
    size_t num_workers;
    size_t work_group_size;
    size_t capacity_per_path;
};

struct SoAQueueDeviceView {
    int* r;
    int* c;
    float* flux;
    float* z_delta;
    float* parent_z_deltas;     // 9 floats per slot: index = (s * num_workers + w) * 9 + (dr*3 + dc)
    uint8_t* is_start;          // uint8_t instead of bool for predictable alignment
    uint8_t* parent_is_start;
    int* parent_indices;        // 8 ints per slot: index = (s * num_workers + w) * 8 + p
    int* num_parents;
    float* min_distance;
    float* minDistXYZ;
    int* isForest;
    int* forest_int_count;
    float* node_infra;

    size_t num_workers;
    size_t capacity_per_path;

    inline size_t idx(size_t slot, int worker_id) const {
        return slot * num_workers + static_cast<size_t>(worker_id);
    }

    inline size_t parent_zd_idx(size_t slot, int worker_id, int r, int c) const {
        return (slot * num_workers + static_cast<size_t>(worker_id)) * 9 + static_cast<size_t>(r * 3 + c);
    }

    inline size_t parent_idx(size_t slot, int worker_id, int p) const {
        return (slot * num_workers + static_cast<size_t>(worker_id)) * 8 + static_cast<size_t>(p);
    }
};

inline constexpr size_t align_128(size_t bytes) {
    return (bytes + 127) & ~static_cast<size_t>(127);
}

inline constexpr size_t soa_bytes_per_node(bool infraBool) {
    return (sizeof(int) * 2          // r, c
          + sizeof(float) * 2        // flux, z_delta
          + sizeof(float) * 9        // parent_z_deltas
          + sizeof(uint8_t) * 2      // is_start, parent_is_start
          + sizeof(int) * 8          // parent_indices
          + sizeof(int)              // num_parents
          + sizeof(float) * 2        // min_distance, minDistXYZ
          + sizeof(int) * 2)         // isForest, forest_int_count
          + (infraBool ? sizeof(float) : 0); // node_infra
}

inline void atomic_max_float(float* addr, float val) {
    uint32_t* u_addr = reinterpret_cast<uint32_t*>(addr);
    uint32_t old_u = *u_addr;
    float old_f = sycl::bit_cast<float>(old_u);
    while (val > old_f || old_f == -9999.0f) {
        uint32_t new_u = sycl::bit_cast<uint32_t>(val);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> ref(*u_addr);
        if (ref.compare_exchange_strong(old_u, new_u)) {
            break;
        }
        old_f = sycl::bit_cast<float>(old_u);
    }
}

inline void atomic_min_float(float* addr, float val) {
    uint32_t* u_addr = reinterpret_cast<uint32_t*>(addr);
    uint32_t old_u = *u_addr;
    float old_f = sycl::bit_cast<float>(old_u);
    while (val < old_f || old_f == 999999.0f) {
        uint32_t new_u = sycl::bit_cast<uint32_t>(val);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> ref(*u_addr);
        if (ref.compare_exchange_strong(old_u, new_u)) {
            break;
        }
        old_f = sycl::bit_cast<float>(old_u);
    }
}

inline WorkConfig determine_optimal_configuration(sycl::queue& q, size_t num_release_cells, size_t total_cells, bool infraBool) {
    auto dev = q.get_device();
    bool is_cpu = dev.is_cpu();
    size_t compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    size_t global_mem = dev.get_info<sycl::info::device::global_mem_size>();
    std::string dev_name = dev.get_info<sycl::info::device::name>();

    size_t num_workers = 1024;
    size_t work_group_size = 64;
    size_t capacity_per_path = 32768;
    size_t bytes_per_node = soa_bytes_per_node(infraBool);

    if (is_cpu) {
        // For CPU devices, workers = CPU hardware threads/cores (e.g. 12 to 32)
        num_workers = std::max<size_t>(1, compute_units);
        num_workers = std::min<size_t>(num_workers, num_release_cells);
        work_group_size = 1;
        capacity_per_path = std::min<size_t>(total_cells, 32768);
    } else {
        // Discrete GPU:
        // Use work_group_size = 64 (2 warps) to allow full SM saturation (48 blocks)
        // while preserving ample VRAM for deep path queues (~29k nodes)
        work_group_size = 64;
        size_t target_blocks = compute_units; // e.g. 48 blocks for 48 SMs
        if (num_release_cells < work_group_size) {
            target_blocks = 1;
        }
        size_t target_workers = target_blocks * work_group_size;

        // Check for manual user override via ACPP_NUM_WORKERS
        const char* env_workers = std::getenv("ACPP_NUM_WORKERS");
        if (env_workers != nullptr) {
            target_workers = std::stoul(env_workers);
            target_workers = ((target_workers + work_group_size - 1) / work_group_size) * work_group_size;
            std::cout << "[SYCL Dynamic Config] User override via ACPP_NUM_WORKERS: " << target_workers << " workers." << std::endl;
        }

        // Memory Budget: 80% on >= 10 GB GPUs, 50% on < 10 GB GPUs
        double vram_fraction = (global_mem >= 10ULL * 1024 * 1024 * 1024) ? 0.80 : 0.50;
        size_t vram_budget = static_cast<size_t>(global_mem * vram_fraction);

        // Dynamically compute capacity per worker that strictly fits inside the VRAM budget
        size_t raw_capacity = vram_budget / (target_workers * bytes_per_node);

        // If raw_capacity is smaller than 20,480 (safe floor to prevent path clipping in large avalanche bowls),
        // scale down target_workers in multiples of work_group_size
        if (raw_capacity < 20480 && vram_budget >= 20480 * bytes_per_node * work_group_size) {
            size_t min_capacity = 20480;
            size_t max_allowed_workers = vram_budget / (min_capacity * bytes_per_node);
            target_workers = (max_allowed_workers >= work_group_size) ? (max_allowed_workers / work_group_size) * work_group_size : work_group_size;
            raw_capacity = vram_budget / (target_workers * bytes_per_node);
        }

        // Clamp capacity: cannot exceed total_cells, and cap at 32,768
        capacity_per_path = std::min<size_t>(raw_capacity, total_cells);
        capacity_per_path = std::min<size_t>(capacity_per_path, 32768);
        if (capacity_per_path >= 128) {
            capacity_per_path = (capacity_per_path / 128) * 128;
        }

        num_workers = target_workers;
    }

    double vram_mb = (num_workers * capacity_per_path * bytes_per_node) / (1024.0 * 1024.0);
    double total_vram_gb = global_mem / (1024.0 * 1024.0 * 1024.0);
    size_t grid_blocks = (num_workers + work_group_size - 1) / work_group_size;

    std::cout << "[SYCL Dynamic Config] Target: " << dev_name 
              << " (" << compute_units << " " << (is_cpu ? "Cores" : "Compute Units / SMs")
              << ", " << total_vram_gb << " GB RAM)\n"
              << "                      -> Auto-tuned Grid: " << grid_blocks << " Blocks (" 
              << num_workers << " Persistent Workers, WG Size: " << work_group_size << ")\n"
              << "                      -> Queue Capacity: " << capacity_per_path << " nodes/worker "
              << "(Work Pool VRAM: " << vram_mb << " MB)" << std::endl;

    return WorkConfig{num_workers, work_group_size, capacity_per_path};
}

py::tuple run_sycl_calculation(
    py::array_t<float> dem,
    py::array_t<float> release,
    py::object infra,                      // none or np.ndarray
    float alpha,
    float exp,
    float flux_threshold,
    float max_z_delta,
    float nodata,
    float cellsize,
    bool infraBool,
    bool forestBool,
    py::dict varParams,
    bool fluxDistOldVersionBool,
    bool previewMode,
    py::object forestArray,                // none or np.ndarray
    py::object forestParams,               // none or py::dict
    std::string device_type
) {
    // extract info from python arrays
    py::buffer_info dem_info = dem.request();
    py::buffer_info rel_info = release.request();
    
    int rows = dem_info.shape[0];
    int cols = dem_info.shape[1];
    int total_cells = rows * cols;
    
    float* dem_ptr = static_cast<float*>(dem_info.ptr);
    float* rel_ptr = static_cast<float*>(rel_info.ptr);

    // unpack infrastructure and forest pointers
    float* infra_ptr = nullptr;
    if (infraBool && !infra.is_none()) {
        auto infra_arr = infra.cast<py::array_t<float>>();
        infra_ptr = static_cast<float*>(infra_arr.request().ptr);
    }
    float* forest_ptr = nullptr;
    if (forestBool && !forestArray.is_none()) {
        auto forest_arr = forestArray.cast<py::array_t<float>>();
        forest_ptr = static_cast<float*>(forest_arr.request().ptr);
    }

    // unpack varParams
    bool varUmaxBool = varParams["varUmaxBool"].cast<bool>();
    float* varUmax_ptr = nullptr;
    if (varUmaxBool && !varParams["varUmaxArray"].is_none()) {
        auto umax_arr = varParams["varUmaxArray"].cast<py::array_t<float>>();
        varUmax_ptr = static_cast<float*>(umax_arr.request().ptr);
    }
    bool varAlphaBool = varParams["varAlphaBool"].cast<bool>();
    float* varAlpha_ptr = nullptr;
    if (varAlphaBool && !varParams["varAlphaArray"].is_none()) {
        auto valpha_arr = varParams["varAlphaArray"].cast<py::array_t<float>>();
        varAlpha_ptr = static_cast<float*>(valpha_arr.request().ptr);
    }
    bool varExponentBool = varParams["varExponentBool"].cast<bool>();
    float* varExponent_ptr = nullptr;
    if (varExponentBool && !varParams["varExponentArray"].is_none()) {
        auto vexponent_arr = varParams["varExponentArray"].cast<py::array_t<float>>();
        varExponent_ptr = static_cast<float*>(vexponent_arr.request().ptr);
    }

    // unpack forestParams
    bool forestInteraction = false;
    std::string forestModule = "";
    float maxAddedFrictionFor = 0.0f;
    float minAddedFrictionFor = 0.0f;
    float velThForFriction = 0.0f;
    float maxDetrainmentFor = 0.0f;
    float minDetrainmentFor = 0.0f;
    float velThForDetrain = 0.0f;
    std::string forestFrictionLayerType = "";
    float skipForestDist = 0.0f;
    if (forestBool && !forestParams.is_none()) {
        py::dict fp = forestParams.cast<py::dict>();

        forestInteraction = fp["forestInteraction"].cast<bool>();
        forestModule      = fp["forestModule"].cast<std::string>();

        maxAddedFrictionFor = fp.contains("maxAddedFriction") ? fp["maxAddedFriction"].cast<float>() : 0.0f;
        minAddedFrictionFor = fp.contains("minAddedFriction") ? fp["minAddedFriction"].cast<float>() : 0.0f;
        maxDetrainmentFor   = fp.contains("maxDetrainment")   ? fp["maxDetrainment"].cast<float>()   : 0.0f;
        minDetrainmentFor   = fp.contains("minDetrainment")   ? fp["minDetrainment"].cast<float>()   : 0.0f;
        velThForFriction    = fp.contains("velThForFriction") ? fp["velThForFriction"].cast<float>() : 0.0f;
        velThForDetrain     = fp.contains("velThForDetrain")  ? fp["velThForDetrain"].cast<float>()  : 0.0f;
        skipForestDist      = fp.contains("skipForestDist")   ? fp["skipForestDist"].cast<float>()   : 0.0f;
        if (fp.contains("fFrLayerType")) {
            forestFrictionLayerType = fp["fFrLayerType"].cast<std::string>();
        }
    }

    int forest_module_enum = 0; // 0: None, 1: forestFriction, 2: forestDetrainment, 3: forestFrictionLayer
    if (forestModule == "forestFriction") forest_module_enum = 1;
    else if (forestModule == "forestDetrainment") forest_module_enum = 2;
    else if (forestModule == "forestFrictionLayer") forest_module_enum = 3;

    int friction_layer_type_enum = 0; // 0: absolute, 1: relative
    if (forestFrictionLayerType == "relative") friction_layer_type_enum = 1;

    constexpr float SQRT2_val = 1.41421356237f;
    constexpr float SQRT2xG = SQRT2_val * 9.81f;
    float noFrictionEffectZDelta = (velThForFriction * velThForFriction) / SQRT2xG;
    float noDetrainmentEffectZdelta = (velThForDetrain * velThForDetrain) / SQRT2xG;

    bool forestDetrainmentBool = false;
    if (forestBool && forest_module_enum == 2) {
        if (maxDetrainmentFor != 0.0f || minDetrainmentFor != 0.0f || velThForDetrain != 0.0f) {
            forestDetrainmentBool = true;
        }
    }

    // find release indices
    std::vector<int> release_flat_indices;
    for (int i = 0; i < total_cells; i++) {
        if (rel_ptr[i] > 0.0f) {
            release_flat_indices.push_back(i);
        }
    }
    int num_release_cells = release_flat_indices.size();
    std::sort(release_flat_indices.begin(), release_flat_indices.end(), [&](int a, int b) {
        return dem_ptr[a] > dem_ptr[b];
    });
    std::cout << "SYCL_CORE DEBUG: total_cells=" << total_cells << ", num_release_cells=" << num_release_cells << std::endl;

    // select execution device
    sycl::device dev;
    if (device_type == "gpu") {
        try {
            dev = sycl::device(sycl::gpu_selector_v);
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not found. Falling back to default selector." << std::endl;
            dev = sycl::device(sycl::default_selector_v);
        }
    } else if (device_type == "cpu") {
        try {
            dev = sycl::device(sycl::cpu_selector_v);
        } catch (const sycl::exception& e) {
            std::cerr << "CPU not found. Falling back to default selector." << std::endl;
            dev = sycl::device(sycl::default_selector_v);
        }
    } else {
        dev = sycl::device(sycl::default_selector_v);
    }
    
    // Asynchronous exception handler to convert GPU kernel errors to Python exceptions
    auto async_handler = [](sycl::exception_list e_list) {
        for (auto& e : e_list) {
            try {
                std::rethrow_exception(e);
            } catch (const sycl::exception& se) {
                throw std::runtime_error(std::string("SYCL GPU Kernel Failure: ") + se.what());
            }
        }
    };
    
    sycl::queue q(dev, async_handler, sycl::property_list{sycl::property::queue::in_order{}});
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    // Host result vectors
    std::vector<float> host_z_delta(total_cells, 0.0f);
    std::vector<float> host_flux(total_cells, -9999.0f);
    std::vector<int> host_counts(total_cells, 0);
    std::vector<float> host_backcalc(total_cells, -9999.0f);
    std::vector<float> host_forest_int(total_cells, 999999.0f);
    
    // Dummy buffers to prevent crash on nullptr
    std::vector<float> dummy(total_cells, 0.0f);
    
    float* var_alpha_data = varAlpha_ptr ? varAlpha_ptr : dummy.data();
    float* var_exponent_data = varExponent_ptr ? varExponent_ptr : dummy.data();
    float* var_umax_data = varUmax_ptr ? varUmax_ptr : dummy.data();
    float* infra_data = infra_ptr ? infra_ptr : dummy.data();
    float* forest_data = forest_ptr ? forest_ptr : dummy.data();

    if (num_release_cells > 0) {
        int* rel_indices = sycl::malloc_device<int>(num_release_cells, q);
        float* dem = sycl::malloc_device<float>(total_cells, q);
        float* z_delta = sycl::malloc_device<float>(total_cells, q);
        float* flux = sycl::malloc_device<float>(total_cells, q);
        int* counts = sycl::malloc_device<int>(total_cells, q);

        float* var_alpha = sycl::malloc_device<float>(total_cells, q);
        float* var_exponent = sycl::malloc_device<float>(total_cells, q);
        float* var_umax = sycl::malloc_device<float>(total_cells, q);
        float* infra_map = sycl::malloc_device<float>(total_cells, q);
        float* forest_map = sycl::malloc_device<float>(total_cells, q);
        float* backcalc = sycl::malloc_device<float>(total_cells, q);
        float* forest_int = sycl::malloc_device<float>(total_cells, q);

        // Dynamically determine optimal configuration based on hardware and terrain
        WorkConfig config = determine_optimal_configuration(q, static_cast<size_t>(num_release_cells), total_cells, infraBool);
        size_t num_workers = config.num_workers;
        size_t work_group_size = config.work_group_size;
        size_t capacity_per_path = config.capacity_per_path;

        size_t total_elements = num_workers * capacity_per_path;

        // Calculate sizes
        size_t size_r = total_elements * sizeof(int);
        size_t size_c = total_elements * sizeof(int);
        size_t size_flux = total_elements * sizeof(float);
        size_t size_z_delta = total_elements * sizeof(float);
        size_t size_parent_zd = total_elements * 9 * sizeof(float);
        size_t size_is_start = total_elements * sizeof(uint8_t);
        size_t size_parent_is_start = total_elements * sizeof(uint8_t);
        size_t size_parent_indices = total_elements * 8 * sizeof(int);
        size_t size_num_parents = total_elements * sizeof(int);
        size_t size_min_dist = total_elements * sizeof(float);
        size_t size_min_dist_xyz = total_elements * sizeof(float);
        size_t size_is_forest = total_elements * sizeof(int);
        size_t size_forest_int = total_elements * sizeof(int);
        size_t size_infra = infraBool ? (total_elements * sizeof(float)) : 0;

        size_t offset = 0;
        size_t offset_r = offset; offset += align_128(size_r);
        size_t offset_c = offset; offset += align_128(size_c);
        size_t offset_flux = offset; offset += align_128(size_flux);
        size_t offset_z_delta = offset; offset += align_128(size_z_delta);
        size_t offset_parent_zd = offset; offset += align_128(size_parent_zd);
        size_t offset_is_start = offset; offset += align_128(size_is_start);
        size_t offset_parent_is_start = offset; offset += align_128(size_parent_is_start);
        size_t offset_parent_indices = offset; offset += align_128(size_parent_indices);
        size_t offset_num_parents = offset; offset += align_128(size_num_parents);
        size_t offset_min_dist = offset; offset += align_128(size_min_dist);
        size_t offset_min_dist_xyz = offset; offset += align_128(size_min_dist_xyz);
        size_t offset_is_forest = offset; offset += align_128(size_is_forest);
        size_t offset_forest_int = offset; offset += align_128(size_forest_int);
        size_t offset_infra = offset;
        if (infraBool) {
            offset += align_128(size_infra);
        }
        size_t total_work_pool_bytes = offset;

        uint8_t* d_work_pool = sycl::malloc_device<uint8_t>(total_work_pool_bytes, q);

        SoAQueueDeviceView q_view;
        q_view.r = reinterpret_cast<int*>(d_work_pool + offset_r);
        q_view.c = reinterpret_cast<int*>(d_work_pool + offset_c);
        q_view.flux = reinterpret_cast<float*>(d_work_pool + offset_flux);
        q_view.z_delta = reinterpret_cast<float*>(d_work_pool + offset_z_delta);
        q_view.parent_z_deltas = reinterpret_cast<float*>(d_work_pool + offset_parent_zd);
        q_view.is_start = reinterpret_cast<uint8_t*>(d_work_pool + offset_is_start);
        q_view.parent_is_start = reinterpret_cast<uint8_t*>(d_work_pool + offset_parent_is_start);
        q_view.parent_indices = reinterpret_cast<int*>(d_work_pool + offset_parent_indices);
        q_view.num_parents = reinterpret_cast<int*>(d_work_pool + offset_num_parents);
        q_view.min_distance = reinterpret_cast<float*>(d_work_pool + offset_min_dist);
        q_view.minDistXYZ = reinterpret_cast<float*>(d_work_pool + offset_min_dist_xyz);
        q_view.isForest = reinterpret_cast<int*>(d_work_pool + offset_is_forest);
        q_view.forest_int_count = reinterpret_cast<int*>(d_work_pool + offset_forest_int);
        q_view.node_infra = infraBool ? reinterpret_cast<float*>(d_work_pool + offset_infra) : nullptr;
        q_view.num_workers = num_workers;
        q_view.capacity_per_path = capacity_per_path;

        // Global Atomic Task Counter for Dynamic Work-Stealing
        int* d_task_counter = sycl::malloc_device<int>(1, q);
        int init_counter = 0;
        q.memcpy(d_task_counter, &init_counter, sizeof(int));

        // Diagnostic: Track maximum queue depth reached across all paths
        int* d_max_queue_depth = sycl::malloc_device<int>(1, q);
        int init_max_depth = 0;
        q.memcpy(d_max_queue_depth, &init_max_depth, sizeof(int));

        // Host to Device copies
        q.memcpy(rel_indices, release_flat_indices.data(), num_release_cells * sizeof(int));
        q.memcpy(dem, dem_ptr, total_cells * sizeof(float));
        q.memcpy(z_delta, host_z_delta.data(), total_cells * sizeof(float));
        q.memcpy(flux, host_flux.data(), total_cells * sizeof(float));
        q.memcpy(counts, host_counts.data(), total_cells * sizeof(int));
        q.memcpy(var_alpha, var_alpha_data, total_cells * sizeof(float));
        q.memcpy(var_exponent, var_exponent_data, total_cells * sizeof(float));
        q.memcpy(var_umax, var_umax_data, total_cells * sizeof(float));
        q.memcpy(infra_map, infra_data, total_cells * sizeof(float));
        q.memcpy(forest_map, forest_data, total_cells * sizeof(float));
        q.memcpy(backcalc, host_backcalc.data(), total_cells * sizeof(float));
        q.memcpy(forest_int, host_forest_int.data(), total_cells * sizeof(float));

        // Launch single persistent kernel over num_workers with dynamic work-group size
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_workers), sycl::range<1>(work_group_size)), [=](sycl::nd_item<1> item) {
            int worker_id = static_cast<int>(item.get_global_id(0));
            const SoAQueueDeviceView queue_soa = q_view;
            const auto& q = queue_soa;

            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> 
                task_counter(*d_task_counter);
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> 
                max_depth_tracker(*d_max_queue_depth);

            auto is_nodata = [=](float val) {
                return sycl::isnan(val) || sycl::fabs(val - nodata) < 1e-3f;
            };

            auto is_neighborhood_valid = [=](int r, int c) {
                if (r - 1 < 0 || r + 1 >= rows || c - 1 < 0 || c + 1 >= cols) {
                    return false;
                }
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        float val = dem[(r + dr) * cols + (c + dc)];
                        if (is_nodata(val)) {
                            return false;
                        }
                    }
                }
                return true;
            };

            sycl::sub_group sg = item.get_sub_group();
            bool has_work = true;
            int q_idx = 0;
            int queue_size = 0;

            float thread_max_z_delta = max_z_delta;
            float thread_alpha = alpha;
            int thread_exp = static_cast<int>(exp);
            float tanAlpha = sycl::tan(thread_alpha * 3.141592653589793f / 180.0f);

            while (true) {
                // If path is complete, write results and claim next release cell
                if (has_work && (q_idx == queue_size)) {
                    if (queue_size > 0) {
                        max_depth_tracker.fetch_max(queue_size);

                        // Write final outputs (z_delta, flux, counts)
                        for (int i = 0; i < queue_size; i++) {
                            size_t idx_i = q.idx(i, worker_id);
                            int qi_r = q.r[idx_i];
                            int qi_c = q.c[idx_i];
                            int flat = qi_r * cols + qi_c;
                            
                            atomic_max_float(&z_delta[flat], q.z_delta[idx_i]);
                            atomic_max_float(&flux[flat], q.flux[idx_i]);
                            
                            // Count as visit if this is the first occurrence of the coordinate in the path queue
                            bool first_occurrence = true;
                            for (int j = 0; j < i; j++) {
                                size_t idx_j = q.idx(j, worker_id);
                                if (q.r[idx_j] == qi_r && q.c[idx_j] == qi_c) {
                                    first_occurrence = false;
                                    break;
                                }
                            }
                            if (first_occurrence) {
                                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> 
                                     atomic_counts(counts[flat]);
                                atomic_counts.fetch_add(1);
                            }
                        }

                        // Backtracking for infrastructure
                        if (infraBool && q.node_infra != nullptr) {
                            for (int i = 0; i < queue_size; i++) {
                                size_t idx_i = q.idx(i, worker_id);
                                float infra_val = infra_map[q.r[idx_i] * cols + q.c[idx_i]];
                                q.node_infra[idx_i] = sycl::max(0.0f, infra_val);
                            }
                            for (int i = queue_size - 1; i >= 0; i--) {
                                size_t idx_i = q.idx(i, worker_id);
                                float val = q.node_infra[idx_i];
                                int np = q.num_parents[idx_i];
                                for (int p = 0; p < np; p++) {
                                    int p_idx = q.parent_indices[q.parent_idx(i, worker_id, p)];
                                    size_t parent_mem_idx = q.idx(p_idx, worker_id);
                                    if (val > q.node_infra[parent_mem_idx]) {
                                        q.node_infra[parent_mem_idx] = val;
                                    }
                                }
                            }
                            for (int i = 0; i < queue_size; i++) {
                                size_t idx_i = q.idx(i, worker_id);
                                int flat_idx = q.r[idx_i] * cols + q.c[idx_i];
                                atomic_max_float(&backcalc[flat_idx], q.node_infra[idx_i]);
                            }
                        }
                        
                        // Forest interaction count update
                        if (forestBool && forestInteraction) {
                            for (int i = 0; i < queue_size; i++) {
                                size_t idx_i = q.idx(i, worker_id);
                                int flat_idx = q.r[idx_i] * cols + q.c[idx_i];
                                atomic_min_float(&forest_int[flat_idx], static_cast<float>(q.forest_int_count[idx_i]));
                            }
                        }

                        queue_size = 0;
                        q_idx = 0;
                    }

                    int my_task_id = task_counter.fetch_add(1);
                    if (my_task_id < num_release_cells) {
                        int start_flat_index = rel_indices[my_task_id];
                        int start_row = start_flat_index / cols;
                        int start_col = start_flat_index % cols;

                        if (is_neighborhood_valid(start_row, start_col)) {
                            queue_size = 0;
                            q_idx = 0;

                            thread_max_z_delta = max_z_delta;
                            thread_alpha = alpha;
                            thread_exp = static_cast<int>(exp);

                            if (varUmaxBool && var_umax[start_flat_index] > 0.0f && var_umax[start_flat_index] <= 8848.0f) {
                                thread_max_z_delta = var_umax[start_flat_index];
                            }
                            if (varAlphaBool && var_alpha[start_flat_index] > 0.0f && var_alpha[start_flat_index] <= 90.0f) {
                                thread_alpha = var_alpha[start_flat_index];
                            }
                            if (varExponentBool && var_exponent[start_flat_index] > 0.0f) {
                                thread_exp = static_cast<int>(var_exponent[start_flat_index]);
                            }
                            
                            tanAlpha = sycl::tan(thread_alpha * 3.141592653589793f / 180.0f);

                            size_t start_slot = static_cast<size_t>(queue_size++);
                            size_t start_idx = q.idx(start_slot, worker_id);
                            q.r[start_idx] = start_row;
                            q.c[start_idx] = start_col;
                            q.flux[start_idx] = 1.0f;
                            q.z_delta[start_idx] = 0.0f;
                            size_t base_pzd = start_idx * 9;
                            for (int k = 0; k < 9; k++) {
                                q.parent_z_deltas[base_pzd + k] = 0.0f;
                            }
                            q.is_start[start_idx] = 1;
                            q.parent_is_start[start_idx] = 0;
                            q.num_parents[start_idx] = 0;
                            q.min_distance[start_idx] = 0.0f;
                            q.minDistXYZ[start_idx] = 0.0f;
                            int start_is_forest = (forestBool && forestInteraction && forest_map[start_flat_index] > 0.0f) ? 1 : 0;
                            q.isForest[start_idx] = start_is_forest;
                            q.forest_int_count[start_idx] = start_is_forest;
                        } else {
                            queue_size = 0;
                            q_idx = 0;
                        }
                    } else {
                        has_work = false;
                        queue_size = 0;
                        q_idx = 0;
                    }
                }

                if (sycl::none_of_group(sg, has_work)) {
                    break;
                }

                if (has_work && (q_idx < queue_size)) {
                    int curr_slot = q_idx++;
                    size_t curr_idx = q.idx(curr_slot, worker_id);
                    int curr_r = q.r[curr_idx];
                    int curr_c = q.c[curr_idx];
                    int curr_flat = curr_r * cols + curr_c;
                    float curr_flux = q.flux[curr_idx];
                    float curr_z_delta = q.z_delta[curr_idx];
                    uint8_t curr_is_start = q.is_start[curr_idx];
                    uint8_t curr_parent_is_start = q.parent_is_start[curr_idx];
                    float curr_min_distance = q.min_distance[curr_idx];
                    float curr_minDistXYZ = q.minDistXYZ[curr_idx];
                    int curr_forest_int_count = q.forest_int_count[curr_idx];
                
                // Calc z_delta_neighbour
                float altitude = dem[curr_flat];
                float z_delta_neighbour[3][3] = {0.0f};
                
                constexpr float RAD90 = 1.57079632679f;
                constexpr float SQRT2 = 1.41421356237f;
                
                // Forest friction influence on tanAlpha
                float thread_tanAlpha = tanAlpha;
                if (forestBool) {
                    if (forest_module_enum == 3) { // forestFrictionLayer
                        if (!curr_is_start && skipForestDist < curr_minDistXYZ) {
                            float FSI_val = forest_map[curr_flat];
                            float AlphaFor = 0.0f;
                            if (friction_layer_type_enum == 0) {
                                AlphaFor = FSI_val;
                            } else {
                                AlphaFor = thread_alpha + FSI_val;
                            }
                            if (AlphaFor < thread_alpha) AlphaFor = thread_alpha;
                            thread_tanAlpha = sycl::tan(AlphaFor * 3.141592653589793f / 180.0f);
                        }
                    } else if (forest_module_enum == 1 || forest_module_enum == 2) { // forestFriction / forestDetrainment
                        float FSI_val = forest_map[curr_flat];
                        if (!curr_is_start && FSI_val > 0.0f && skipForestDist < curr_minDistXYZ) {
                            float friction = minAddedFrictionFor;
                            if (curr_z_delta < noFrictionEffectZDelta) {
                                float rest = maxAddedFrictionFor * FSI_val;
                                float slope = (rest - minAddedFrictionFor) / (0.0f - noFrictionEffectZDelta);
                                friction = sycl::max(minAddedFrictionFor, slope * curr_z_delta + rest);
                            }
                            float alpha_calc = thread_alpha + sycl::max(0.0f, friction);
                            thread_tanAlpha = sycl::tan(alpha_calc * 3.141592653589793f / 180.0f);
                        }
                    }
                }

                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        int r_idx = dr + 1;
                        int c_idx = dc + 1;
                        
                        float ds = (dr != 0 && dc != 0) ? SQRT2 : 1.0f;
                        if (dr == 0 && dc == 0) ds = 0.0f;
                        
                        float dem_val = dem[(curr_r + dr) * cols + (curr_c + dc)];
                        float z_gamma = altitude - dem_val;
                        float z_alpha = ds * cellsize * thread_tanAlpha;
                        
                        float zd_neigh = curr_z_delta + z_gamma - z_alpha;
                        if (zd_neigh < 0.0f) zd_neigh = 0.0f;
                        if (zd_neigh > thread_max_z_delta) zd_neigh = thread_max_z_delta;
                        
                        z_delta_neighbour[r_idx][c_idx] = zd_neigh;
                    }
                }
                
                // Calc persistence
                float persistence[3][3] = {0.0f};
                float no_flow[3][3] = {
                    {1.0f, 1.0f, 1.0f},
                    {1.0f, 1.0f, 1.0f},
                    {1.0f, 1.0f, 1.0f}
                };
                
                if (curr_is_start || curr_parent_is_start) {
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            persistence[r][c] = 1.0f;
                        }
                    }
                } else {
                    size_t base_pzd = curr_idx * 9;
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            float maxweight = q.parent_z_deltas[base_pzd + (dr + 1) * 3 + (dc + 1)];
                            if (maxweight > 0.0f) {
                                no_flow[dr + 1][dc + 1] = 0.0f;
                                int dx = dc;
                                int dy = dr;
                                
                                if (dx == -1) {
                                    if (dy == -1) {
                                        persistence[2][2] += maxweight;
                                        persistence[2][1] += 0.707f * maxweight;
                                        persistence[1][2] += 0.707f * maxweight;
                                    } else if (dy == 0) {
                                        persistence[1][2] += maxweight;
                                        persistence[2][2] += 0.707f * maxweight;
                                        persistence[0][2] += 0.707f * maxweight;
                                    } else if (dy == 1) {
                                        persistence[0][2] += maxweight;
                                        persistence[0][1] += 0.707f * maxweight;
                                        persistence[1][2] += 0.707f * maxweight;
                                    }
                                } else if (dx == 0) {
                                    if (dy == -1) {
                                        persistence[2][1] += maxweight;
                                        persistence[2][0] += 0.707f * maxweight;
                                        persistence[2][2] += 0.707f * maxweight;
                                    } else if (dy == 1) {
                                        persistence[0][1] += maxweight;
                                        persistence[0][0] += 0.707f * maxweight;
                                        persistence[0][2] += 0.707f * maxweight;
                                    }
                                } else if (dx == 1) {
                                    if (dy == -1) {
                                        persistence[2][0] += maxweight;
                                        persistence[1][0] += 0.707f * maxweight;
                                        persistence[2][1] += 0.707f * maxweight;
                                    } else if (dy == 0) {
                                        persistence[1][0] += maxweight;
                                        persistence[0][0] += 0.707f * maxweight;
                                        persistence[2][0] += 0.707f * maxweight;
                                    } else if (dy == 1) {
                                        persistence[0][0] += maxweight;
                                        persistence[0][1] += 0.707f * maxweight;
                                        persistence[1][0] += 0.707f * maxweight;
                                    }
                                }
                            }
                        }
                    }
                }
                
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        persistence[r][c] *= no_flow[r][c];
                    }
                }
                
                // Calc tan_beta
                float tan_beta[3][3] = {0.0f};
                float r_t[3][3] = {0.0f};
                float sum_rt = 0.0f;
                
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) {
                            continue;
                        }
                        int r_idx = dr + 1;
                        int c_idx = dc + 1;
                        
                        if (z_delta_neighbour[r_idx][c_idx] <= 0.0f || persistence[r_idx][c_idx] <= 0.0f) {
                            continue;
                        }
                        
                        float ds = (dr != 0 && dc != 0) ? SQRT2 : 1.0f;
                        float distance = ds * cellsize;
                        float dem_val = dem[(curr_r + dr) * cols + (curr_c + dc)];
                        float slope_term = (altitude - dem_val) / distance;
                        float beta = sycl::atan(slope_term) + RAD90;
                        float tb = sycl::tan(beta / 2.0f);
                        
                        tan_beta[r_idx][c_idx] = tb;
                        float rt_val = sycl::pow(tb, static_cast<float>(thread_exp));
                        r_t[r_idx][c_idx] = rt_val;
                        sum_rt += rt_val;
                    }
                }
                
                if (sum_rt > 0.0f) {
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            r_t[r][c] /= sum_rt;
                        }
                    }
                }
                
                // Calc detrainment and adjust current flux
                float current_flux = curr_flux;
                if (forestBool && forestDetrainmentBool && !curr_is_start) {
                    float FSI_val = forest_map[curr_flat];
                    float rest = maxDetrainmentFor * FSI_val;
                    float slope = 0.0f;
                    if (noDetrainmentEffectZdelta != 0.0f) {
                        slope = (rest - minDetrainmentFor) / (0.0f - noDetrainmentEffectZdelta);
                    }
                    float detrainment = sycl::max(minDetrainmentFor, slope * curr_z_delta + rest);
                    current_flux = sycl::max(0.0003f, current_flux - detrainment);
                }

                // Calc dist
                float dist[3][3] = {0.0f};
                float sum_p_rt = 0.0f;
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        dist[r][c] = persistence[r][c] * r_t[r][c];
                        sum_p_rt += dist[r][c];
                    }
                }
                
                if (sum_p_rt > 0.0f) {
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            dist[r][c] = (dist[r][c] / sum_p_rt) * current_flux;
                        }
                    }
                }
                
                // Redistribute flux below threshold
                int count = 0;
                float mass_to_distribute = 0.0f;
                
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        float d_val = dist[r][c];
                        if (fluxDistOldVersionBool) {
                            if (d_val > 0.0f && d_val < flux_threshold) {
                                count++;
                            }
                        } else {
                            if (d_val >= flux_threshold) {
                                count++;
                            }
                        }
                        if (d_val < flux_threshold) {
                            mass_to_distribute += d_val;
                        }
                    }
                }
                
                if (mass_to_distribute > 0.0f && count > 0) {
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            if (dist[r][c] >= flux_threshold) {
                                dist[r][c] += mass_to_distribute / static_cast<float>(count);
                            } else {
                                dist[r][c] = 0.0f;
                            }
                        }
                    }
                }
                
                float sum_dist = 0.0f;
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        sum_dist += dist[r][c];
                    }
                }
                
                if (sum_dist != current_flux && count > 0) {
                    float diff = (current_flux - sum_dist) / static_cast<float>(count);
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            if (dist[r][c] >= flux_threshold) {
                                dist[r][c] += diff;
                            }
                        }
                    }
                }
                
                // Distribute to valid neighbors
                struct NeighborNode {
                    int r;
                    int c;
                    float flux;
                    float z_delta;
                };
                NeighborNode valid_neighbors[8];
                int num_valid = 0;

                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) {
                            continue;
                        }
                        int r_idx = dr + 1;
                        int c_idx = dc + 1;
                        float dist_val = dist[r_idx][c_idx];
                        if (dist_val >= flux_threshold) {
                            int nr = curr_r + dr;
                            int nc = curr_c + dc;
                            if (is_neighborhood_valid(nr, nc)) {
                                float z_delta_neigh_val = z_delta_neighbour[r_idx][c_idx];
                                valid_neighbors[num_valid++] = NeighborNode{nr, nc, dist_val, z_delta_neigh_val};
                            }
                        }
                    }
                }

                // Sort valid neighbors lexicographically: z_delta, flux, r, c (ascending)
                for (int i = 0; i < num_valid - 1; i++) {
                    for (int j = 0; j < num_valid - i - 1; j++) {
                        bool swap_needed = false;
                        if (valid_neighbors[j].z_delta != valid_neighbors[j + 1].z_delta) {
                            swap_needed = valid_neighbors[j].z_delta > valid_neighbors[j + 1].z_delta;
                        } else if (valid_neighbors[j].flux != valid_neighbors[j + 1].flux) {
                            swap_needed = valid_neighbors[j].flux > valid_neighbors[j + 1].flux;
                        } else if (valid_neighbors[j].r != valid_neighbors[j + 1].r) {
                            swap_needed = valid_neighbors[j].r > valid_neighbors[j + 1].r;
                        } else {
                            swap_needed = valid_neighbors[j].c > valid_neighbors[j + 1].c;
                        }

                        if (swap_needed) {
                            NeighborNode temp = valid_neighbors[j];
                            valid_neighbors[j] = valid_neighbors[j + 1];
                            valid_neighbors[j + 1] = temp;
                        }
                    }
                }

                // Distribute to valid neighbors in sorted order
                for (int n = 0; n < num_valid; n++) {
                    int nr = valid_neighbors[n].r;
                    int nc = valid_neighbors[n].c;
                    float dist_val = valid_neighbors[n].flux;
                    float z_delta_neigh_val = valid_neighbors[n].z_delta;
                    int dr = nr - curr_r;
                    int dc = nc - curr_c;

                    int found_idx = -1;
                    for (int i = q_idx; i < queue_size; i++) {
                        size_t check_idx = q.idx(i, worker_id);
                        if (q.r[check_idx] == nr && q.c[check_idx] == nc) {
                            found_idx = i;
                            break;
                        }
                    }

                    float dx = (dc == 0) ? 0.0f : cellsize;
                    float dy = (dr == 0) ? 0.0f : cellsize;
                    float dz = sycl::fabs(dem[curr_flat] - dem[nr * cols + nc]);
                    float dist2d = sycl::sqrt(dx * dx + dy * dy);
                    float dist3d = sycl::sqrt(dy * dy + dy * dy + dz * dz); // Replicate Python typo where dx is replaced by dy

                    if (found_idx != -1) {
                        size_t f_idx = q.idx(found_idx, worker_id);
                        q.flux[f_idx] += dist_val;
                        q.parent_z_deltas[q.parent_zd_idx(found_idx, worker_id, 1 - dr, 1 - dc)] = curr_z_delta;
                        if (z_delta_neigh_val > q.z_delta[f_idx]) {
                            q.z_delta[f_idx] = z_delta_neigh_val;
                            q.parent_is_start[f_idx] = curr_is_start;
                        }
                        
                        float candidate_2d = curr_min_distance + dist2d;
                        if (candidate_2d < q.min_distance[f_idx]) {
                            q.min_distance[f_idx] = candidate_2d;
                        }
                        if (forestBool) {
                            float candidate_3d = curr_minDistXYZ + dist3d;
                            if (candidate_3d < q.minDistXYZ[f_idx]) {
                                q.minDistXYZ[f_idx] = candidate_3d;
                            }
                        }
                        if (forestBool && forestInteraction) {
                            int candidate_forest = curr_forest_int_count + q.isForest[f_idx];
                            if (candidate_forest < q.forest_int_count[f_idx]) {
                                q.forest_int_count[f_idx] = candidate_forest;
                            }
                        }
                        
                        // Add parent tracking
                        bool already_parent = false;
                        int np = q.num_parents[f_idx];
                        for (int p = 0; p < np; p++) {
                            if (q.parent_indices[q.parent_idx(found_idx, worker_id, p)] == curr_slot) {
                                already_parent = true;
                                break;
                            }
                        }
                        if (!already_parent && np < 8) {
                            q.parent_indices[q.parent_idx(found_idx, worker_id, np)] = curr_slot;
                            q.num_parents[f_idx] = np + 1;
                        }
                    } else {
                        if (queue_size < capacity_per_path) {
                            size_t new_slot = static_cast<size_t>(queue_size++);
                            size_t new_idx = q.idx(new_slot, worker_id);
                            q.r[new_idx] = nr;
                            q.c[new_idx] = nc;
                            q.flux[new_idx] = dist_val;
                            q.z_delta[new_idx] = z_delta_neigh_val;
                            
                            size_t base_pzd = new_idx * 9;
                            for (int k = 0; k < 9; k++) {
                                q.parent_z_deltas[base_pzd + k] = 0.0f;
                            }
                            q.parent_z_deltas[base_pzd + (1 - dr) * 3 + (1 - dc)] = curr_z_delta;
                            
                            q.is_start[new_idx] = 0;
                            q.parent_is_start[new_idx] = curr_is_start;
                            
                            q.min_distance[new_idx] = curr_min_distance + dist2d;
                            if (forestBool) {
                                q.minDistXYZ[new_idx] = curr_minDistXYZ + dist3d;
                            } else {
                                q.minDistXYZ[new_idx] = 0.0f;
                            }
                            int new_is_forest = (forestBool && forestInteraction && forest_map[nr * cols + nc] > 0.0f) ? 1 : 0;
                            q.isForest[new_idx] = new_is_forest;
                            q.forest_int_count[new_idx] = curr_forest_int_count + new_is_forest;
                            
                            q.num_parents[new_idx] = 1;
                            q.parent_indices[q.parent_idx(new_slot, worker_id, 0)] = curr_slot;
                        }
                    }
                }
                }
            }
        });
        q.wait_and_throw();

        // Query diagnostic peak queue depth
        int host_max_queue_depth = 0;
        q.memcpy(&host_max_queue_depth, d_max_queue_depth, sizeof(int));
        q.wait_and_throw();
        sycl::free(d_max_queue_depth, q);

        double queue_pct = (100.0 * host_max_queue_depth) / capacity_per_path;
        std::cout << "[SYCL Diagnostic] Peak Queue Depth Reached: " << host_max_queue_depth 
                  << " / " << capacity_per_path << " nodes (" << std::fixed << std::setprecision(1) 
                  << queue_pct << "% capacity utilized)" << std::endl;

        // Device to Host memory copies
        q.memcpy(host_z_delta.data(), z_delta, total_cells * sizeof(float));
        q.memcpy(host_flux.data(), flux, total_cells * sizeof(float));
        q.memcpy(host_counts.data(), counts, total_cells * sizeof(int));
        q.memcpy(host_backcalc.data(), backcalc, total_cells * sizeof(float));
        q.memcpy(host_forest_int.data(), forest_int, total_cells * sizeof(float));

        q.wait_and_throw();

        // Free device allocations
        sycl::free(rel_indices, q);
        sycl::free(dem, q);
        sycl::free(z_delta, q);
        sycl::free(flux, q);
        sycl::free(counts, q);
        sycl::free(var_alpha, q);
        sycl::free(var_exponent, q);
        sycl::free(var_umax, q);
        sycl::free(infra_map, q);
        sycl::free(forest_map, q);
        sycl::free(backcalc, q);
        sycl::free(forest_int, q);
        sycl::free(d_work_pool, q);
        sycl::free(d_task_counter, q);
    }
    
    // Convert vectors to 2D numpy arrays
    auto py_z_delta = py::array_t<float>({rows, cols});
    auto py_flux = py::array_t<float>({rows, cols});
    auto py_counts = py::array_t<int>({rows, cols});
    auto py_backcalc = py::array_t<float>({rows, cols});
    auto py_forest_int = py::array_t<float>({rows, cols});
    
    std::copy(host_z_delta.begin(), host_z_delta.end(), static_cast<float*>(py_z_delta.request().ptr));
    std::copy(host_flux.begin(), host_flux.end(), static_cast<float*>(py_flux.request().ptr));
    std::copy(host_counts.begin(), host_counts.end(), static_cast<int*>(py_counts.request().ptr));
    
    // Post-process backcalc and forest_int on host
    for (int i = 0; i < total_cells; i++) {
        if (host_counts[i] <= 0) {
            host_backcalc[i] = -9999.0f;
            host_forest_int[i] = -9999.0f;
        } else {
            if (host_forest_int[i] > 99999.0f) {
                host_forest_int[i] = -9999.0f;
            }
        }
    }
    
    std::copy(host_backcalc.begin(), host_backcalc.end(), static_cast<float*>(py_backcalc.request().ptr));
    std::copy(host_forest_int.begin(), host_forest_int.end(), static_cast<float*>(py_forest_int.request().ptr));
    
    return py::make_tuple(py_z_delta, py_flux, py_counts, py_backcalc, py_forest_int);
}

PYBIND11_MODULE(sycl_core, m) {
    m.def("run_sycl_calculation", &run_sycl_calculation);
}

}
