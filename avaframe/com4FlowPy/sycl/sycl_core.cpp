#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace py = pybind11;

namespace sycl_flow {

#define MAX_QUEUE_SIZE 1024
#define MAX_VISITED 1024

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

    // none checking
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
        forestModule = fp["forestModule"].cast<std::string>();
        maxAddedFrictionFor = fp["maxAddedFrictionFor"].cast<float>();
        minAddedFrictionFor = fp["minAddedFrictionFor"].cast<float>();
        velThForFriction = fp["velThForFriction"].cast<float>();
        maxDetrainmentFor = fp["maxDetrainmentFor"].cast<float>();
        minDetrainmentFor = fp["minDetrainmentFor"].cast<float>();
        velThForDetrain = fp["velThForDetrain"].cast<float>();
        forestFrictionLayerType = fp["forestFrictionLayerType"].cast<std::string>();
        skipForestDist = fp["skipForestDist"].cast<float>();
    }

    // find release indices
    std::vector<int> release_flat_indices;
    for (int i = 0; i < total_cells; i++) {
        if (rel_ptr[i] > 0.0f) {
            release_flat_indices.push_back(i);
        }
    }
    int num_release_cells = release_flat_indices.size();

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
    
    sycl::queue q(dev);
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    // Buffer vectors
    std::vector<float> host_z_delta(total_cells, 0.0f);
    std::vector<float> host_flux(total_cells, 0.0f);
    std::vector<int> host_counts(total_cells, 0);
    
    // Dummy buffers to prevent SYCL crash on nullptr
    std::vector<float> dummy(total_cells, 0.0f);
    
    float* var_alpha_data = varAlpha_ptr ? varAlpha_ptr : dummy.data();
    float* var_exponent_data = varExponent_ptr ? varExponent_ptr : dummy.data();
    float* var_umax_data = varUmax_ptr ? varUmax_ptr : dummy.data();

    // create buffers and execute kernel
    if (num_release_cells > 0) {
        sycl::buffer<int, 1> buf_release_indices(release_flat_indices.data(), sycl::range<1>(num_release_cells));
        sycl::buffer<float, 1> buf_dem(dem_ptr, sycl::range<1>(total_cells));
        sycl::buffer<float, 1> buf_z_delta(host_z_delta.data(), sycl::range<1>(total_cells));
        sycl::buffer<float, 1> buf_flux(host_flux.data(), sycl::range<1>(total_cells));
        sycl::buffer<int, 1> buf_counts(host_counts.data(), sycl::range<1>(total_cells));
        
        sycl::buffer<float, 1> buf_var_alpha(var_alpha_data, sycl::range<1>(total_cells));
        sycl::buffer<float, 1> buf_var_exponent(var_exponent_data, sycl::range<1>(total_cells));
        sycl::buffer<float, 1> buf_var_umax(var_umax_data, sycl::range<1>(total_cells));

        q.submit([&](sycl::handler& cgh) {
            // Accessors
            auto dem = buf_dem.get_access<sycl::access::mode::read>(cgh);
            auto rel_indices = buf_release_indices.get_access<sycl::access::mode::read>(cgh);
            auto z_delta = buf_z_delta.get_access<sycl::access::mode::read_write>(cgh);
            auto flux = buf_flux.get_access<sycl::access::mode::read_write>(cgh);
            auto counts = buf_counts.get_access<sycl::access::mode::read_write>(cgh);
            auto var_alpha = buf_var_alpha.get_access<sycl::access::mode::read>(cgh);
            auto var_exponent = buf_var_exponent.get_access<sycl::access::mode::read>(cgh);
            auto var_umax = buf_var_umax.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for(sycl::range<1>(num_release_cells), [=](sycl::id<1> idx) {
                int thread_id = idx[0];
                int start_flat_index = rel_indices[thread_id];
                int start_row = start_flat_index / cols;
                int start_col = start_flat_index % cols;
                
                auto is_nodata = [=](float val) {
                    return sycl::fabs(val - nodata) < 1e-3f;
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

                // Validate start cell neighborhood
                if (!is_neighborhood_valid(start_row, start_col)) {
                    return;
                }

                //if (thread_id < 5) {
                //    printf("SYCL Thread %d mapped to release cell (%d, %d) flat_idx: %d | DEM: %f\n", thread_id, start_row, start_col, start_flat_index, dem[start_flat_index]);
                //}
                
                float thread_max_z_delta = max_z_delta;
                float thread_alpha = alpha;
                int thread_exp = static_cast<int>(exp);

                if (varUmaxBool && var_umax[start_flat_index] > 0.0f && var_umax[start_flat_index] <= 8848.0f) {
                    thread_max_z_delta = var_umax[start_flat_index];
                }
                if (varAlphaBool && var_alpha[start_flat_index] > 0.0f && var_alpha[start_flat_index] <= 90.0f) {
                    thread_alpha = var_alpha[start_flat_index];
                }
                if (varExponentBool && var_exponent[start_flat_index] > 0.0f) {
                    thread_exp = static_cast<int>(var_exponent[start_flat_index]);
                }
                
                float tanAlpha = sycl::tan(thread_alpha * 3.141592653589793f / 180.0f);

                struct PathNode {
                    int r;
                    int c;
                    float flux;
                    float z_delta;
                    float parent_z_deltas[3][3];
                    bool is_start;
                    bool parent_is_start;
                };

                PathNode queue[MAX_QUEUE_SIZE];
                int queue_size = 0;

                PathNode start_node;
                start_node.r = start_row;
                start_node.c = start_col;
                start_node.flux = 1.0f;
                start_node.z_delta = 0.0f;
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        start_node.parent_z_deltas[r][c] = 0.0f;
                    }
                }
                start_node.is_start = true;
                start_node.parent_is_start = false;
                queue[queue_size++] = start_node;

                int q_idx = 0;
                while (q_idx < queue_size) {
                    PathNode curr = queue[q_idx++];
                    int curr_flat = curr.r * cols + curr.c;
                    
                    // Write z_delta and flux via fetch_max
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> 
                        atomic_z_delta(z_delta[curr_flat]);
                    atomic_z_delta.fetch_max(curr.z_delta);
                    
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> 
                        atomic_flux(flux[curr_flat]);
                    atomic_flux.fetch_max(curr.flux);
                    
                    // Write counts if first visit in queue for this path
                    bool first_visit = true;
                    for (int i = 0; i < q_idx - 1; i++) {
                        if (queue[i].r == curr.r && queue[i].c == curr.c) {
                            first_visit = false;
                            break;
                        }
                    }
                    if (first_visit) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> 
                            atomic_counts(counts[curr_flat]);
                        atomic_counts.fetch_add(1);
                    }
                    
                    // Calc z_delta_neighbour
                    float altitude = dem[curr_flat];
                    float z_delta_neighbour[3][3] = {0.0f};
                    
                    float RAD90 = 1.57079632679f;
                    float SQRT2 = 1.41421356237f;
                    
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int r_idx = dr + 1;
                            int c_idx = dc + 1;
                            
                            float ds = (dr != 0 && dc != 0) ? SQRT2 : 1.0f;
                            if (dr == 0 && dc == 0) ds = 0.0f;
                            
                            float dem_val = dem[(curr.r + dr) * cols + (curr.c + dc)];
                            float z_gamma = altitude - dem_val;
                            float z_alpha = ds * cellsize * tanAlpha;
                            
                            float zd_neigh = curr.z_delta + z_gamma - z_alpha;
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
                    
                    if (curr.is_start || curr.parent_is_start) {
                        for (int r = 0; r < 3; r++) {
                            for (int c = 0; c < 3; c++) {
                                persistence[r][c] = 1.0f;
                            }
                        }
                    } else {
                        for (int dr = -1; dr <= 1; dr++) {
                            for (int dc = -1; dc <= 1; dc++) {
                                float maxweight = curr.parent_z_deltas[dr + 1][dc + 1];
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
                            float dem_val = dem[(curr.r + dr) * cols + (curr.c + dc)];
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
                                dist[r][c] = (dist[r][c] / sum_p_rt) * curr.flux;
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
                    
                    if (sum_dist != curr.flux && count > 0) {
                        float diff = (curr.flux - sum_dist) / static_cast<float>(count);
                        for (int r = 0; r < 3; r++) {
                            for (int c = 0; c < 3; c++) {
                                if (dist[r][c] >= flux_threshold) {
                                    dist[r][c] += diff;
                                }
                            }
                        }
                    }
                    
                    // Distribute to valid neighbors
                    // Collect valid neighbors
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
                                int nr = curr.r + dr;
                                int nc = curr.c + dc;
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
                        int dr = nr - curr.r;
                        int dc = nc - curr.c;

                        int found_idx = -1;
                        for (int i = q_idx; i < queue_size; i++) {
                            if (queue[i].r == nr && queue[i].c == nc) {
                                found_idx = i;
                                break;
                            }
                        }
                        if (found_idx != -1) {
                            queue[found_idx].flux += dist_val;
                            queue[found_idx].parent_z_deltas[1 - dr][1 - dc] = curr.z_delta;
                            if (z_delta_neigh_val > queue[found_idx].z_delta) {
                                queue[found_idx].z_delta = z_delta_neigh_val;
                                queue[found_idx].parent_is_start = curr.is_start;
                            }
                        } else {
                            if (queue_size < MAX_QUEUE_SIZE) {
                                PathNode new_node;
                                new_node.r = nr;
                                new_node.c = nc;
                                new_node.flux = dist_val;
                                new_node.z_delta = z_delta_neigh_val;
                                for (int r = 0; r < 3; r++) {
                                    for (int c = 0; c < 3; c++) {
                                        new_node.parent_z_deltas[r][c] = 0.0f;
                                    }
                                }
                                new_node.parent_z_deltas[1 - dr][1 - dc] = curr.z_delta;
                                new_node.is_start = false;
                                new_node.parent_is_start = curr.is_start;
                                queue[queue_size++] = new_node;
                            }
                        }
                    }
                }
            });
        });
        
        q.wait_and_throw();
    } // Buffer destructors block and copy results from device back to host vector
    
    // Convert vectors to numpy arrays
    auto py_z_delta = py::array_t<float>(total_cells);
    auto py_flux = py::array_t<float>(total_cells);
    auto py_counts = py::array_t<int>(total_cells);
    
    std::copy(host_z_delta.begin(), host_z_delta.end(), static_cast<float*>(py_z_delta.request().ptr));
    std::copy(host_flux.begin(), host_flux.end(), static_cast<float*>(py_flux.request().ptr));
    std::copy(host_counts.begin(), host_counts.end(), static_cast<int*>(py_counts.request().ptr));
    
    // Reshape to 2D
    py_z_delta.resize({rows, cols});
    py_flux.resize({rows, cols});
    py_counts.resize({rows, cols});
    
    return py::make_tuple(py_z_delta, py_flux, py_counts);
}

PYBIND11_MODULE(sycl_core, m) {
    m.def("run_sycl_calculation", &run_sycl_calculation);
}

}
