#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace py = pybind11;

namespace sycl_flow {

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
    
    // create buffers
    {
        sycl::buffer<float, 1> buf_dem(dem_ptr, sycl::range<1>(total_cells));
        sycl::buffer<float, 1> buf_rel(rel_ptr, sycl::range<1>(total_cells));
        sycl::buffer<float, 1> buf_z_delta(host_z_delta.data(), sycl::range<1>(total_cells));
        sycl::buffer<float, 1> buf_flux(host_flux.data(), sycl::range<1>(total_cells));
        sycl::buffer<int, 1> buf_counts(host_counts.data(), sycl::range<1>(total_cells));
        
        q.submit([&](sycl::handler& cgh) {
            // Accessors
            auto dem = buf_dem.get_access<sycl::access::mode::read>(cgh);
            auto rel = buf_rel.get_access<sycl::access::mode::read>(cgh);
            auto z_delta = buf_z_delta.get_access<sycl::access::mode::write>(cgh);
            auto flux = buf_flux.get_access<sycl::access::mode::write>(cgh);
            auto counts = buf_counts.get_access<sycl::access::mode::write>(cgh);
            
            // Dummy
            cgh.parallel_for(sycl::range<1>(total_cells), [=](sycl::id<1> idx) {
                int i = idx[0];
                z_delta[i] = dem[i] + 10.0f;
                flux[i] = rel[i] * 5.0f;
                counts[i] = 1;
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
