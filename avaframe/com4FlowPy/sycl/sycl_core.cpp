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
    py::array_t<float> dem_arr,
    py::array_t<float> release_arr,
    float alpha,
    int exp,
    float flux_threshold,
    float max_z_delta,
    float cellsize,
    float nodata,
    std::string device_type
) {
    // Extract pointer and dimension info from Python arrays
    py::buffer_info dem_info = dem_arr.request();
    py::buffer_info rel_info = release_arr.request();
    
    int rows = dem_info.shape[0];
    int cols = dem_info.shape[1];
    int total_cells = rows * cols;
    
    float* dem_ptr = static_cast<float*>(dem_info.ptr);
    float* rel_ptr = static_cast<float*>(rel_info.ptr);
    
    // Select execution device
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
