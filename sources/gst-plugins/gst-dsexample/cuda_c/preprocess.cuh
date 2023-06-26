#ifndef PREPROCESS_CUH
#define PREPROCESS_CUH

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <assert.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

__global__ void chwtohwc_kernel(uint8_t* src, int id, float* dst, int count);
__global__ void arcface_chwtohwc_kernel(uint8_t* src, int id, float* dst, int count);
__global__ void cos_dis_dot(float *facecsv_tensor, float *cur_tensor, float *output, int id, int count);

extern "C"
void cuda_preprocess_init();
void cuda_preprocess(uint8_t* src, int id, float* dst, int img_w, int img_h, cudaStream_t stream);
void cuda_preprocess_destroy();

void arcface_cuda_preprocess_init();
void arcface_cuda_preprocess(uint8_t* src, int id, float* dst, int img_w, int img_h, cudaStream_t stream);
void arcface_cuda_preprocess_destroy();

void arcface_cuda_cosdis(float *facecsv_tensor, float* cur_tensor, int index, int count, float*out, cudaStream_t stream);
#endif

