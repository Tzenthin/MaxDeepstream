#include "preprocess.cuh"
static uint8_t* img_buffer_device = nullptr;
static uint8_t* arcface_img_buffer_device = nullptr;
// must be the arcface embeding size
const int threadsPerBlock = 512;

__global__ void chwtohwc_kernel(
        uint8_t* src, int id, float* dst, int count) {
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x)
    {
        dst[id *count *3 + count * 0 + index] = src[index * 3 + 0] ;
        dst[id *count *3 + count * 1 + index] = src[index * 3 + 1] ;
        dst[id *count *3 + count * 2 + index] = src[index * 3 + 2] ;
    }
}

__global__ void arcface_chwtohwc_kernel(
        uint8_t* src, int id, float* dst, int count) {
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x)
    {
        dst[id *count *3 + count * 0 + index] = ((src[index * 3 + 0] / 255.)- 0.5)/0.5;
        dst[id *count *3 + count * 1 + index] = ((src[index * 3 + 1] / 255.)- 0.5)/0.5;
        dst[id *count *3 + count * 2 + index] = ((src[index * 3 + 2] / 255.)- 0.5)/0.5;
    }
}

__global__ void cos_dis_dot(float *facecsv_tensor, float *cur_tensor, float *output, int id, int count)
{
    __shared__ float dot_cache[threadsPerBlock];
    __shared__ float facecsv_tensor_cache[threadsPerBlock];
    __shared__ float cur_tensor_cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    int tid_cur = tid % threadsPerBlock;

    float dot_temp = 0.0;
    float facecsv_tensor_temp = 0.0;
    float cur_tensor_temp = 0.0;
    while(tid < count)
    {
      dot_temp += facecsv_tensor[tid] * cur_tensor[id * threadsPerBlock + tid_cur];
      facecsv_tensor_temp += facecsv_tensor[tid] * facecsv_tensor[tid];
      cur_tensor_temp += cur_tensor[id * threadsPerBlock + tid_cur] * cur_tensor[id * threadsPerBlock + tid_cur];
      tid += blockDim.x * gridDim.x;
    }

    dot_cache[cacheIndex] = dot_temp;
    facecsv_tensor_cache[cacheIndex] = facecsv_tensor_temp;
    cur_tensor_cache[cacheIndex] = cur_tensor_temp;

    __syncthreads();

    int i = blockDim.x/2;
    while(i != 0)
    {
      if(cacheIndex < i)
      {
        dot_cache[cacheIndex] += dot_cache[cacheIndex + i];
        facecsv_tensor_cache[cacheIndex] += facecsv_tensor_cache[cacheIndex + i];
        cur_tensor_cache[cacheIndex] += cur_tensor_cache[cacheIndex+i];
      }
      __syncthreads();
      i /= 2;
    }
    if(cacheIndex == 0)
    {
      output[blockIdx.x] = (dot_cache[0] / (std::sqrt(facecsv_tensor_cache[0]) * std::sqrt(cur_tensor_cache[0]))) * 0.5 + 0.5 ;
    }
    

}


void cuda_preprocess_init() {
  // prepare input data in device memory
  CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, 960 * 540 * 3 * sizeof(uint8_t)));
}

void cuda_preprocess(uint8_t* src, int id, float* dst, int img_w, int img_h, cudaStream_t stream)
{   
    
    int img_size = img_w * img_h * 3;
    // copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, src, img_size*sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    int count = img_w * img_h;
    dim3 dimGrid((count + 1024 - 1) / 1024);
    dim3 dimBlock(1024);
    chwtohwc_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      img_buffer_device, id, dst, count);
    
}

void cuda_preprocess_destroy() {
  CUDA_CHECK(cudaFree(img_buffer_device));
}

//arcface preprocess
void arcface_cuda_preprocess_init() {
  // prepare input data in device memory
  CUDA_CHECK(cudaMalloc((void**)&arcface_img_buffer_device, 112 * 112 * 3 * sizeof(uint8_t)));
}

void arcface_cuda_preprocess(uint8_t* src, int id, float* dst, int img_w, int img_h, cudaStream_t stream)
{   
    
    int img_size = img_w * img_h * 3;
    // copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(arcface_img_buffer_device, src, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    int count = img_w * img_h;
    dim3 dimGrid((count + 1024 - 1) / 1024);
    dim3 dimBlock(1024);
    arcface_chwtohwc_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      arcface_img_buffer_device, id, dst, count);
    
}

void arcface_cuda_preprocess_destroy() {
  CUDA_CHECK(cudaFree(arcface_img_buffer_device));
}

// //cos distance (tensor dot)
void arcface_cuda_cosdis(float *facecsv_tensor, float* cur_tensor, int index, int count, float*out, cudaStream_t stream)
{
    dim3 dimGrid((count + 512 - 1) / 512);
    dim3 dimBlock(512);
    cos_dis_dot<<<dimGrid, dimBlock, 0, stream>>>(
      facecsv_tensor, cur_tensor, out, index, count);
}