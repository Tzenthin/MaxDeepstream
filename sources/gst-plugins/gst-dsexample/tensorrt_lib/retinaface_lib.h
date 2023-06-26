#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "utils.h"
#include <NvInfer.h>
using namespace nvinfer1;
#ifdef __cplusplus
extern "C" 
{
#endif 
typedef struct
{
    // float* input;
    float* output;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* exe_context;
    void* buffers[2];
    cudaStream_t cuda_stream;
    cv::Mat refer_matrix;
    int inputIndex;
    int outputIndex;
    int OUTPUT_SIZE;
}RetinaFaceTRTContext;
 
void retinaface_decode_outputs(cv::Mat &img, cv::Mat &refer_matrix, float* dec_out, float conf_thresh, int source_index,
            std::vector<DetectBox>& det, std::vector<cv::Mat> &align_imgs);

void * retinaface_trt_create(const char * engine_name, int batch_size);
 
int retinaface_trt_detect(void *h);
 
void retinaface_trt_destroy(void *h);
 
#ifdef __cplusplus
}
#endif 


