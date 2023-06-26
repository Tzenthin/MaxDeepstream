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
    float* output;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* exe_context;
    void* buffers[2];
    cudaStream_t cuda_stream;
    int inputIndex;
    int outputIndex;
    int OUTPUT_SIZE;
    void* facecsv_buffer_device[2];
}ArcfaceTRTContext;

// void decode_outputs(std::map<std::string, std::vector<float>> &face_csv, float* output_emb, 
//                            int curBatchSize, std::vector<DetectBox>&  det, float conf_thresh);

void * arcface_trt_create(const char * engine_name, const char* face_csv_name, std::vector<float> &face_csv, std::vector<std::string> &faces_name);

int arcface_trt_detect(void *h, int curBatchSize);
void arcface_trt_destroy(void *h);
 
#ifdef __cplusplus
}
#endif 