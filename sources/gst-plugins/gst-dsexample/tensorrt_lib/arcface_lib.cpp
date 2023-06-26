#include <iostream>
#include <chrono>
#include <vector>
#include "cuda_runtime_api.h"
#include <NvInfer.h>
#include "logging.h"
#include "arcface_lib.h"
#include "cuda_utils.h"
#include "utils.h"

using namespace nvinfer1;

const char* ARCFACE_INPUT_BLOB_NAME = "input";
const char* ARCFACE_OUTPUT_BLOB_NAME = "output";

// void decode_outputs(std::map<std::string, std::vector<float>> &face_csv, float* output_emb, 
//                            int curBatchSize, std::vector<DetectBox>&  det, float conf_thresh){
//     for(int i=0; i< curBatchSize; i++)
//     {   
//         float max = 0.0;
//         std::string facename;
//         std::map<std::string, std::vector<float>>::iterator face_csv_item;
//         for (face_csv_item = face_csv.begin(); face_csv_item != face_csv.end(); face_csv_item++)
//         {   
//             float num = 0.0;
//             float csv_sum_l2 = 0.0;
//             float cur_sum_l2 = 0.0;
//             float sim = 0.0;
//             float denom = 0.0;
//             assert(face_csv_item->second.size() == ARCFACE_EMB_SIZE);
//             for(int j = 0; j < ARCFACE_EMB_SIZE; j++)
//             {
//                 num += face_csv_item->second[j] * output_emb[i*ARCFACE_EMB_SIZE + j];
//                 csv_sum_l2 += face_csv_item->second[j] * face_csv_item->second[j];
//                 cur_sum_l2 += output_emb[i*ARCFACE_EMB_SIZE + j] * output_emb[i*ARCFACE_EMB_SIZE + j];
//             }
//             denom = std::sqrt(csv_sum_l2) * std::sqrt(cur_sum_l2);
//             sim = 0.5 + 0.5 * (num / denom);
//             if(sim > max)
//             {
//                 max = sim;
//                 facename = face_csv_item->first;
//             }
//         }
//         if(max > conf_thresh)
//         {   
//             det[i].confidence = max;
//             det[i].classname = facename;
//         }
//     }
// }

static void detectnet_doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers,
                        float* output, int output_size, int curBatchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    // CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, LANDMARK_ARCFACE_MAX_BATCH_SIZE * 3 * LANDMARK_ARCFACE_INPUT_H * LANDMARK_ARCFACE_INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    Dims4 inputDims{curBatchSize, 3, LANDMARK_ARCFACE_INPUT_H, LANDMARK_ARCFACE_INPUT_W};
    context.setBindingDimensions(0, inputDims);
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void * arcface_trt_create(const char * engine_name, const char* face_csv_name, std::vector<float> &face_csv, std::vector<std::string> &faces_name)
{   
    //get the face csv
    std::ifstream cvs_file;
    cvs_file.open(face_csv_name, std::ios::in);
    std::string line;
    while(std::getline(cvs_file, line))
    {
        std::istringstream face_tensor_info(line);
        std::string face_name;

        if(!(face_tensor_info >> face_name))
        {
            std::cout << "read face csv faile!!!!" << std::endl;
            exit(0);
        }
        faces_name.push_back(face_name);
        float face_tensor[ARCFACE_EMB_SIZE];
        int i = 0;
        float face_tensor_item;
        while((face_tensor_info >> face_tensor_item))
        {
            face_tensor[i] = face_tensor_item;
            i++;
        }
        face_csv.insert(face_csv.end(), face_tensor, face_tensor+ARCFACE_EMB_SIZE);
    }
    cvs_file.close();
    
    size_t size = 0;
    char *trtModelStream = NULL;
    ArcfaceTRTContext * trt_ctx = NULL;
 
    trt_ctx = new ArcfaceTRTContext();
    assert(faces_name.size() > 0);
    CUDA_CHECK(cudaMalloc(&trt_ctx->facecsv_buffer_device[0], faces_name.size() * ARCFACE_EMB_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&trt_ctx->facecsv_buffer_device[1], faces_name.size() * sizeof(float)));
 
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }else
        return NULL;
 
    trt_ctx->runtime = createInferRuntime(gLogger);
    assert(trt_ctx->runtime != nullptr);
    trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trt_ctx->engine != nullptr);
    trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();
    delete[] trtModelStream;
    assert(trt_ctx->engine->getNbBindings() == 2);
 
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(ARCFACE_INPUT_BLOB_NAME);
    trt_ctx->outputIndex = trt_ctx->engine->getBindingIndex(ARCFACE_OUTPUT_BLOB_NAME);

    assert(trt_ctx->inputIndex == 0);
    assert(trt_ctx->outputIndex == 1);
    
    // trt_ctx->input_data = new float[LANDMARK_ARCFACE_MAX_BATCH_SIZE * 3 * LANDMARK_ARCFACE_INPUT_H * LANDMARK_ARCFACE_INPUT_W];
    trt_ctx->OUTPUT_SIZE = LANDMARK_ARCFACE_MAX_BATCH_SIZE * ARCFACE_EMB_SIZE;
    trt_ctx->output  = new float[trt_ctx->OUTPUT_SIZE];

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], LANDMARK_ARCFACE_MAX_BATCH_SIZE * 3 * LANDMARK_ARCFACE_INPUT_H * LANDMARK_ARCFACE_INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], LANDMARK_ARCFACE_MAX_BATCH_SIZE * ARCFACE_EMB_SIZE * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(trt_ctx->facecsv_buffer_device[0], face_csv.data(), faces_name.size() * ARCFACE_EMB_SIZE * sizeof(float), cudaMemcpyHostToDevice, trt_ctx->cuda_stream));
    return (void *)trt_ctx;
}

int arcface_trt_detect(void *h, int curBatchSize)
{   
    // std::cout << "arcface detect" << std::endl; 
    ArcfaceTRTContext *trt_ctx;
    trt_ctx = (ArcfaceTRTContext *)h;
 	// whether det is empty , if not, empty det

    // Run inference
    detectnet_doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, trt_ctx->output, 
                        trt_ctx->OUTPUT_SIZE, curBatchSize);
    return 1;
}

void arcface_trt_destroy(void *h)
{
    ArcfaceTRTContext *trt_ctx;
 
    trt_ctx = (ArcfaceTRTContext *)h;
    
    // Release stream and buffers
    cudaStreamDestroy(trt_ctx->cuda_stream);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]));
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->outputIndex]));
    
    // Destroy the engine
    trt_ctx->exe_context->destroy();
    trt_ctx->engine->destroy();
    trt_ctx->runtime->destroy();
 
    // delete trt_ctx->input_data;
    delete trt_ctx->output;
 
    delete trt_ctx;
}
