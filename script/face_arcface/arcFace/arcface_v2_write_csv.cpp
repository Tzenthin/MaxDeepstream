#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <unistd.h>

#define CUDA_CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0
#define BATCH_SIZE 1

static const int INPUT_H   = 112;
static const int INPUT_W   = 112;
static const int emb_size   = 512;

const char* INPUT_BLOB_NAME   = "input";
const char* OUTPUT_BLOB_NAME = "output";

const std::string landmark_file = "/opt/project/tmp/result/landmark/";
const char* facecsv_file = "/opt/project/config/face.csv";
const std::string face_file = "/opt/project/tmp/result/face/";
const std::string facesrc_file = "/opt/project/tmp/face_picture/";

static Logger gLogger;
using namespace nvinfer1;

// 预处理
//hwc -> chw
float* process_image(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (int c = 0; c < channels; c++) 
    {
        for (int  h = 0; h < img_h; h++) 
        {
            for (int w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] = (((float)img.at<cv::Vec3b>(h, w)[c] / 255.)- 0.5)/0.5;
            }
        }
    }
    return blob;
}

// 推理
void doInference(IExecutionContext& context, float* input, float* output1, const int output_size1, const int batch_size) {
    const ICudaEngine& engine = context.getEngine();

    // // 获取索引编号的数量
    // std::cout << engine.getNbBindings() << std::endl; 
    // assert(engine.getNbBindings() == 3);

    // 获取输入输出的名称、以及输入输出的索引编号
    void* buffers[2];
    const int inputIndex   = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batch_size * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], output_size1*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Inference DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //context.enqueue(batch_size, buffers, stream, nullptr);
    Dims4 inputDims{batch_size, 3, INPUT_H, INPUT_W};
    context.setBindingDimensions(0, inputDims);
    
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output1, buffers[outputIndex], output_size1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    std::string engine_name = "/opt/project/engine/arcface/arcface_b256_dynamic_fp16.engine";

   // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // set buffer
    auto out_dims = engine->getBindingDimensions(1);
    assert(out_dims.d[1] == emb_size);
    auto output_size = BATCH_SIZE * emb_size;
    static float* emb = new float[output_size];

    //write face_cvs l2_norm tensor
    std::ofstream face_cvs;
    face_cvs.open(facecsv_file, std::ios::app);
    if(!face_cvs)
    {
        std::cout << "csv file open faile!!" << std::endl;
        exit(0);
    }

    std::cout << "batchsize run inference......." << std::endl;
    auto start = std::chrono::system_clock::now();
    
    std::string name = argv[1];  
    std::string filename;
    std::string facename;
    float norm_sum = 0.0;
    facename = name.substr(0,name.size() - 4);
    face_cvs << facename;
    
    cv::Mat img;
    try{
        img = cv::imread(landmark_file + name);
    }
    catch(cv::Exception &e)
    {   
        std::cout << "arcface 图片读取失败！" << std::endl;
        return -1;
    }
    float* blob = process_image(img);
    doInference(*context, blob, emb, output_size, BATCH_SIZE); 
    delete blob;
    // 删除临时face人脸文件
    remove((landmark_file+name).c_str());
    remove((face_file+name).c_str());
    remove((facesrc_file+name).c_str());
    //norm
    for(int i = 0; i < emb_size; i++)
    {
        norm_sum += emb[i] * emb[i];
    }
    for(int i = 0; i < emb_size; i++)
    {
        face_cvs << " " << emb[i];
    }
    face_cvs << "\n";
    
    face_cvs.close();
    
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // destroy the engine
    delete emb;
   
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}