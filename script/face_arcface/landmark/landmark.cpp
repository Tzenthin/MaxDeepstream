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
// #define DEBUG
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
static const int emb_size   = 134;
static const int aglin_5_point[5] = {65, 66, 36, 53, 59};

const char* INPUT_BLOB_NAME   = "input";
const char* OUTPUT_BLOB_NAME = "output";

const std::string input_file = "/opt/project/tmp/result/face/";
const std::string output_file = "/opt/project/tmp/result/landmark/";

static Logger gLogger;
using namespace nvinfer1;

// 预处理
//hwc -> chw
float* process_image(cv::Mat& img){
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
                blob[c * img_w * img_h + h * img_w + w] = ((float)img.at<cv::Vec3b>(h, w)[c] / 255.0 - 0.5)/0.5;
            }
        }
    }
    return blob;
}

static inline std::string get_date_time()
{
	auto to_string = [](const std::chrono::system_clock::time_point& t)->std::string
	{
		auto as_time_t = std::chrono::system_clock::to_time_t(t);
		struct tm tm;
		localtime_r(&as_time_t, &tm);//linux api，线程安全

		std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
		char buf[128];
		// snprintf(buf, sizeof(buf), "%04d-%02d-%02d-%02d:%02d:%02d-%03lld ",
		// 	tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ms.count() % 1000);
        snprintf(buf, sizeof(buf), "%04d-%02d-%02d-%02d-%02d-%02d-%03lld",
			tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ms.count() % 1000);
		return buf;
	};

	std::chrono::system_clock::time_point t = std::chrono::system_clock::now();
	return to_string(t);
}


cv::Mat FindNonReflectiveTransform(std::vector<cv::Point2d> source_points,
                                                    std::vector<cv::Point2d> target_points,
                                                    cv::Mat& Tinv)
{
    assert(source_points.size() == target_points.size());
    assert(source_points.size() >= 2);
    cv::Mat U = cv::Mat::zeros(target_points.size() * 2, 1, CV_64F);
    cv::Mat X = cv::Mat::zeros(source_points.size() * 2, 4, CV_64F);
    for (int i = 0; i < target_points.size(); i++)
    {
            U.at<double>(i * 2, 0) = source_points[i].x;
            U.at<double>(i * 2 + 1, 0) = source_points[i].y;
            X.at<double>(i * 2, 0) = target_points[i].x;
            X.at<double>(i * 2, 1) = target_points[i].y;
            X.at<double>(i * 2, 2) = 1;
            X.at<double>(i * 2, 3) = 0;
            X.at<double>(i * 2 + 1, 0) = target_points[i].y;
            X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
            X.at<double>(i * 2 + 1, 2) = 0;
            X.at<double>(i * 2 + 1, 3) = 1;
    }
    cv::Mat r = X.inv(cv::DECOMP_SVD)*U;
    Tinv = (cv::Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
                        r.at<double>(1), r.at<double>(0), 0,
                        r.at<double>(2), r.at<double>(3), 1);
    cv::Mat T = Tinv.inv(cv::DECOMP_SVD);
    Tinv = Tinv(cv::Rect(0, 0, 2, 3)).t();
    return T(cv::Rect(0,0,2,3)).t();
}

cv::Mat FindSimilarityTransform(std::vector<cv::Point2d> source_points,
                                                    std::vector<cv::Point2d> target_points,
                                                    cv::Mat& Tinv)
{
    cv::Mat Tinv1, Tinv2;
    cv::Mat trans1 = FindNonReflectiveTransform(source_points, target_points, Tinv1);
    std::vector<cv::Point2d> source_point_reflect;
    for (auto sp : source_points)
    //for(int i = 0; i < source_points.size(); i++)
    {
        //cv::Point2d sp = source_points.at(i);
        source_point_reflect.push_back(cv::Point2d(-sp.x, sp.y));
    }
    cv::Mat trans2 = FindNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
    trans2.colRange(0,1) *= -1;
    std::vector<cv::Point2d> trans_points1, trans_points2;
    cv::transform(source_points, trans_points1, trans1);
    cv::transform(source_points, trans_points2, trans2);
    double norm1 = norm(cv::Mat(trans_points1), cv::Mat(target_points), cv::NORM_L2);
    double norm2 = norm(cv::Mat(trans_points2), cv::Mat(target_points), cv::NORM_L2);
    Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
    return norm1 < norm2 ? trans1 : trans2;
}

bool Alig112X112(cv::Mat &Image, std::vector<cv::Point2d> &points, cv::Mat &CropFaceMat) {

    std::vector<cv::Point2d> target_points = {{30.2946 + 8.0, 51.6963},
                                              {65.5318 + 8.0, 51.5014},
                                              {48.0252 + 8.0, 71.7366},
                                              {33.5493 + 8.0, 92.3655},
                                              {62.7299 + 8.0, 92.2041}};


    cv::Mat trans_inv;
    cv::Mat trans = FindSimilarityTransform(points, target_points, trans_inv);
    cv::Mat temp;
    cv::warpAffine(Image, CropFaceMat, trans, cv::Size(112, 112), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return true;
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

void stringsplit(std::string str, const char split, std::vector<std::string>& rstr)
{
    std::istringstream iss(str);
    std::string token;
    while(getline(iss, token, split))
    {
        rstr.push_back(token);
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    std::string engine_name = "/opt/project/engine/arcface/landmark_b256_dynamic_fp16.engine";

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

    auto out_dims = engine->getBindingDimensions(1);
    assert(out_dims.d[1] == emb_size);
    auto output_size = BATCH_SIZE * emb_size;
    static float* emb = new float[output_size];

    std::cout << "[landmark] batchsize run inference......." << std::endl;
    auto start = std::chrono::system_clock::now();
    std::string result_name = argv[1];
    cv::Mat img;
    try{
        img = cv::imread(input_file + result_name);
    }
    catch(cv::Exception &e)
    {   
        std::cout << "landmark 图片读取失败！" << std::endl;
        return -1;
    }
    //resize
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H)); 
    float* blob = process_image(img);
    doInference(*context, blob, emb, output_size, BATCH_SIZE);
    std::vector<cv::Point2d> pts_real;
    for(int j = 0; j < 5; j++)
    {   
        int point_idx = aglin_5_point[j];
        int x = emb[point_idx * 2] * 112;
        int y = emb[point_idx * 2 + 1] * 112;
        pts_real.push_back(cv::Point2d(x , y));
    }
    assert(pts_real.size() == 5);

    cv::Mat croppedMat;
    Alig112X112(img, pts_real, croppedMat);

    std::string faceimg_name = output_file + result_name;
    cv::imwrite(faceimg_name, croppedMat);
    delete blob;
       
    
    delete emb;

    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}