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
#include <json/json.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "get_post.h"

struct Device_Config_Param
{
    //device param
    int enable;
    std::string ip;
    std::string http_trans;
    int http_port;
    int rtsp_port;
    int udp_port;
    std::string width;
    std::string height;
    int model_type;
};

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
#define MAX_BATCH_SIZE 20

static const int INPUT_H   = 112;
static const int INPUT_W   = 112;
static const int emb_size   = 512;

const char* INPUT_BLOB_NAME   = "input";
const char* OUTPUT_BLOB_NAME = "output";
const char* facecsv_file = "/opt/project/config/face.csv";
const std::string landmark_file = "/opt/project/tmp/result/landmark/";
const std::string face_file = "/opt/project/tmp/result/face/";
const std::string facesrc_file = "/opt/project/tmp/face_picture/";
const char* http_json = "/opt/project/config/Algorithm_face.json";

static Logger gLogger;
using namespace nvinfer1;

static inline void stringsplit(std::string str, const char split, std::vector<std::string>& rstr)
{
    std::istringstream iss(str);
    std::string token;
    while(getline(iss, token, split))
    {
        rstr.push_back(token);
    }
}

void cos_sim_save(std::vector<std::string> &csv_face_name, std::vector<float> &src_embeddedFace, 
                    float* output_emb, std::vector<std::string> &cur_face_name, bool &env_happen, std::stringstream &out_string)
{
   
   for(int i = 0; i < cur_face_name.size(); i++)
   {
        float max = 0.0;
        std::string save_name;
        std::string cur_facename = cur_face_name[i];
        for(int j = 0; j < csv_face_name.size(); j++)
        {
            std::string csv_facename = csv_face_name[j];
            float num = 0.0;
            float csv_sum_l2 = 0.0;
            float cur_sum_l2 = 0.0;
            float sim = 0.0;
            float denom = 0.0;
            for(int k = 0; k < emb_size; k++)
            {
                num += src_embeddedFace[j * emb_size + k] * output_emb[i * emb_size + k];
                csv_sum_l2 += src_embeddedFace[j * emb_size + k] * src_embeddedFace[j * emb_size + k];
                cur_sum_l2 += output_emb[i * emb_size + k] * output_emb[i * emb_size + k];
            }
            denom = std::sqrt(csv_sum_l2) * std::sqrt(cur_sum_l2);
            sim = 0.5 + 0.5 * (num / denom);
            if(sim > max)
            {
                max = sim;
                save_name = csv_facename;
            }
        }
        if(max > 0.77)
        {   
            env_happen = true;
            out_string << ",{\"code\":\"39\",";
            out_string << "\"status\":\"200\",";
            out_string <<  "\"info\":\""<< save_name << "\"";
            out_string << "}";
        }
   }
}

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

void mat2stream(std::vector<cv::Mat>& imgMats, float* stream) {
    int curBatchSize = imgMats.size();
    if (curBatchSize > MAX_BATCH_SIZE) {
        std::cout << "[WARNING]::Batch size overflow, input will be truncated!" << std::endl;
        curBatchSize = MAX_BATCH_SIZE;
    }
    for (int batch = 0; batch < curBatchSize; ++batch) {
        cv::Mat tempMat = imgMats[batch];
        float *img_blob = process_image(tempMat);
        for(int idx = 0 ; idx < (3*INPUT_H*INPUT_W); idx ++)
        {
            stream[batch*3*INPUT_H*INPUT_W + idx] = img_blob[idx];
        }
        delete img_blob;
    }
}

int main(int argc, char** argv) {
    bool env_happen = false;
    std::stringstream out_string;
    std::ofstream error_log;
    bool device_field = false;
    error_log.open("/opt/project/log/json_parse_error_log.txt", std::ios::out);
    if(!error_log)
    {
        std::cout << "error_log file open faile!" << std::endl;
        return 0;
    }
    std::ifstream ifs;
    ifs.open(http_json);
    if(!ifs)
    {
        return 0;
    }
    Json::Value jsonRoot;
    Json::Reader jsonReader;
    if(!jsonReader.parse(ifs, jsonRoot))
    {   
        error_log << "http接口返回字符jsonReader解析失败\n";
        error_log.close();
		ifs.close();
        return 0;
    }
	std::map<std::string, std::map<std::string, std::string>> temp;
    for(auto it = jsonRoot.begin(); it!=jsonRoot.end(); it++)
    {   
        std::map<std::string, std::string> stu_temp;
        for(auto sit = it->begin(); sit!= it->end(); sit++)
        {   
            stu_temp.insert(std::pair<std::string, std::string>(sit.name(), (*sit).asString()));
        }
        temp.insert({it.name(), stu_temp});
    }
	Device_Config_Param device_param;
	for(auto item : temp)
	{
		if(item.first == "device")
        {   
            device_field = true;
			device_param.http_trans = temp["device"]["http-trans"];
            device_param.http_port = std::stoi(temp["device"]["http-port"]);
        } 
	}
	if(!device_field)
    {
        error_log << "json配置文件中未查找到机器设备device字段属性\n";
        error_log.close();
		ifs.close();
        return 0;
    }
    error_log.close();
    ifs.close();
    cudaSetDevice(DEVICE);
    std::string engine_name = "/opt/project/engine/arcface/arcface_b256_dynamic_fp16.engine";
    
    //read face_cvs l2_norm tensor
    std::vector<float> src_embeddedFace;
    std::vector<std::string> facecsv_name;
    std::ifstream face_cvs;
    face_cvs.open(facecsv_file, std::ios::in);
    std::string line;
    while(std::getline(face_cvs, line))
    {
        std::istringstream face_tensor_info(line);
        std::string face_name;
        if(!(face_tensor_info >> face_name))
        {
            std::cout << "read face csv faile!!!!" << std::endl;
            exit(0);
        }
        facecsv_name.push_back(face_name);

        float face_tensor[emb_size];
        int i = 0;
        float face_tensor_item;
        while((face_tensor_info >> face_tensor_item))
        {
            face_tensor[i] = face_tensor_item;
            i++;
        }
        src_embeddedFace.insert(src_embeddedFace.end(), face_tensor, face_tensor+emb_size);
    }
    face_cvs.close();

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

    std::cout << "batchsize run inference......." << std::endl;
    auto start = std::chrono::system_clock::now();
    std::vector<cv::Mat> img_mat;
    std::vector<std::string> cur_facename;
    std::string name = argv[1];   
    cv::Mat img;
    try{
        img = cv::imread(landmark_file + name);
    }
    catch(cv::Exception &e)
    {   
        std::cout << "arcface 图片读取失败！" << std::endl;
        return -1;
    }
    //resize
    img_mat.push_back(img);
    cur_facename.push_back(name);
    
    auto out_dims = engine->getBindingDimensions(1);
    assert(out_dims.d[1] == emb_size);
    auto output_size = img_mat.size() * emb_size;
    static float* emb = new float[output_size];
    float *blob = new float[img_mat.size() * 3 * INPUT_H * INPUT_W];
    mat2stream(img_mat, blob);
    doInference(*context, blob, emb, output_size, img_mat.size()); 
    cos_sim_save(facecsv_name, src_embeddedFace, emb, cur_facename, env_happen, out_string);
    if(env_happen)
    {   
        out_string << "]";
        std::string http_string = out_string.str();
        http_string[0] = '[';
        std::cout <<"env: " << http_string << std::endl;
        char warning_http[512];
        sprintf(warning_http,WARNING_OUT_HTTP2,device_param.http_trans.c_str(),device_param.http_port);
        http_post(warning_http, http_string.c_str());
        return 0;
    }
    else
    {
        out_string << "[{\"code\":\"39\",";
        out_string << "\"status\":\"404\",";
        out_string <<  "\"info\":\"face recognition failed\"";
        out_string << "}]";
        std::string http_string = out_string.str();
        std::cout <<"env: " << http_string << std::endl;
        char warning_http[512];
        sprintf(warning_http,WARNING_OUT_HTTP2,device_param.http_trans.c_str(),device_param.http_port);
        http_post(warning_http, http_string.c_str());
        return -1;
    }
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // destroy the engine
    delete blob;
    remove((landmark_file+name).c_str());
    remove((face_file+name).c_str());
    remove((facesrc_file+name).c_str());
    delete emb;
   

    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}