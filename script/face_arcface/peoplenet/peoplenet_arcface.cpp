//#include <iostream>
//#include <chrono>
//#include <cmath>
//#include "cuda_utils.h"
//#include "logging.h"
//#include "common.hpp"
//#include "utils.h"
//#include "calibrator.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <json/json.h>
#include "logging.h"
#include "get_post.h"
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
#define NMS_THRESH 0.3
#define BBOX_CONF_THRESH 0.3
#define BATCH_SIZE 1

static const int CLASS_NUM = 3;
static const int INPUT_H   = 544;
static const int INPUT_W   = 960;

static const int box_norm = 35.0;
static const int stride = 16;
static const int frame_interval = 1;
static const int num_grid_y = INPUT_H / stride;
static const int num_grid_x = INPUT_W / stride;
const char* INPUT_BLOB_NAME   = "input_1";
const char* OUTPUT_BLOB_NAME1 = "output_bbox/BiasAdd";
const char* OUTPUT_BLOB_NAME2 = "output_cov/Sigmoid";

const std::string input_file = "/opt/project/tmp/face_picture/";
const std::string output_file = "/opt/project/tmp/result/face/";
const char* http_json = "/opt/project/config/Algorithm_face.json";

static Logger gLogger;
using namespace nvinfer1;

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

// 预处理
cv::Mat process_image(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    // cv::Mat re(INPUT_H, INPUT_W, CV_8UC3);
    // //resize
    // cv::resize(img, re, re.size());
    //bgr2rgb
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    return out;
}
//hwc -> chw
float* blob_image(cv::Mat& img){
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
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c]/255.0;
            }
        }
    }
    return blob;
}



// 后处理
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void generate_grids_and_stride(const int stride, std::vector<float>& grid_centers_w, std::vector<float>& grid_centers_h)
{
    for(int g1 = 0; g1 < num_grid_y; g1++)
    {   
        grid_centers_h.push_back((g1 * stride + 0.5) / box_norm);
    }
    
    for (int g0 = 0; g0 < num_grid_x; g0++)
    {
        grid_centers_w.push_back((g0 * stride + 0.5) / box_norm);
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void nms_big_bboxes(const std::vector<Object>& faceobjects, std::vector<cv::Rect>& tmpContours, float nms_threshold, float *scale)
{
    
    const int n = faceobjects.size();
    std::vector<int> iou_test(n,0); 
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }
    for (int i = 0; i < n; i++)
    {   
        if(iou_test[i])
            continue;
	    std::vector<cv::Point> contours;
        const Object& a = faceobjects[i];
        for(int j = i+1; j < n; j++)
        {
            const Object& b = faceobjects[j];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
            {
                iou_test[j] = 1;
                contours.push_back(cv::Point2i(int(a.rect.x), int(a.rect.y)));
                contours.push_back(cv::Point2i(int(a.rect.x + a.rect.width), int(a.rect.y + a.rect.height)));
                contours.push_back(cv::Point2i(int(b.rect.x), int(b.rect.y)));
                contours.push_back(cv::Point2i(int(b.rect.x + b.rect.width), int(b.rect.y + b.rect.height)));
            }
        }
        if(contours.size() > 0)
        {   
            cv::Rect rect = cv::boundingRect(contours);
            rect.x /= scale[0];
            rect.y /= scale[1];
            rect.width /= scale[0];
            rect.height /= scale[1];
            tmpContours.push_back(rect);  
        }

    
    }
}

static void generate_yolox_proposals(std::vector<float>& grid_centers_w, std::vector<float>& grid_centers_h, 
                                            float* dec_rect, float* dec_prob, 
                                            float prob_threshold, std::vector<Object>& objects)
{

    for (int class_idx = 0; class_idx < CLASS_NUM; class_idx++)
    {
        int x1_idx = class_idx * 4 * num_grid_y * num_grid_x;
        int y1_idx = x1_idx + num_grid_y * num_grid_x;
        int x2_idx = y1_idx + num_grid_y * num_grid_x;
        int y2_idx = x2_idx + num_grid_y * num_grid_x;
        for(int h = 0; h < num_grid_y; h++)
            for(int w = 0; w < num_grid_x; w++)
            {
                int idx = w + h * num_grid_x;
                if(dec_prob[class_idx * num_grid_y * num_grid_x + idx] >= prob_threshold)
                {
                    float x1_pre = dec_rect[x1_idx + idx];
                    float y1_pre = dec_rect[y1_idx + idx];
                    float x2_pre = dec_rect[x2_idx + idx];
                    float y2_pre = dec_rect[y2_idx + idx];
                    x1_pre = (x1_pre - grid_centers_w[w]) * -box_norm;
                    y1_pre = (y1_pre - grid_centers_h[h]) * -box_norm;
                    x2_pre = (x2_pre + grid_centers_w[w]) * box_norm;
                    y2_pre = (y2_pre + grid_centers_h[h]) * box_norm;
                    Object obj;
                    if(class_idx == 2)
                    {
                        obj.rect.x = x1_pre;
                        obj.rect.y = y1_pre;
                        obj.rect.width = x2_pre - x1_pre;
                        obj.rect.height = y2_pre - y1_pre;
                        obj.label = class_idx;
                        obj.prob = dec_prob[class_idx * num_grid_y * num_grid_x + idx];
                        objects.push_back(obj);
                    }
                }
            }
    }
}

int decode_outputs(std::vector<float> &grid_centers_w, std::vector<float> &grid_centers_h, 
                            float* dec_rect, float* dec_prob, std::vector<Object>& objects,
                            float *scale, const int img_w, const int img_h, std::string facename,
                            bool &env_happen, std::stringstream &out_string) {
    std::vector<Object> proposals;
    generate_yolox_proposals(grid_centers_w, grid_centers_h, dec_rect, dec_prob, BBOX_CONF_THRESH, proposals);
    std::cout << "num of boxes before nms: " << proposals.size() << std::endl;
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    int count = picked.size();
    std::cout << "num of boxes: " << count << std::endl;
    if(count == 0 || count > 5)
    {   
        env_happen = true;
        out_string << ",{\"code\":\"39\",";
        out_string << "\"status\":\"404\",";
        out_string <<  "\"info\":\"no face or more than 5 faces\"";
        out_string << "}";
        return 0;
    }
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale[0];
        float y0 = (objects[i].rect.y) / scale[1];
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale[0];
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale[1];

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    return 1;
}

static void draw_objects(const cv::Mat& bgr, std::vector<Object>& objects, std::string wn)
{
    static const char* class_names[] = { "person", "bag", "face" };

    const float color_list[3][3] =
    {
        {0.000, 0.447, 0.741},
        {0.850, 0.325, 0.098},
        {0.929, 0.694, 0.125}
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        Object& obj = objects[i];

        fprintf(stderr, "%s %s = %.5f at %.2f %.2f %.2f x %.2f\n\n", wn.c_str(), class_names[obj.label], obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        float maxlen = std::max(obj.rect.width, obj.rect.height);
        float central_x = obj.rect.x + obj.rect.width/2;
        float central_y = obj.rect.y + obj.rect.height/2;
        obj.rect.x = ((central_x - maxlen/2) >=0 ? (central_x - maxlen/2) : 0);
        obj.rect.y = ((central_y - maxlen/2) >=0 ? (central_y - maxlen/2) : 0);
        obj.rect.width = (obj.rect.x + maxlen <= image.cols ? maxlen : (image.cols - obj.rect.x));
        obj.rect.height = (obj.rect.y + maxlen <= image.rows ? maxlen : (image.rows - obj.rect.y));
        if((obj.rect.width * obj.rect.height ) < 35*35)
            continue;
        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        // cv::rectangle(image, obj.rect, color * 255, 2);
        cv:: Mat face_img = image(obj.rect).clone();
        // cv::imwrite(output_file + std::to_string(i) + "_" + wn , face_img);
        cv::imwrite(output_file + wn , face_img);
    }
    // cv:: Mat face_img = image(obj.rect).clone();
    // cv::imwrite(wn, face_img);
    //fprintf(stderr, "save vis file\n");
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

// 推理
void doInference(IExecutionContext& context, float* input, float* output1, const int output_size1, float* output2, const int output_size2, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // // 获取索引编号的数量
    // std::cout << engine.getNbBindings() << std::endl; 
    // assert(engine.getNbBindings() == 3);

    // 获取输入输出的名称、以及输入输出的索引编号
    void* buffers[3];
    const int inputIndex   = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);
    const int outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
    // std::cout << inputIndex   << "  " << engine.getBindingName(0) << std::endl;
    // std::cout << outputIndex1 << "  " << engine.getBindingName(1) << std::endl;
    // std::cout << outputIndex2 << "  " << engine.getBindingName(2) << std::endl;

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex1], output_size1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex2], output_size2*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Inference DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], output_size1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output2, buffers[outputIndex2], output_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex1]));
    CUDA_CHECK(cudaFree(buffers[outputIndex2]));
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
    std::string engine_name = "/opt/project/engine/arcface/resnet34_peoplenet_int8.etlt_b1_gpu0_int8.engine";
    std::cout << " ----------------[face-det]---------------- " << std::endl;
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

    // 获取输出空间大小
    auto out_dims1 = engine->getBindingDimensions(1);
    auto output_size1 = 1;
    for(int j=0;j<out_dims1.nbDims;j++) {
        output_size1 *= out_dims1.d[j];
    }
    static float* det_rect = new float[output_size1];

    // 获取输出空间大小
    auto out_dims2 = engine->getBindingDimensions(2);
    auto output_size2 = 1;
    for(int j=0;j<out_dims2.nbDims;j++) {
        output_size2 *= out_dims2.d[j];
    }
    static float* det_prob = new float[output_size2];

    //网络后处理参数初始化
    std::vector<float> grid_centers_w, grid_centers_h;
    generate_grids_and_stride(stride, grid_centers_w, grid_centers_h);

    //读取图片，推理
    std::string result_name = argv[1];
    cv::Mat img;
    try{
        img = cv::imread(input_file + result_name);
    }
    catch(cv::Exception &e)
    {   
        out_string << ",{\"code\":\"39\",";
        out_string << "\"status\":\"404\",";
        out_string <<  "\"info\":\"picture read failed\"";
        out_string << "}";
        out_string << "]";
        std::string http_string = out_string.str();
        http_string[0] = '[';
        std::cout <<"env: " << http_string << std::endl;
        char warning_http[512];
        sprintf(warning_http,WARNING_OUT_HTTP2,device_param.http_trans.c_str(),device_param.http_port);
        http_post(warning_http, http_string.c_str());
        return -1;
    }
    int img_w = img.cols;
    int img_h = img.rows;
    std::cout << img_w << " " <<  img_h << std::endl;
    if(img_w * img_h < 1600)
    {   
        out_string << ",{\"code\":\"39\",";
        out_string << "\"status\":\"404\",";
        out_string <<  "\"info\":\"picture size is too small\"";
        out_string << "}";
        out_string << "]";
        std::string http_string = out_string.str();
        http_string[0] = '[';
        std::cout <<"env: " << http_string << std::endl;
        char warning_http[512];
        sprintf(warning_http,WARNING_OUT_HTTP2,device_param.http_trans.c_str(),device_param.http_port);
        http_post(warning_http, http_string.c_str());
        return -1;
    }
    cv::Mat pr_img = process_image(img);  // resize
    float* blob = blob_image(pr_img);  // 数据展开成一维
    
    float scale[] = {INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0)};

    // run inference
    doInference(*context, blob, det_rect, output_size1, det_prob, output_size2, pr_img.size());

    std::vector<Object> objects;
    int state = decode_outputs(grid_centers_w, grid_centers_h, det_rect, det_prob, objects, scale, img_w, img_h, result_name, env_happen, out_string);
    if(state)
    {
        draw_objects(img, objects, result_name);
    }
    // destroy the engine
    delete blob;
        
    if(env_happen)
    {   
        out_string << "]";
        std::string http_string = out_string.str();
        http_string[0] = '[';
        std::cout <<"env: " << http_string << std::endl;
        char warning_http[512];
        sprintf(warning_http,WARNING_OUT_HTTP2,device_param.http_trans.c_str(),device_param.http_port);
        http_post(warning_http, http_string.c_str());
        return -1;
    }

    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
