#pragma once 
#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <chrono>
#include "logging.h"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 15

static Logger gLogger;

static const float DETECTNET_NMS_THRESH = 0.3;
static const float DETECTNET_CONF_THRESH = 0.5;
static const int DETECTNET_BOX_NORM = 35.0;
static const int DETECTNET_STRIDE = 16;
static const int DETECTNET_INPUT_H = 544;
static const int DETECTNET_INPUT_W = 960;
static const int DETECTNET_CLASS_NUM = 3;
static const int num_grid_y = DETECTNET_INPUT_H / DETECTNET_STRIDE;
static const int num_grid_x = DETECTNET_INPUT_W / DETECTNET_STRIDE;

static const float RETINAFACE_NMS_THRESH = 0.45;
static const float RETINAFACE_CONF_THRESH = 0.8;
static const float base_cx = 7.5;
static const float base_cy = 7.5;

const std::vector<std::vector<int>> feature_maps{{17, 30}, {34,60}, {68,120}};
const std::vector<int> feature_steps{32, 16, 8};
const std::vector<std::vector<int>> anchor_sizes{{512,256}, {128,64}, {32,16}};
static const int RETINAFACE_INPUT_H = 540;
static const int RETINAFACE_INPUT_W = 960;
static const int SUM_OF_FEATURE = 21420;
static const int BBOX_HEAD = 3;
static const int LANDMARK_HEAD = 10;

static const float INTERVAL_THRESH = 150;
static const float MATCH_THRESH = 3;
static const float ARCFACE_CONF_THRESH = 0.66;
static const int LANDMARK_ARCFACE_INPUT_H = 112;
static const int LANDMARK_ARCFACE_INPUT_W = 112;
static const int LANDMARK_ARCFACE_MAX_BATCH_SIZE = 128;
static const int LANDMARK_EMB_SIZE   = 134;
static const int ALIGN_5_POINT[5] = {65, 66, 36, 53, 59};
static const int ARCFACE_EMB_SIZE   = 512;


static const int INTERVAL_FRAME = 200;

typedef enum
{   
    PERSON = 0,
    BAG,
    FACE,
}DETECTNETV2_LABELS;

static const char* peoplenetlabels[] = {
    "person", "bag", "face"
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2d> ldmark_pts;
};

struct ArcfaceObj
{
    cv::Mat img;
    std::string facename;
    float prob;
};

typedef struct DetectBox {
    DetectBox(float x1=0, float y1=0, float x2=0, float y2=0, 
            float confidence=0, float classID=-1, int sourceID=-1,float trackID=-1, std::string ClassName = "", float Inner=-1, std::vector<int> LandMark={0}) {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->confidence = confidence;
        this->classID = classID;
        this->sourceID = sourceID;
        this->trackID = trackID;
        this->classname = ClassName;
        this->inner = Inner;
        this->landmark = LandMark;
    }
    float x1, y1, x2, y2;
    float confidence;
    float classID;
    int sourceID;
    float trackID;
    std::string classname;
    float inner;
    std::vector<int> landmark;
} DetectBox;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};


static const float color_list[80][3] = {
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

static inline int preprocess_offset(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    return y;
}

static inline cv::Mat detectnet_preprocess_img(cv::Mat& img, int input_w, int input_h) {
     cv::Mat re(input_h, input_w, CV_8UC3);
    //resize
    cv::resize(img, re, re.size());
    //bgr2rgb
    cv::cvtColor(re, re, cv::COLOR_BGR2RGB);
    return re;
}

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
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

static inline std::string get_video_date_time()
{   
	auto to_string = [](const std::chrono::system_clock::time_point& t)->std::string
	{
		auto as_time_t = std::chrono::system_clock::to_time_t(t);
		struct tm tm;
		localtime_r(&as_time_t, &tm);//linux api，线程安全

		std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
		char buf[128];
        snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d ",
			tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
		return buf;
	};

	std::chrono::system_clock::time_point t = std::chrono::system_clock::now();
	return to_string(t);
}

static inline void stringsplit(std::string str, const char split, std::vector<std::string>& rstr)
{
    std::istringstream iss(str);
    std::string token;
    while(getline(iss, token, split))
    {
        rstr.push_back(token);
    }
}

static inline bool env_http_out(std::map<std::string, int> &env_out, std::string key, int cur_frame_num)
{
    int dur_frame_num;
    if(env_out.count(key))
    {
        dur_frame_num = cur_frame_num - env_out[key];
        if(dur_frame_num > INTERVAL_FRAME)
        {   
            env_out[key] = cur_frame_num;
            return true;
        }
    }
    else
    {
        env_out[key] = cur_frame_num;
        return true;
    }
    return false;

}
#endif  // TRTX_YOLOV5_UTILS_H_

