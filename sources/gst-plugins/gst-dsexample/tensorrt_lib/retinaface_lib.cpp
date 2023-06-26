#include <iostream>
#include <chrono>
#include <vector>
#include "cuda_runtime_api.h"
#include <NvInfer.h>
#include "logging.h"
#include "retinaface_lib.h"
#include "cuda_utils.h"
#include "utils.h"

using namespace nvinfer1;

const char* RETINAFACE_INPUT_BLOB_NAME = "data";
const char* RETINAFACE_OUTPUT_BLOB_NAME = "output";

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

static inline void GenerateAnchors(cv::Mat &refer_matrix)
{
    int line = 0;
    for(size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) {
        for (int height = 0; height < feature_maps[feature_map][0]; ++height) {
            for (int width = 0; width < feature_maps[feature_map][1]; ++width) {
                for (int anchor = 0; anchor < (int)anchor_sizes[feature_map].size(); ++anchor) {
                    auto *row = refer_matrix.ptr<float>(line);
                    row[0] = base_cx + (float)width * feature_steps[feature_map];
                    row[1] = base_cy + (float)height * feature_steps[feature_map];
                    row[2] = anchor_sizes[feature_map][anchor];
                    line++;
                }
            }
        }
    }
}
static inline void generate_retinaface_proposals(cv::Mat &refer_matrix, float* dec_rect, float prob_threshold, std::vector<Object>& objects)
{
    int index = 0;
    int result_cols = 2 + BBOX_HEAD + LANDMARK_HEAD;
    cv::Mat result_matrix = cv::Mat(SUM_OF_FEATURE, result_cols, CV_32FC1, dec_rect);
    for (int item = 0; item < result_matrix.rows; ++item) {
        auto *current_row = result_matrix.ptr<float>(item);
        if (current_row[0] > prob_threshold)
        {
            Object obj;
            obj.label = 0;
            obj.prob = current_row[0];
            auto *anchor = refer_matrix.ptr<float>(item);
            auto *bbox = current_row + 1;
            auto *keyp = current_row + 2 + BBOX_HEAD;
            obj.rect.width = anchor[2] * exp(bbox[2]);
            obj.rect.height = anchor[2] * exp(bbox[3]);
            obj.rect.x = (anchor[0] + bbox[0] * anchor[2]) - obj.rect.width / 2;
            obj.rect.y = (anchor[1] + bbox[1] * anchor[2]) - obj.rect.height / 2;
            for (int i = 0; i < LANDMARK_HEAD / 2; i++) {
                cv::Point2d point;
                point.x = int((anchor[0] + keyp[2 * i] * anchor[2]));
                point.y = int((anchor[1] + keyp[2 * i + 1] * anchor[2]));
                obj.ldmark_pts.push_back(point);
            }
            objects.push_back(obj);
        }
    }
  
}

void retinaface_decode_outputs(cv::Mat &img, cv::Mat &refer_matrix, float* dec_out, float conf_thresh, int source_index,
            std::vector<DetectBox>& det, std::vector<cv::Mat> &align_imgs){
    std::vector<Object> proposals;
    generate_retinaface_proposals(refer_matrix, dec_out, conf_thresh, proposals);
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, RETINAFACE_NMS_THRESH);
    int count = picked.size();
    for (int i = 0; i < count; i++)
    {
        // adjust offset to original unpadded
        float x0 = proposals[picked[i]].rect.x;
        float y0 = proposals[picked[i]].rect.y;
        float x1 = proposals[picked[i]].rect.x + proposals[picked[i]].rect.width;
        float y1 = proposals[picked[i]].rect.y + proposals[picked[i]].rect.height;

        // clip
        x0 = std::max(std::min(x0, (float)(RETINAFACE_INPUT_W - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(RETINAFACE_INPUT_H - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(RETINAFACE_INPUT_W - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(RETINAFACE_INPUT_H - 1)), 0.f);
        
        
        //align
        float height = y1-y0;
        float width = x1-x0;
        cv::Point2f c_point = cv::Point2f(int(x0 + width/2), int(y0 + height/2));
        float max_length = std::max(height, width);
        cv::Rect rect = cv::Rect(int(c_point.x - max_length/2), int(c_point.y - max_length/2),
                                int(max_length), int(max_length));
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y)); 
        cv::Mat tempMat = img(rect).clone();
        DetectBox dd(x0, y0, x1, y1, proposals[picked[i]].prob, proposals[picked[i]].label, source_index);
        dd.landmark.clear();
        std::vector<cv::Point2d> pts_real;
        for(int j = 0; j < 5; j++)
        {   
            int x = (proposals[picked[i]].ldmark_pts[j].x - rect.x);
            int y = (proposals[picked[i]].ldmark_pts[j].y - rect.y);
            x = std::max(std::min(x, RETINAFACE_INPUT_W - 1), 0);
            y = std::max(std::min(y, RETINAFACE_INPUT_H - 1), 0);
            pts_real.push_back(cv::Point2d(x , y));
        }
        assert(pts_real.size() == 5);
        cv::Mat croppedMat;
        Alig112X112(tempMat, pts_real, croppedMat);
        align_imgs.push_back(croppedMat);
        det.push_back(dd);
        // cv::imwrite(frame_name + std::to_string(source_index) + "/" + std::to_string(frame_num) + "_" +std::to_string(i) + ".jpg", croppedMat);
    }
    
}

static void detectnet_doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers,
                        float* output, int output_size) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void * retinaface_trt_create(const char * engine_name, int batch_size)
{   
    size_t size = 0;
    char *trtModelStream = NULL;
    RetinaFaceTRTContext * trt_ctx = NULL;
 
    trt_ctx = new RetinaFaceTRTContext();
 
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
    trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(RETINAFACE_INPUT_BLOB_NAME);
    trt_ctx->outputIndex = trt_ctx->engine->getBindingIndex(RETINAFACE_OUTPUT_BLOB_NAME);
    assert(trt_ctx->inputIndex == 0);
    assert(trt_ctx->outputIndex == 1);
   
    auto out_dims1 = trt_ctx->engine->getBindingDimensions(1);
    auto output_size1 = 1;
    for(int j=0;j<out_dims1.nbDims;j++) {
        output_size1 *= out_dims1.d[j];
    }
    trt_ctx->OUTPUT_SIZE = output_size1;
    trt_ctx->output  = new float[trt_ctx->OUTPUT_SIZE];
    
    trt_ctx->refer_matrix = cv::Mat(SUM_OF_FEATURE, BBOX_HEAD, CV_32FC1);
    GenerateAnchors(trt_ctx->refer_matrix);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], batch_size * 3 * RETINAFACE_INPUT_H * RETINAFACE_INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex],output_size1 * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
    return (void *)trt_ctx;
}
 
int retinaface_trt_detect(void *h)
{   
    // std::cout << "retinaface_landmark" << std::endl;
    RetinaFaceTRTContext *trt_ctx;
    trt_ctx = (RetinaFaceTRTContext *)h;
    // Run inference
    detectnet_doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, trt_ctx->output, 
                        trt_ctx->OUTPUT_SIZE);
    return 1;
}

 
void retinaface_trt_destroy(void *h)
{
    RetinaFaceTRTContext *trt_ctx;
 
    trt_ctx = (RetinaFaceTRTContext *)h;
    
    // Release stream and buffers
    cudaStreamDestroy(trt_ctx->cuda_stream);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]));
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->outputIndex]));
    
    // Destroy the engine
    trt_ctx->exe_context->destroy();
    trt_ctx->engine->destroy();
    trt_ctx->runtime->destroy();
 
    // delete trt_ctx->input;
    delete trt_ctx->output;
 
    delete trt_ctx;
}


