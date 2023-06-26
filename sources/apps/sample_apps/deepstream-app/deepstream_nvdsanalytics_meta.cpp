/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include "gstnvdsmeta.h"
#include "deepstream_app.h"
#include "nvds_analytics_meta.h"
#include "get_post.h"
#include <algorithm>
#include <math.h>
#include <assert.h>

#define MAX_DISPLAY_LEN 64
#define FIRST_DETECTOR_UID 1
#define SECONDARY_DETECTOR_UID 2
#define SECONDARY_CLASSIFIER_UID 3
#define SGIE_CLASS_ID_LPD 0
#define NO_MASK_LABLE 0
#define HEAD 0

static gint64 interval_frame = 200;
static int disappear_thr = 2;
/* iou
float hove_per_rect_left = std::get<0>(roi_temp[std::string(obj_meta->object_id)]);
float hove_per_rect_top = std::get<1>(roi_temp[std::string(obj_meta->object_id)]);
float hove_per_rect_right = std::get<2>(roi_temp[std::string(obj_meta->object_id)]);
float hove_per_rect_bottom = std::get<3>(roi_temp[std::string(obj_meta->object_id)]);
float cur_rect_left = obj_meta->rect_params.left;
float cur_rect_top = obj_meta->rect_params.top;
float cur_rect_right = obj_meta->rect_params.left + obj_meta->rect_params.width;
float cur_rect_bottom = obj_meta->rect_params.top + obj_meta->rect_params.height;
float hove_area = (hove_per_rect_right - hove_per_rect_left + 1) * (hove_per_rect_bottom - hove_per_rect_top + 1);
float cur_area = (obj_meta->rect_params.width + 1) * (obj_meta->rect_params.height + 1);
float xx1 = std::max(hove_per_rect_left, cur_rect_left);
float yy1 = std::max(hove_per_rect_top, cur_rect_top);
float xx2 = std::min(hove_per_rect_right, cur_rect_right);
float yy2 = std::min(hove_per_rect_bottom, cur_rect_bottom);
float w = std::max(0.0f, xx2 - xx1 + 1.0f);
float h = std::max(0.0f, yy2 - yy1 + 1.0f);
float inter = w * h;
float overlap_percent = inter / (hove_area + cur_area - inter);
if(overlap_percent > IOU_THR)*/

/* parse_nvdsanalytics_meta_data
 * and extract nvanalytics metadata etc. */
extern "C" {
    bool intersection_analytics(std::vector<int>sets_per, std::vector<int>sets_cur)
    {
        int length = sets_cur.size()/2;
        std::vector<int> intersection_v;
        std::sort(sets_per.begin(), sets_per.end());
        std::sort(sets_cur.begin(), sets_cur.end());
        set_intersection(sets_per.begin(), sets_per.end(), sets_cur.begin(), sets_cur.end(), back_inserter(intersection_v));
        if(intersection_v.size() > length )
            return true;
        else
            return false;

    }
    
    void hover_analytics(std::map<std::string, std::map<std::string, std::vector<int>>> &temp,
                            std::map<std::string, std::vector<int>> cur_tmp,
                            std::string roi_key)
    {
        std::map<std::string, std::vector<int>> &roi_tmp = temp[roi_key];
        if (roi_tmp.size())
        {
            std::map<std::string, std::vector<int>>::iterator roi_it;
            std::map<std::string, std::vector<int>>::iterator cur_it;
            for (roi_it = roi_tmp.begin(); roi_it != roi_tmp.end(); roi_it++)
            {
                if (cur_tmp.count(roi_it->first))
                {
                    roi_it->second[0] += 1;
                    roi_it->second[1] = 0;
                }
                else
                    roi_it->second[1] += 1;
            }
            for (cur_it = cur_tmp.begin(); cur_it != cur_tmp.end(); cur_it++)
            {
                if (!roi_tmp.count(cur_it->first))
                    roi_tmp.insert({ cur_it->first, cur_it->second });

            }

        }
        else
            roi_tmp = cur_tmp;

    }   



    void running_analytics(std::map<std::string, std::map<std::string, std::tuple<float, float, float, float, int, int>>> &temp,
                            std::map<std::string, std::tuple<float, float, float, float, int, int>> cur_tmp,
                            std::string roi_key, 
                            int run_behind_thr,
                            int run_front_thr,
                            int run_boundary,
                            int dim)
    {
        std::map<std::string, std::tuple<float, float, float, float, int, int>> &roi_tmp = temp[roi_key];
        if (roi_tmp.size())
        {
            std::map<std::string, std::tuple<float, float, float, float, int, int>>::iterator roi_it;
            std::map<std::string, std::tuple<float, float, float, float, int, int>>::iterator cur_it;
            for (roi_it = roi_tmp.begin(); roi_it != roi_tmp.end(); roi_it++)
            {
                if (cur_tmp.count(roi_it->first))
                {   
                    int run_frame_thr = 0;
                    float top_left_x_roi = std::get<0>(roi_it->second);
                    float top_left_y_roi = std::get<1>(roi_it->second);
                    float top_left_x_cur = std::get<0>(cur_tmp[roi_it->first]);
                    float top_left_y_cur = std::get<1>(cur_tmp[roi_it->first]);

                    float top_right_x_roi = std::get<0>(roi_it->second) + std::get<2>(roi_it->second);
                    float top_right_y_roi = top_left_y_roi;
                    float top_right_x_cur = std::get<0>(cur_tmp[roi_it->first]) + std::get<2>(cur_tmp[roi_it->first]);
                    float top_right_y_cur = top_left_y_cur;

                    float bottom_right_x_roi = std::get<0>(roi_it->second) + std::get<2>(roi_it->second);
                    float bottom_right_y_roi = std::get<1>(roi_it->second) + std::get<3>(roi_it->second);
                    float bottom_right_x_cur = std::get<0>(cur_tmp[roi_it->first]) + std::get<2>(cur_tmp[roi_it->first]);
                    float bottom_right_y_cur = std::get<1>(cur_tmp[roi_it->first]) + std::get<3>(cur_tmp[roi_it->first]);

                    float bottom_left_x_roi = std::get<0>(roi_it->second);
                    float bottom_left_y_roi = std::get<1>(roi_it->second) + std::get<3>(roi_it->second);
                    float bottom_left_x_cur = std::get<0>(cur_tmp[roi_it->first]);
                    float bottom_left_y_cur = std::get<1>(cur_tmp[roi_it->first]) + std::get<3>(cur_tmp[roi_it->first]);
                    

                    
                    float dis_top_left = sqrt((top_left_x_roi - top_left_x_cur) * (top_left_x_roi - top_left_x_cur) +
                                            (top_left_y_roi - top_left_y_cur) * (top_left_y_roi - top_left_y_cur));
                    float dis_top_right = sqrt((top_right_x_roi - top_right_x_cur) * (top_right_x_roi - top_right_x_cur) +
                                            (top_right_y_roi - top_right_y_cur) * (top_right_y_roi - top_right_y_cur));
                    float dis_bottom_left = sqrt((bottom_left_x_roi - bottom_left_x_cur) * (bottom_left_x_roi - bottom_left_x_cur) +
                                            (bottom_left_y_roi - bottom_left_y_cur) * (bottom_left_y_roi - bottom_left_y_cur));
                    float dis_bottom_right = sqrt((bottom_right_x_roi - bottom_right_x_cur) * (bottom_right_x_roi - bottom_right_x_cur) +
                                            (bottom_right_y_roi - bottom_right_y_cur) * (bottom_right_y_roi - bottom_right_y_cur));

                    if(dim == 0)
                    {
                        if(bottom_right_y_cur <= run_boundary)
                        {
                            if(dis_top_left > run_behind_thr &&  dis_top_left < (3*run_behind_thr))
                                run_frame_thr+=1;
                            if(dis_top_right > run_behind_thr && dis_top_right < (3*run_behind_thr))
                                run_frame_thr+=1;
                            if(dis_bottom_left > run_behind_thr && dis_bottom_left < (3*run_behind_thr))
                                run_frame_thr+=1;
                            if(dis_bottom_right > run_behind_thr && dis_bottom_right < (3*run_behind_thr))
                                run_frame_thr+=1;
                        }
                        else{
                            if(dis_top_left > run_front_thr)
                                run_frame_thr+=1;
                            if(dis_top_right > run_front_thr)
                                run_frame_thr+=1;
                            if(dis_bottom_left > run_front_thr)
                                run_frame_thr+=1;
                            if(dis_bottom_right > run_front_thr)
                                run_frame_thr+=1;
                        }
                    }
                    if(dim == 1)
                    {
                        if(bottom_right_x_cur <= run_boundary)
                        {
                            if(dis_top_left > run_behind_thr &&  dis_top_left < (3*run_behind_thr))
                                run_frame_thr+=1;
                            if(dis_top_right > run_behind_thr && dis_top_right < (3*run_behind_thr))
                                run_frame_thr+=1;
                            if(dis_bottom_left > run_behind_thr && dis_bottom_left < (3*run_behind_thr))
                                run_frame_thr+=1;
                            if(dis_bottom_right > run_behind_thr && dis_bottom_right < (3*run_behind_thr))
                                run_frame_thr+=1;
                        }
                        else{
                            if(dis_top_left > run_front_thr)
                                run_frame_thr+=1;
                            if(dis_top_right > run_front_thr)
                                run_frame_thr+=1;
                            if(dis_bottom_left > run_front_thr)
                                run_frame_thr+=1;
                            if(dis_bottom_right > run_front_thr)
                                run_frame_thr+=1;
                        }
                    }
                    std::get<0>(roi_it->second) = std::get<0>(cur_tmp[roi_it->first]);
                    std::get<1>(roi_it->second) = std::get<1>(cur_tmp[roi_it->first]);
                    std::get<2>(roi_it->second) = std::get<2>(cur_tmp[roi_it->first]);
                    std::get<3>(roi_it->second) = std::get<3>(cur_tmp[roi_it->first]);
                    if(run_frame_thr > 2){
                        std::get<4>(roi_it->second) += 1;
                        std::get<5>(roi_it->second) = 0;
                    }
                    else{
                        std::get<4>(roi_it->second) -= 1;
                        if(std::get<4>(roi_it->second) < 0)
                            std::get<4>(roi_it->second) = 0;
                        std::get<5>(roi_it->second) = 0;
                    }
                }
                else{
                    std::get<4>(roi_it->second) -= 1;
                    if(std::get<4>(roi_it->second) < 0)
                        std::get<4>(roi_it->second) = 0;
                    std::get<5>(roi_it->second) += 1;
                }
                
            }
            for (cur_it = cur_tmp.begin(); cur_it != cur_tmp.end(); cur_it++)
            {
                if (!roi_tmp.count(cur_it->first))
                    roi_tmp.insert({ cur_it->first, cur_it->second });

            }

        }
        else
            roi_tmp = cur_tmp;

    } 
    

    void lpd_analytics(std::map<std::string, std::tuple<float, float, float, float>> &temp,
                        float left, float top, float width,float height, std::map<std::string, std::vector<int>> &stu_roi_env)
    {   
        std::map<std::string, std::tuple<float, float, float, float>>::iterator roi_it;
        for (roi_it = temp.begin(); roi_it != temp.end(); roi_it++)
        {   
            float lpd_left_loc = std::get<0>(roi_it->second);
            float lpd_top_loc = std::get<1>(roi_it->second);
            float lpd_width = std::get<2>(roi_it->second);
            float lpd_height = std::get<3>(roi_it->second);
            if(left < lpd_left_loc && top < lpd_top_loc && width > lpd_width && height > lpd_height)
            {
                stu_roi_env.insert({roi_it->first, std::vector<int>{1, 0} });
            }
        }
    }

    void smoking_analytics(std::vector<int> &temp, int obj_id, std::map<std::string, std::vector<int>> &stu_roi_env)
    {
        for(auto item : temp){
            if(obj_id == item)
            {
                stu_roi_env.insert({std::to_string(obj_id), std::vector<int>{1, 0} });
            }
        }
    }

    bool env_http_out(std::map<std::string, gint64> &env_out, std::string roi_key, gint64 cur_frame_num)
    {
        gint64 dur_frame_num;
        if(env_out.count(roi_key))
        {
            dur_frame_num = cur_frame_num - env_out[roi_key];
            if(dur_frame_num > interval_frame)
            {   
                env_out[roi_key] = cur_frame_num;
                return true;
            }
        }
        else
        {
            env_out[roi_key] = cur_frame_num;
            return true;
        }
        return false;

    }

    bool lpr_env_http_out(std::map<std::string, gint64> &env_out, gint64 cur_frame_num)
    {
        gint64 dur_frame_num;
        std::map<std::string, gint64>::iterator env_it;
        
        for(env_it = env_out.begin(); env_it != env_out.end();)
        {
            std::map<std::string, gint64>::iterator it_back = env_it;
            bool is_first_element = false;
            if(it_back != env_out.begin())
                it_back--;
            else
                is_first_element = true;
            dur_frame_num = cur_frame_num - env_it->second;
            if(dur_frame_num > interval_frame)
            {
                env_out.erase(env_it);
                if(is_first_element)
                    env_it = env_out.begin();
                else
                    env_it = ++it_back;
            }
            else
                env_it++;      
        } 
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

    void parse_nvdsanalytics_meta_data (AppCtx * appCtx, NvDsBatchMeta *batch_meta)
    {   
        static std::map<std::string, std::vector<int>>  parent_env;
        static std::map<std::string, std::map<std::string, std::vector<int>>> hover_env;
        static std::map<std::string, std::map<std::string, std::tuple<float, float, float, float, int, int>>> running_env;
        static std::map<std::string, std::vector<std::string>> abnormal_id;
        static std::map<int, std::string, std::greater<int>> car_lrp;
        static std::map<std::string, gint64> env_out;
        static std::map<std::string, gint64> lpr_env_out;
        bool post_http_info = false;
        bool debug = false;
        pid_t status;
        NvDsObjectMeta *obj_meta = NULL;
        NvDsMetaList * l_frame = NULL;
        NvDsMetaList * l_obj = NULL;
        NvDsMetaList * l_class = NULL;
        NvDsDisplayMeta *display_meta = NULL;
        NvDsClassifierMeta * class_meta = NULL;
        NvDsMetaList * l_label = NULL;
        NvDsLabelInfo * label_info = NULL;
        guint32 stream_id = 0;
        guint label_i = 0;
        std::stringstream out_string, time_string;
        gint64 frame_num;
        struct tm *time_p;
        time_t now;
        now = time(NULL);
        time_p = localtime(&now);
        //NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
        
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
                l_frame = l_frame->next) {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
            stream_id = frame_meta->source_id;
            frame_num = frame_meta->frame_num;
            /* Iterate user metadata in frames to search analytics metadata*/
            for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
                    l_user != NULL; l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
                if (user_meta->base_meta.meta_type != NVDS_USER_FRAME_META_NVDSANALYTICS)
                    continue;
                /* convert to  metadata */
                NvDsAnalyticsFrameMeta *meta =
                    (NvDsAnalyticsFrameMeta *) user_meta->user_meta_data;
                /* Get the labels from nvdsanalytics config file */
                switch(appCtx->config.multi_source_config[stream_id].env_type){
                    case NV_DS_TARGET_CROWDED:
                        for (std::pair<std::string, bool> status : meta->ocStatus){
                            if(status.second){
                                std::vector<int> stu_env;
                                //abnormal analysis
                                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                    // Access attached user meta for each object
                                    for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                            l_user_meta = l_user_meta->next) {
                                        NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);
                                        if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                        {   
                                            NvDsAnalyticsObjInfo * user_meta_data =
                                                (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                            if (user_meta_data->ocStatus.size()){
                                                for(std::string obj_cls: user_meta_data->ocStatus)
                                                {
                                                    if(obj_cls == status.first)
                                                    {
                                                        // out_string << " object id = " << obj_meta->object_id;
                                                        stu_env.push_back(obj_meta->object_id);

                                                    }
                                                }

                                            }
                                        }
                                    }
                                }
                                if(parent_env.count(std::to_string(stream_id) + status.first))
                                {
                                    if(intersection_analytics(parent_env[std::to_string(stream_id) + status.first], stu_env))
                                        post_http_info |= false;
                                    else{
                                        if(env_http_out(env_out, std::to_string(stream_id) + status.first, frame_num))
                                        {
                                            out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                            out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                            out_string << "\"sourceNum\":\"\",";
                                            out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                            out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                            out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                            out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                            out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                            out_string << "\"filePath\":\"\",";
                                            out_string << "\"carNumber\":\"\",";
                                            out_string << "\"faceId\":\"\",";
                                            out_string << "\"imgPath\":\"\",";
                                            out_string <<  "\"remarks\":\"" << " 请注意！画面中" << status.first << "标注区域处于拥挤" << "\"";
                                            out_string << "}";
                                            post_http_info |= true;
                                        }
                                        parent_env[std::to_string(stream_id) + status.first] = stu_env;
                                    }
                                }
                                else
                                {       
                                    if(env_http_out(env_out, std::to_string(stream_id) + status.first, frame_num))
                                    {
                                        out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                        out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                        out_string << "\"sourceNum\":\"\",";
                                        out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                        out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                        out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                        out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                        out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                        out_string << "\"filePath\":\"\",";
                                        out_string << "\"carNumber\":\"\",";
                                        out_string << "\"faceId\":\"\",";
                                        out_string << "\"imgPath\":\"\",";
                                        out_string <<  "\"remarks\":\"" << " 请注意！画面中" << status.first << "标注区域处于拥挤" << "\"";
                                        out_string << "}";
                                        post_http_info |= true;
                                    }
                                    parent_env[std::to_string(stream_id) + status.first] = stu_env;
                                }
                                break;
                            }
                        }
                        break;

                    case NV_DS_TARGET_HOVER:
                        int hover_thr;
                        hover_thr = appCtx->config.multi_source_config[stream_id].hover_interval;
                        if(!hover_thr)
                            break;
                        for (std::pair<std::string, uint32_t> status : meta->objInROIcnt)
                        {   
                            if(status.second)
                            {   
                                std::map<std::string, std::vector<int>> stu_roi_env;
                                //abnormal analysis
                                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                    // Access attached user meta for each object
                                    for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                            l_user_meta = l_user_meta->next) {
                                        NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);
                                        if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                        {   
                                            NvDsAnalyticsObjInfo * user_meta_data =
                                                (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                            if (user_meta_data->roiStatus.size()){
                                                for(std::string obj_cls: user_meta_data->roiStatus)
                                                {
                                                    if(obj_cls == status.first)
                                                    {   
                                                        stu_roi_env.insert({std::to_string(obj_meta->object_id) + obj_meta->obj_label, std::vector<int>{1, 0} });
                                                    }
                                                }

                                            }
                                        }
                                    }
                                }
                                if(hover_env.count(std::to_string(stream_id) + status.first))
                                {  
                                    hover_analytics(hover_env, stu_roi_env, std::to_string(stream_id) + status.first);
                                }
                                else
                                {
                                    hover_env.insert({std::to_string(stream_id) + status.first, stu_roi_env});
                                }

                            }
                            else
                            {
                                if(hover_env.count(std::to_string(stream_id) + status.first)){
                                    std::map<std::string, std::vector<int>>::iterator roi_it;
                                    for(roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); 
                                        roi_it != hover_env[std::to_string(stream_id) + status.first].end(); roi_it++)
                                    {
                                        roi_it->second[1] += 1;
                                    }
                                }
                                
                            }
                            //analysis hover
                            if(hover_env.count(std::to_string(stream_id) + status.first))    
                            {   
                                bool hover_happen = false;
                                std::map<std::string, std::vector<int>>::iterator roi_it;
                                for (roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); roi_it != hover_env[std::to_string(stream_id) + status.first].end();)
                                {
                                    std::map<std::string, std::vector<int>>::iterator it_back = roi_it;
                                    bool is_first_element = false;
                                    if (it_back != hover_env[std::to_string(stream_id) + status.first].begin())
                                        it_back--;
                                    else
                                        is_first_element = true;
                                    
                                    if (roi_it->second[0] >= hover_thr)
                                    {
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) == abnormal_id[std::to_string(stream_id) + status.first].end())
                                            abnormal_id[std::to_string(stream_id) + status.first].push_back(roi_it->first);
                                        hover_happen |= true;
                                        roi_it->second[0] = 0;
                                        roi_it->second[1] = 0;
                                    }
                                
                                    if (roi_it->second[1] > disappear_thr)
                                    {   
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) != abnormal_id[std::to_string(stream_id) + status.first].end())
                                        {
                                            int dis = std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                                abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) - abnormal_id[std::to_string(stream_id) + status.first].begin();
                                            abnormal_id[std::to_string(stream_id) + status.first].erase(abnormal_id[std::to_string(stream_id) + status.first].begin() + dis);
                                        }
                                        hover_env[std::to_string(stream_id) + status.first].erase(roi_it);
                                        if (is_first_element)
                                            roi_it = hover_env[std::to_string(stream_id) + status.first].begin();
                                        else
                                            roi_it = ++it_back;
                                    }
                                    else
                                        roi_it++;

                                }
                                if(hover_happen)
                                {
                                    if(env_http_out(env_out, std::to_string(stream_id) + status.first, frame_num))
                                    {   
                                        out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                        out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                        out_string << "\"sourceNum\":\"\",";
                                        out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                        out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                        out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                        out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                        out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                        out_string << "\"filePath\":\"\",";
                                        out_string << "\"carNumber\":\"\",";
                                        out_string << "\"faceId\":\"\",";
                                        out_string << "\"imgPath\":\"\",";
                                        out_string <<  "\"remarks\":\"" << "请注意！有疑似目标异常徘徊或占用超过" << hover_thr/25 << "秒" << "\"";
                                        out_string << "}";
                                        post_http_info |= true;
                                    }
                                }

                            }
                        }
                        break;

                    case NV_DS_PERSON_DIR:
                        int dir_thr;
                        dir_thr = appCtx->config.multi_source_config[stream_id].hover_interval;
                        if(!dir_thr)
                            break;
                        for (std::pair<std::string, uint32_t> status : meta->objInROIcnt)
                        {
                            if(status.second)
                            { 
                                std::map<std::string, std::vector<int>> stu_roi_env;
                                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                    // Access attached user meta for each object
                                    for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                            l_user_meta = l_user_meta->next) {
                                        NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);
                                        if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                        {
                                            NvDsAnalyticsObjInfo * user_meta_data =
                                                (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                            if (user_meta_data->roiStatus.size()){
                                                for(std::string obj_cls: user_meta_data->roiStatus)
                                                {
                                                    if(obj_cls == status.first)
                                                    {   
                                                       if (user_meta_data->dirStatus.length()){
                                                            //out_string << " object " << obj_meta->object_id << " is moving in " <<  user_meta_data->dirStatus;
                                                            stu_roi_env.insert({std::to_string(obj_meta->object_id) + obj_meta->obj_label, std::vector<int>{1, 0} });
                                                        }
                                                    }
                                                }

                                            }
                                        }
                                    }
                                }
                                if(hover_env.count(std::to_string(stream_id) + status.first))
                                {  
                                    hover_analytics(hover_env, stu_roi_env, std::to_string(stream_id) + status.first);
                                }
                                else
                                {
                                    hover_env.insert({std::to_string(stream_id) + status.first, stu_roi_env});
                                }
                            }
                            else
                            {
                                if(hover_env.count(std::to_string(stream_id) + status.first)){
                                    std::map<std::string, std::vector<int>>::iterator roi_it;
                                    for(roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); 
                                        roi_it != hover_env[std::to_string(stream_id) + status.first].end(); roi_it++)
                                    {
                                        roi_it->second[1] += 1;
                                    }
                                }
                                
                            }
                            
                            //analysis hover
                            if(hover_env.count(std::to_string(stream_id) + status.first))    
                            {   
                                bool hover_happen = false;
                                std::map<std::string, std::vector<int>>::iterator roi_it;
                                for (roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); roi_it != hover_env[std::to_string(stream_id) + status.first].end();)
                                {
                                    std::map<std::string, std::vector<int>>::iterator it_back = roi_it;
                                    bool is_first_element = false;
                                    if (it_back != hover_env[std::to_string(stream_id) + status.first].begin())
                                        it_back--;
                                    else
                                        is_first_element = true;
                                    
                                    if (roi_it->second[0] >= dir_thr)
                                    {
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) == abnormal_id[std::to_string(stream_id) + status.first].end())
                                            abnormal_id[std::to_string(stream_id) + status.first].push_back(roi_it->first);
                                        hover_happen |= true;
                                        roi_it->second[0] = 0;
                                        roi_it->second[1] = 0;
                                    }
                                
                                    if (roi_it->second[1] > disappear_thr)
                                    {   
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) != abnormal_id[std::to_string(stream_id) + status.first].end())
                                        {
                                            int dis = std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                                abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) - abnormal_id[std::to_string(stream_id) + status.first].begin();
                                            abnormal_id[std::to_string(stream_id) + status.first].erase(abnormal_id[std::to_string(stream_id) + status.first].begin() + dis);
                                        }
                                        hover_env[std::to_string(stream_id) + status.first].erase(roi_it);
                                        if (is_first_element)
                                            roi_it = hover_env[std::to_string(stream_id) + status.first].begin();
                                        else
                                            roi_it = ++it_back;
                                    }
                                    else
                                        roi_it++;

                                }
                                if(hover_happen)
                                {   
                                    if(env_http_out(env_out, std::to_string(stream_id) + status.first, frame_num))
                                    {
                                        out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                        out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                        out_string << "\"sourceNum\":\"\",";
                                        out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                        out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                        out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                        out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                        out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                        out_string << "\"filePath\":\"\",";
                                        out_string << "\"carNumber\":\"\",";
                                        out_string << "\"faceId\":\"\",";
                                        out_string << "\"imgPath\":\"\",";
                                        out_string <<  "\"remarks\":\"" << "请注意！有疑似目标在" + status.first + "区域攀爬或逆行 " << "\"";
                                        out_string << "}";
                                        post_http_info |= true;
                                    }
                                    break;
                                }

                            }
                        }
                        break;
                        
                    case NV_DS_PERSON_RUNNING:
                        int run_behind_thr;
                        run_behind_thr = appCtx->config.multi_source_config[stream_id].run_params.roi_behind;
                        int run_front_thr; 
                        run_front_thr = appCtx->config.multi_source_config[stream_id].run_params.roi_front;
                        int run_interval_thr;
                        run_interval_thr = appCtx->config.multi_source_config[stream_id].run_params.run_interval_thr;
                        int run_boundary; 
                        run_boundary = appCtx->config.multi_source_config[stream_id].run_params.boundary;
                        int dim;
                        dim = appCtx->config.multi_source_config[stream_id].run_params.dim;
                        if(!run_behind_thr)
                            break;
                        for (std::pair<std::string, uint32_t> status : meta->objInROIcnt)
                        {   
                            if(status.second)
                            {   
                                std::map<std::string, std::tuple<float, float, float, float, int, int>> stu_roi_env;
                                //abnormal analysis
                                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                    // Access attached user meta for each object
                                    for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                            l_user_meta = l_user_meta->next) {
                                        NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);
                                        if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                        {   
                                            NvDsAnalyticsObjInfo * user_meta_data =
                                                (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                            if (user_meta_data->roiStatus.size()){
                                                for(std::string obj_cls: user_meta_data->roiStatus)
                                                {
                                                    if(obj_cls == status.first)
                                                    {   
                                                        // std::cout<< "<" << obj_meta->rect_params.left << " " << obj_meta->rect_params.top << " " << obj_meta->rect_params.width<< " " <<
                                                        //     obj_meta->rect_params.height << ">" ;
                                                        stu_roi_env.insert({std::to_string(obj_meta->object_id) + obj_meta->obj_label, std::make_tuple(
                                                                                                    obj_meta->rect_params.left,
                                                                                                    obj_meta->rect_params.top,
                                                                                                    obj_meta->rect_params.width,
                                                                                                    obj_meta->rect_params.height,
                                                                                                    1, 0) });
                                                    }
                                                }

                                            }
                                        }
                                    }
                                }
                                if(running_env.count(std::to_string(stream_id) + status.first))
                                {  
                                    running_analytics(running_env, stu_roi_env, std::to_string(stream_id) + status.first, run_behind_thr, run_front_thr, run_boundary, dim);
                                }
                                else
                                {
                                    running_env.insert({std::to_string(stream_id) + status.first, stu_roi_env});
                                }

                            }
                            else
                            {
                                if(running_env.count(std::to_string(stream_id) + status.first))
                                {
                                    std::map<std::string, std::tuple<float, float, float, float, int, int>>::iterator roi_it;
                                    for(roi_it = running_env[std::to_string(stream_id) + status.first].begin(); 
                                        roi_it != running_env[std::to_string(stream_id) + status.first].end(); roi_it++)
                                    {   
                                        std::get<4>(roi_it->second) -= 1;
                                        if(std::get<4>(roi_it->second) < 0)
                                            std::get<4>(roi_it->second) = 0;
                                        std::get<5>(roi_it->second) += 1;
                                    }
                                }
                            }
                        
                        //analysis hover 
                        if(running_env.count(std::to_string(stream_id) + status.first))
                        {   
                            bool hover_happen = false;
                            std::map<std::string, std::tuple<float, float, float, float, int, int>>::iterator roi_it;
                            for (roi_it = running_env[std::to_string(stream_id) + status.first].begin(); 
                                 roi_it != running_env[std::to_string(stream_id) + status.first].end();)
                            {
                                std::map<std::string, std::tuple<float, float, float, float, int, int>>::iterator it_back = roi_it;
                                bool is_first_element = false;
                                if (it_back != running_env[std::to_string(stream_id) + status.first].begin())
                                    it_back--;
                                else
                                    is_first_element = true;

                                
                                if (std::get<4>(roi_it->second) >= run_interval_thr)
                                {
                                    if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                        abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) == abnormal_id[std::to_string(stream_id) + status.first].end())
                                        {
                                            abnormal_id[std::to_string(stream_id) + status.first].push_back(roi_it->first);
                                            hover_happen |= true;
                                        }
                                    // std::get<4>(roi_it->second) = 0;
                                    // std::get<5>(roi_it->second) = 0;
                                }
                                if(std::get<4>(roi_it->second) < run_interval_thr)
                                {
                                    if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                        abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) != abnormal_id[std::to_string(stream_id) + status.first].end())
                                    {
                                        int dis = std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) - abnormal_id[std::to_string(stream_id) + status.first].begin();
                                        abnormal_id[std::to_string(stream_id) + status.first].erase(abnormal_id[std::to_string(stream_id) + status.first].begin() + dis);
                                    }
                                }

                                if (std::get<5>(roi_it->second) >= disappear_thr)
                                {   
                                    if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                        abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) != abnormal_id[std::to_string(stream_id) + status.first].end())
                                    {
                                        int dis = std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) - abnormal_id[std::to_string(stream_id) + status.first].begin();
                                        abnormal_id[std::to_string(stream_id) + status.first].erase(abnormal_id[std::to_string(stream_id) + status.first].begin() + dis);
                                    }
                                    running_env[std::to_string(stream_id) + status.first].erase(roi_it);
                                    if (is_first_element)
                                        roi_it = running_env[std::to_string(stream_id) + status.first].begin();
                                    else
                                        roi_it = ++it_back;
                                }
                                else
                                    roi_it++;

                            }
                            if(hover_happen)
                            {   
                                if(env_http_out(env_out, std::to_string(stream_id) + status.first, frame_num))
                                {
                                    out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                    out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                    out_string << "\"sourceNum\":\"\",";
                                    out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                    out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                    out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                    out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                    out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                    out_string << "\"filePath\":\"\",";
                                    out_string << "\"carNumber\":\"\",";
                                    out_string << "\"faceId\":\"\",";
                                    out_string << "\"imgPath\":\"\",";
                                    out_string <<  "\"remarks\":\"" << "请注意！画面中" + status.first + "标注区域，有疑似目标在奔跑或超速"<< "\"";
                                    out_string << "}";
                                    post_http_info |= true;
                                }
                            }

                        }
                        }
                        break;
                    case NV_DS_CAR_LRP:
                        int same_thr;
                        same_thr = appCtx->config.multi_source_config[stream_id].hover_interval;
                        if(same_thr)
                        {   
                            std::map<std::string, std::vector<int>> stu_roi_env; 
                            std::map<std::string, std::tuple<float, float, float, float>> car_lpd;
                            for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next){
                                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                    if(obj_meta->unique_component_id == FIRST_DETECTOR_UID)
                                    {
                                        for (std::pair<std::string, uint32_t> status : meta->objInROIcnt)
                                        {
                                            if(status.second)
                                            {
                                                 // Access attached user meta for each object
                                                for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                                        l_user_meta = l_user_meta->next) {
                                                    NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);
                                                    if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                                    {   
                                                        NvDsAnalyticsObjInfo * user_meta_data =
                                                            (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                                        if (user_meta_data->roiStatus.size()){
                                                            for(std::string obj_cls: user_meta_data->roiStatus)
                                                            {
                                                                if(obj_cls == status.first)
                                                                {   
                                                                    if(car_lpd.size())
                                                                    {
                                                                        lpd_analytics(car_lpd, obj_meta->rect_params.left, obj_meta->rect_params.top, 
                                                                                                    obj_meta->rect_params.width, obj_meta->rect_params.height, stu_roi_env);
                                                                        
                                                                    }
                                                                
                                                                }
                                                            }

                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if(obj_meta->unique_component_id == SECONDARY_DETECTOR_UID)
                                    {   
                                        for (l_class = obj_meta->classifier_meta_list; l_class != NULL;
                                                l_class = l_class->next) {
                                            class_meta = (NvDsClassifierMeta *)(l_class->data);
                                            if (!class_meta)
                                                continue;
                                            if (class_meta->unique_component_id == SECONDARY_CLASSIFIER_UID) {
                                                for ( label_i = 0, l_label = class_meta->label_info_list;
                                                    label_i < class_meta->num_labels && l_label; label_i++,
                                                    l_label = l_label->next) {
                                                label_info = (NvDsLabelInfo *)(l_label->data);
                                                if (label_info) {
                                                    if (label_info->label_id == 0 && label_info->result_class_id == 1) {
                                                        car_lpd.insert({label_info->result_label, std::make_tuple(
                                                                                                            obj_meta->rect_params.left,
                                                                                                            obj_meta->rect_params.top,
                                                                                                            obj_meta->rect_params.width,
                                                                                                            obj_meta->rect_params.height) });
                                                    }
                                                }
                                            }
                                        }
                                    } 
                                }  
                                            
                            }
                            if(stu_roi_env.size())
                            {
                                if(hover_env.count(std::to_string(stream_id) + "lpr"))
                                {  
                                    hover_analytics(hover_env, stu_roi_env, std::to_string(stream_id) + "lpr");
                                }
                                else
                                {
                                    hover_env.insert({std::to_string(stream_id) + "lpr", stu_roi_env});
                                }
                            }
                            else
                            {
                                if(hover_env.count(std::to_string(stream_id) + "lpr")){
                                    std::map<std::string, std::vector<int>>::iterator roi_it;
                                    for(roi_it = hover_env[std::to_string(stream_id) + "lpr"].begin(); 
                                        roi_it != hover_env[std::to_string(stream_id) + "lpr"].end(); roi_it++)
                                    {
                                        roi_it->second[1] += 1;
                                    }
                                }
                            }
                            lpr_env_http_out(lpr_env_out, frame_num);
                            //analysis hover
                            if(hover_env.count(std::to_string(stream_id) + "lpr"))    
                            {   
                                std::map<std::string, std::vector<int>>::iterator roi_it;
                                for (roi_it = hover_env[std::to_string(stream_id) + "lpr"].begin(); roi_it != hover_env[std::to_string(stream_id) + "lpr"].end();)
                                {
                                    std::map<std::string, std::vector<int>>::iterator it_back = roi_it;
                                    bool is_first_element = false;
                                    if (it_back != hover_env[std::to_string(stream_id) + "lpr"].begin())
                                        it_back--;
                                    else
                                        is_first_element = true;
                                    
                                    if (roi_it->second[0] >= same_thr && (!lpr_env_out.count(roi_it->first)))
                                    {
                                        lpr_env_out.insert({roi_it->first, frame_num});
                                        out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                        out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                        out_string << "\"sourceNum\":\"\",";
                                        out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                        out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                        out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                        out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                        out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                        out_string << "\"filePath\":\"\",";
                                        out_string << "\"carNumber\":\"" << roi_it->first << "\",";
                                        out_string << "\"faceId\":\"\",";
                                        out_string << "\"imgPath\":\"\",";
                                        out_string <<  "\"remarks\":\"" << "请注意，有车辆进出" << "\"";
                                        out_string << "}";
                                        post_http_info |= true;
                                        
                                        roi_it->second[0] = 0;
                                        roi_it->second[1] = 0;
                                    }
                                
                                    if (roi_it->second[1] > disappear_thr)
                                    {   
                                        hover_env[std::to_string(stream_id) + "lpr"].erase(roi_it);
                                        if (is_first_element)
                                            roi_it = hover_env[std::to_string(stream_id) + "lpr"].begin();
                                        else
                                            roi_it = ++it_back;
                                    }
                                    else
                                        roi_it++;

                                }  
                            }
                        }
                        break;
                    case NV_DS_FACE_MASK:
                        int mask_same_thr;
                        mask_same_thr = appCtx->config.multi_source_config[stream_id].hover_interval;
                        if(!mask_same_thr)
                            break;
                        for (std::pair<std::string, uint32_t> status : meta->objInROIcnt)
                        {
                            if(status.second)
                            {   
                                std::map<std::string, std::vector<int>> stu_roi_env;
                                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next){
                                obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                        l_user_meta = l_user_meta->next) {
                                    NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);
                                    if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                    {
                                        NvDsAnalyticsObjInfo * user_meta_data =
                                            (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                            if (user_meta_data->roiStatus.size()){
                                                for(std::string obj_cls: user_meta_data->roiStatus)
                                                {   
                                                    if(obj_cls == status.first)
                                                    {
                                                        for (l_class = obj_meta->classifier_meta_list; l_class != NULL;
                                                            l_class = l_class->next) {
                                                            class_meta = (NvDsClassifierMeta *)(l_class->data);
                                                            if (!class_meta)
                                                                continue;
                                                            //std::cout << "frame [" << frame_num << "] " << class_meta->unique_component_id << std::endl;
                                                            if (class_meta->unique_component_id == SECONDARY_CLASSIFIER_UID) {
                                                                for ( label_i = 0, l_label = class_meta->label_info_list;
                                                                    label_i < class_meta->num_labels && l_label; label_i++,
                                                                    l_label = l_label->next) {
                                                                label_info = (NvDsLabelInfo *)(l_label->data);
                                                                if (label_info && label_info->result_class_id == NO_MASK_LABLE) {
                                                                    // std::cout << "frame [" << frame_num << "] " << " id = " << obj_meta->object_id << " label = "<< obj_meta->class_id << std::endl;   
                                                                    stu_roi_env.insert({std::to_string(obj_meta->object_id) + label_info->result_label, std::vector<int>{1, 0} }); 
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if(hover_env.count(std::to_string(stream_id) + status.first))
                            {  
                                hover_analytics(hover_env, stu_roi_env, std::to_string(stream_id) + status.first);
                            }
                            else
                            {
                                hover_env.insert({std::to_string(stream_id) + status.first, stu_roi_env});
                            }
                            }
                            else
                            {
                                if(hover_env.count(std::to_string(stream_id) + status.first)){
                                    std::map<std::string, std::vector<int>>::iterator roi_it;
                                    for(roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); 
                                        roi_it != hover_env[std::to_string(stream_id) + status.first].end(); roi_it++)
                                    {
                                        roi_it->second[1] += 1;
                                    }
                                }
                                
                            }
                            //analysis hover
                            // std::cout << std::endl;
                            if(hover_env.count(std::to_string(stream_id) + status.first))    
                            {   
                                // std::cout << "frame [" << frame_num << "] " << " stream id = " << std::to_string(stream_id) + status.first << " ";
                                bool hover_happen = false;
                                std::map<std::string, std::vector<int>>::iterator roi_it;
                                for (roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); roi_it != hover_env[std::to_string(stream_id) + status.first].end();)
                                {
                                    std::map<std::string, std::vector<int>>::iterator it_back = roi_it;
                                    bool is_first_element = false;
                                    if (it_back != hover_env[std::to_string(stream_id) + status.first].begin())
                                        it_back--;
                                    else
                                        is_first_element = true;
                                    // std::cout << " obj id = " << roi_it->first << " status = [ " << roi_it->second[0] << "," <<  roi_it->second[1] << " ]";
                                    // std::cout << "num = " << mask_same_thr << std::endl;
                                    if (roi_it->second[0] >= mask_same_thr)
                                    {
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) == abnormal_id[std::to_string(stream_id) + status.first].end())
                                            abnormal_id[std::to_string(stream_id) + status.first].push_back(roi_it->first);
                                        hover_happen |= true;
                                        roi_it->second[0] = 0;
                                        roi_it->second[1] = 0;
                                    }
                                
                                    if (roi_it->second[1] > disappear_thr)
                                    {   
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) != abnormal_id[std::to_string(stream_id) + status.first].end())
                                        {
                                            int dis = std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                                abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) - abnormal_id[std::to_string(stream_id) + status.first].begin();
                                            abnormal_id[std::to_string(stream_id) + status.first].erase(abnormal_id[std::to_string(stream_id) + status.first].begin() + dis);
                                        }
                                        hover_env[std::to_string(stream_id) + status.first].erase(roi_it);
                                        if (is_first_element)
                                            roi_it = hover_env[std::to_string(stream_id) + status.first].begin();
                                        else
                                            roi_it = ++it_back;
                                    }
                                    else
                                        roi_it++;

                                }
                                if(hover_happen)
                                {
                                    if(env_http_out(env_out, std::to_string(stream_id) + status.first, frame_num))
                                    {   
                                        
                                        out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                        out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                        out_string << "\"sourceNum\":\"\",";
                                        out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                        out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                        out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                        out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                        out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                        out_string << "\"filePath\":\"\",";
                                        out_string << "\"carNumber\":\"\",";
                                        out_string << "\"faceId\":\"\",";
                                        out_string << "\"imgPath\":\"\",";
                                        out_string <<  "\"remarks\":\"" << "请注意！有疑似目标未佩戴口罩" << "\"";
                                        out_string << "}";
                                        post_http_info |= true;
                                    }
                                }

                            }

                        }
                        break;
                    case NV_DS_CHEF_HEAD:
                        int head_hover_thr;
                        head_hover_thr = appCtx->config.multi_source_config[stream_id].hover_interval;
                        if(!head_hover_thr)
                            break;
                        for (std::pair<std::string, uint32_t> status : meta->objInROIcnt)
                        {
                            if(status.second)
                            {
                                std::map<std::string, std::vector<int>> stu_roi_env;
                                //abnormal analysis
                                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                    // Access attached user meta for each object
                                    for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                            l_user_meta = l_user_meta->next) {
                                        NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);  
                                        if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                        {   
                                            NvDsAnalyticsObjInfo * user_meta_data =
                                                (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                            if (user_meta_data->roiStatus.size()){
                                                for(std::string obj_cls: user_meta_data->roiStatus)
                                                {   
                                                    if(obj_cls == status.first && obj_meta->class_id == HEAD)
                                                    {
                                                        stu_roi_env.insert({std::to_string(obj_meta->object_id) + obj_meta->obj_label, std::vector<int>{1, 0} });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if(hover_env.count(std::to_string(stream_id) + status.first))
                                {  
                                    hover_analytics(hover_env, stu_roi_env, std::to_string(stream_id) + status.first);
                                }
                                else
                                {
                                    hover_env.insert({std::to_string(stream_id) + status.first, stu_roi_env});
                                }
                            }
                            else
                            {
                                if(hover_env.count(std::to_string(stream_id) + status.first)){
                                    std::map<std::string, std::vector<int>>::iterator roi_it;
                                    for(roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); 
                                        roi_it != hover_env[std::to_string(stream_id) + status.first].end(); roi_it++)
                                    {
                                        roi_it->second[1] += 1;
                                    }
                                }
                                
                            }
                            //analysis hover
                            if(hover_env.count(std::to_string(stream_id) + status.first))    
                            {   
                                bool hover_happen = false;
                                std::map<std::string, std::vector<int>>::iterator roi_it;
                                for (roi_it = hover_env[std::to_string(stream_id) + status.first].begin(); roi_it != hover_env[std::to_string(stream_id) + status.first].end();)
                                {
                                    std::map<std::string, std::vector<int>>::iterator it_back = roi_it;
                                    bool is_first_element = false;
                                    if (it_back != hover_env[std::to_string(stream_id) + status.first].begin())
                                        it_back--;
                                    else
                                        is_first_element = true;
                                    
                                    if (roi_it->second[0] >= head_hover_thr)
                                    {
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) == abnormal_id[std::to_string(stream_id) + status.first].end())
                                            abnormal_id[std::to_string(stream_id) + status.first].push_back(roi_it->first);
                                        hover_happen |= true;
                                        roi_it->second[0] = 0;
                                        roi_it->second[1] = 0;
                                    }
                                
                                    if (roi_it->second[1] > disappear_thr)
                                    {   
                                        if(std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                            abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) != abnormal_id[std::to_string(stream_id) + status.first].end())
                                        {
                                            int dis = std::find(abnormal_id[std::to_string(stream_id) + status.first].begin(), 
                                                abnormal_id[std::to_string(stream_id) + status.first].end(), roi_it->first) - abnormal_id[std::to_string(stream_id) + status.first].begin();
                                            abnormal_id[std::to_string(stream_id) + status.first].erase(abnormal_id[std::to_string(stream_id) + status.first].begin() + dis);
                                        }
                                        hover_env[std::to_string(stream_id) + status.first].erase(roi_it);
                                        if (is_first_element)
                                            roi_it = hover_env[std::to_string(stream_id) + status.first].begin();
                                        else
                                            roi_it = ++it_back;
                                    }
                                    else
                                        roi_it++;

                                }
                                if(hover_happen)
                                {
                                    if(env_http_out(env_out, std::to_string(stream_id) + status.first, frame_num))
                                    {   
                                        
                                        out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                        out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                        out_string << "\"sourceNum\":\"\",";
                                        out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                        out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                        out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                        out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                        out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                        out_string << "\"filePath\":\"\",";
                                        out_string << "\"carNumber\":\"\",";
                                        out_string << "\"faceId\":\"\",";
                                        out_string << "\"imgPath\":\"\",";
                                        out_string <<  "\"remarks\":\"" << "未戴厨师帽" << "\"";
                                        out_string << "}";
                                        post_http_info |= true;
                                    }
                                }

                            }

                        }
                        break;
                    case NV_DS_SMOKING:
                        int smoke_hover_thr;
                        smoke_hover_thr = appCtx->config.multi_source_config[stream_id].hover_interval;
                        if(smoke_hover_thr)
                        {   
                            std::map<std::string, std::vector<int>> stu_roi_env; 
                            std::vector<int> smoking_hand_id;
                            for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next){
                                obj_meta = (NvDsObjectMeta *) (l_obj->data);
                                if(obj_meta->unique_component_id == SECONDARY_DETECTOR_UID)
                                {
                                    smoking_hand_id.push_back(obj_meta->parent->object_id);
                                    // std::cout << "frame [" << frame_num << "] " << " stream id = " << std::to_string(stream_id) \
                                    // << " id = " << obj_meta->parent->object_id << std::endl;
                                }
                                if(obj_meta->unique_component_id == FIRST_DETECTOR_UID)
                                {
                                    for (std::pair<std::string, uint32_t> status : meta->objInROIcnt)
                                    {
                                        if(status.second)
                                        {
                                            // Access attached user meta for each object
                                            for (NvDsMetaList *l_user_meta = obj_meta->obj_user_meta_list; l_user_meta != NULL;
                                                    l_user_meta = l_user_meta->next) {
                                                NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user_meta->data);
                                                if(user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS)
                                                {   
                                                    NvDsAnalyticsObjInfo * user_meta_data =
                                                        (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
                                                    if (user_meta_data->roiStatus.size()){
                                                        for(std::string obj_cls: user_meta_data->roiStatus)
                                                        {
                                                            if(obj_cls == status.first)
                                                            {   
                                                                if(smoking_hand_id.size())
                                                                {
                                                                    smoking_analytics(smoking_hand_id, obj_meta->object_id, stu_roi_env);
                                                                }
                                                            
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } 
                                }
                            }
                            if(stu_roi_env.size())
                            {
                                if(hover_env.count(std::to_string(stream_id) + "smoking"))
                                {  
                                    hover_analytics(hover_env, stu_roi_env, std::to_string(stream_id) + "smoking");
                                }
                                else
                                {
                                    hover_env.insert({std::to_string(stream_id) + "smoking", stu_roi_env});
                                }
                            }
                            else
                            {
                                if(hover_env.count(std::to_string(stream_id) + "smoking")){
                                    std::map<std::string, std::vector<int>>::iterator roi_it;
                                    for(roi_it = hover_env[std::to_string(stream_id) + "smoking"].begin(); 
                                        roi_it != hover_env[std::to_string(stream_id) + "smoking"].end(); roi_it++)
                                    {
                                        roi_it->second[1] += 1;
                                    }
                                }
                            }
                            //analysis hover
                            if(hover_env.count(std::to_string(stream_id) + "smoking"))    
                            {   
                                bool hover_happen = false;
                                std::map<std::string, std::vector<int>>::iterator roi_it;
                                for (roi_it = hover_env[std::to_string(stream_id) + "smoking"].begin(); roi_it != hover_env[std::to_string(stream_id) + "smoking"].end();)
                                {
                                    std::map<std::string, std::vector<int>>::iterator it_back = roi_it;
                                    bool is_first_element = false;
                                    if (it_back != hover_env[std::to_string(stream_id) + "smoking"].begin())
                                        it_back--;
                                    else
                                        is_first_element = true;
                                    if (roi_it->second[0] >= smoke_hover_thr)
                                    {
                                        hover_happen |= true;
                                        roi_it->second[0] = 0;
                                        roi_it->second[1] = 0;
                                    }
                                    if (roi_it->second[1] > disappear_thr)
                                    {   
                                        hover_env[std::to_string(stream_id) + "smoking"].erase(roi_it);
                                        if (is_first_element)
                                            roi_it = hover_env[std::to_string(stream_id) + "smoking"].begin();
                                        else
                                            roi_it = ++it_back;
                                    }
                                    else
                                        roi_it++;
                                }
                                if(hover_happen)
                                {
                                    if(env_http_out(env_out, std::to_string(stream_id) + "smoking", frame_num))
                                    {   
                                        
                                        out_string << g_strdup_printf (",{\"cameraCode\":\"%d\",",stream_id);
                                        out_string << g_strdup_printf("\"deviceIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].father_ip);
                                        out_string << "\"sourceNum\":\"\",";
                                        out_string << g_strdup_printf ("\"cameraIp\":\"%s\",", appCtx->config.multi_source_config[stream_id].ip);
                                        out_string << g_strdup_printf("\"typeCode1\":\"%d\",", appCtx->config.multi_source_config[stream_id].model_type);
                                        out_string << g_strdup_printf("\"typeName1\":\"%s\",", model_type[appCtx->config.multi_source_config[stream_id].model_type-1]);
                                        out_string << g_strdup_printf("\"typeCode2\":\"%d\",", appCtx->config.multi_source_config[stream_id].env_type);
                                        out_string << g_strdup_printf("\"typeName2\":\"%s\",", envs_type[appCtx->config.multi_source_config[stream_id].env_type-1]);
                                        out_string << "\"filePath\":\"\",";
                                        out_string << "\"carNumber\":\"\",";
                                        out_string << "\"faceId\":\"\",";
                                        out_string << "\"imgPath\":\"\",";
                                        out_string <<  "\"remarks\":\"" << "请注意！有疑似目标在抽烟" << "\"";
                                        out_string << "}";
                                        post_http_info |= true;
                                    }
                                }

                            }
                        }
                        break;
                    default:
                        break;

                }
            
            }

            //draw the abnormal rect item
            int offset = 0;
            display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
            display_meta->num_labels = 1;
            txt_params->display_text = (char*)g_malloc0(2*MAX_DISPLAY_LEN);
            char *wday[] = {"周日", "周一", "周二", "周三", "周四", "周五", "周六"};
            time_string << 1900 + time_p->tm_year << "-" <<  g_strdup_printf ("%02d",1+time_p->tm_mon) << "-" << g_strdup_printf ("%02d",time_p->tm_mday) << " " << wday[time_p->tm_wday] 
             << " " << g_strdup_printf ("%02d",time_p->tm_hour) << ":"<< g_strdup_printf ("%02d",time_p->tm_min) << ":" << g_strdup_printf ("%02d",time_p->tm_sec) << " " << frame_num;
            snprintf(txt_params->display_text, 2*MAX_DISPLAY_LEN, "%s ", time_string.str().c_str());
            // std::cout << "time:" << time_string.str() << std::endl;
            time_string.clear();
            time_string.str("");
            /* Now set the offsets where the string should appear */
            
            txt_params->x_offset = 10;
            txt_params->y_offset = 45;

            /* Font , font-color and font-size */
            txt_params->font_params.font_name = "Serif";
            txt_params->font_params.font_size = 20;
            txt_params->font_params.font_color.red = 1.0;
            txt_params->font_params.font_color.green = 1.0;
            txt_params->font_params.font_color.blue = 1.0;
            txt_params->font_params.font_color.alpha = 1.0;

            /* Text background color */
            txt_params->set_bg_clr = 1;
            txt_params->text_bg_clr.red = 0.0;
            txt_params->text_bg_clr.green = 0.0;
            txt_params->text_bg_clr.blue = 0.0;
            txt_params->text_bg_clr.alpha = 0.1;
            if(abnormal_id.size())
            {   
                int abnormal_num = 0;

                NvOSD_RectParams *rect_params = NULL;
                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                    std::map<std::string, std::vector<std::string>>::iterator it;
                    for (it = abnormal_id.begin(); it != abnormal_id.end(); ++it)
                    {   
                        std::vector<std::string> file_name;
                        stringsplit(it->first, 'r', file_name);
                        //std::cout << std::stoi(*file_name.begin()) << std::endl;
                        if(it->second.size() && std::stoi(*file_name.begin()) == stream_id) 
                        {
                           if(std::find(it->second.begin(), it->second.end(), std::to_string(obj_meta->object_id)+(obj_meta->obj_label)) != it->second.end() )
                            {
                                //display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                                rect_params = &display_meta->rect_params[abnormal_num];
                                rect_params->left =  obj_meta->rect_params.left;
                                rect_params->top =  obj_meta->rect_params.top;
                                rect_params->width = obj_meta->rect_params.width;
                                rect_params->height = obj_meta->rect_params.height;
                                rect_params->border_width = 5;
                                rect_params->border_color = (NvOSD_ColorParams){1, 0 , 0, 1};
                                rect_params->has_bg_color = 1;
                                rect_params->bg_color = (NvOSD_ColorParams){1, 0 , 0, 0.2};
                                display_meta->num_rects++;
                                abnormal_num++;
                            }
                            if(std::find(it->second.begin(), it->second.end(), std::to_string(obj_meta->object_id)+"no-mask") != it->second.end() )
                            {
                                //display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                                rect_params = &display_meta->rect_params[abnormal_num];
                                rect_params->left =  obj_meta->rect_params.left;
                                rect_params->top =  obj_meta->rect_params.top;
                                rect_params->width = obj_meta->rect_params.width;
                                rect_params->height = obj_meta->rect_params.height;
                                rect_params->border_width = 5;
                                rect_params->border_color = (NvOSD_ColorParams){1, 0 , 0, 1};
                                rect_params->has_bg_color = 1;
                                rect_params->bg_color = (NvOSD_ColorParams){1, 0 , 0, 0.2};
                                display_meta->num_rects++;
                                abnormal_num++;
                            }
                        }

                    }

                }
            }
            nvds_add_display_meta_to_frame(frame_meta, display_meta);
        }
        out_string << "]";
        if (post_http_info){
                std::string http_string = out_string.str();
                http_string[0] = '[';
                g_print (" Frame Number = %d,  %s\n",frame_num, http_string.c_str());
                char warning_http[512];
                sprintf(warning_http,WARNING_OUT_HTTP,appCtx->config.multi_source_config[0].http_trans,appCtx->config.multi_source_config[0].http_port);
                // http_post(warning_http, http_string.c_str());
            }
        // std::cout << "frame num:" << frame_num << std::endl;
    }
}
