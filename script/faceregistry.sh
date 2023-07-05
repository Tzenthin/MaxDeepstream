#!/bin/bash
#运行人脸注册的三个级联算法，输入的是你要识别的图片名称
/opt/project/script/face_arcface/peoplenet/build/peoplenet $1
sleep 1
/opt/project/script/face_arcface/landmark/build/landmark $1
sleep 1
/opt/project/script/face_arcface/arcFace/build/arcface_write $1