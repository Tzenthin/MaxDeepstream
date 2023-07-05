#!/bin/bash
#运行人脸识别的三个级联算法，输入的是你要识别的图片名称
/opt/project/script/face_arcface/peoplenet/build/peoplenet_arcface $1
sleep 1
/opt/project/script/face_arcface/peoplenet/build/landmark $1
sleep 1
/opt/project/script/face_arcface/peoplenet/build/arcface_recog $1


