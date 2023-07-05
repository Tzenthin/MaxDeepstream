#!/bin/bash
# deepstream要执行的算法,根据你选择的config配置文件运行对应的算法
deepstreamPath=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app
cd ${deepstreamPath}
./deepstream-app -c ./demo_config/face_arcface/deepstream_app.txt