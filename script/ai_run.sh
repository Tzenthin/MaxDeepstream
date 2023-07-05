#!/bin/bash
#运行deepstream脚本
nohup /opt/project/script/deepstream.sh > /opt/project/log/deepstream_log.txt 2>&1 &
sleep 1
#运行后端通信脚本
nohup python /opt/project/script/http_post.py 8081
sleep 1
#运行算法进程监测脚本
nohup /opt/project/script/check_deepAI.sh > /opt/project/log/checkDeepAI_log.txt 2>&1 &
