#!/bin/bash
#可变参数变量

while true
do  
    sleep 5
    pid_deepstream=`nvidia-smi | grep ./deepstream-app | grep -v grep | awk '{print $5}'`
    if [ -n "$pid_deepstream" ]
    then
        echo "deepstream online"
    else
        #不关闭机器，只是重启算法
        echo \"xuwang\" | sudo -S nohup /opt/project/script/restart_deepstream.sh > /opt/project/log/deepstream_log.txt 2>&1 &
    fi

    #要是有多个算法进程，算法重启
    deepstream_index=0
    for i in $pid_deepstream; do
        deepstream_index=$(($deepstream_index+1))
    done

    if [ "$deepstream_index" -gt "2" ] || [ "$deepstream_index" -eq "1" ]
    then
        echo \"xuwang\" | sudo -S nohup /opt/project/script/restart_deepstream.sh > /opt/project/log/deepstream_log.txt 2>&1 &
    fi
done