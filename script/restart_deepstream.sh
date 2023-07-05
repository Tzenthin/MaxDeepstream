#!/bin/bash
#deepstream算法重新启动脚本
pid_deepstream=`nvidia-smi | grep ./deepstream-app | grep -v grep | awk '{print $5}'`
pid_deepstream_arr=()

index=0
for i in $pid_deepstream; do
    pid_deepstream_arr[index]=$i
    index=$(($index+1))
done
#杀死算法进程
for (( i=0;i<${#pid_deepstream_arr[@]};i++ ))
do
    echo "xuwang" | sudo -S kill -9 ${pid_deepstream_arr[i]}
done
#删除算法缓存
rm -rf /home/xuwang/.cache/gstreamer-1.0/registry.x86_64.bin
#重启算法
echo "xuwang" | sudo -S su - xuwang -c "nohup /opt/project/script/deepstream.sh > /opt/project/log/deepstream_log.txt 2>&1 &"


