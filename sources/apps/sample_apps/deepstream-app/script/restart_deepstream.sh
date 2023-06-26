#!/bin/bash
pid_deepstream=`nvidia-smi | grep ./deepstream-app | grep -v grep | awk '{print $5}'`
pid_deepstream_arr=()

index=0
for i in $pid_deepstream; do
    pid_deepstream_arr[index]=$i
    index=$(($index+1))
done
for (( i=0;i<${#pid_deepstream_arr[@]};i++ ))
do
    echo "xuwang" | sudo -S kill -9 ${pid_deepstream_arr[i]}
done
rm -rf /home/ahjz/.cache/gstreamer-1.0/registry.x86_64.bin
echo "xuwang" | sudo -S su - ahjz -c "nohup /opt/project/deepstream_auto/config/deepstream.sh > /opt/project/deepstream_auto/log/deepstream_log.txt 2>&1 &"


