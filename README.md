# 介绍
deepstream的升级版，简称**maxDeepstream**，在deepstream-6.1及以上版本中，基于其sample_apps中deepstream-app源码，提出了几点修改:  
- 1、在deepstream的gst-plugins基础上，修改并支持任何tensorRT的模型引擎的推理;  
- 2、将deepstream-nvdsanalytics-test的代码整理融合进deepstream-app中进行目标检测以及跟踪后的meta-data数据分析。
- 3、添加了http通信协议，将预警数据以json数据包的形式通过http_post发送

# 安装以及代码更新
为简短描述，**令PATH=/opt/nvidia/deepstream/deepstream-6.1/sources/**  
*step1*、参照官方指导文件，安装deepstream-6.1：[deepstream安装指南](https://docs.nvidia.com/metropolis/deepstream/6.1/dev-guide/text/DS_Quickstart.html)  

*step2*、在确保deepstream-6.1安装完毕以及deepstream-app正常运行后，将以下代码文件，经过文件对比后，同步到deepstream-6.1中  
- a、在$PATH/apps/apps-common/includes/ 以及../src添加本资源相应位置的get_post.h 和get_post.c文件  
- b、前后对比本资源与官方版本，更新$PATH/apps/sample_apps/deepstream-app/deepstream_app_main.c  
- c、前后对比本资源与官方版本，更新$PATH/apps/sample_apps/deepstream-app/deepstream_app.h以及./deepstream_app.c
- d、在$PATH/apps/sample_apps/deepstream-app下添加本资源相应目录deepstream_nvdsanalytics_meta.cpp  
- e、因配置文件中[sources]字段添加了以下几个属性：
   * **model-type** （异常事件类型1级分类）
   * **env-type**（异常事件类型2级分类）
   * **hover-interval**（异常事件触发持续阈值）
   * **father-ip**（算法设备ip）
   * **ip**（摄像头ip）
   * **http-port**（http通信端口）
   * **http-trans**（http协议地址）
   * **争对奔跑检测事件，hover-interval属性要改为run-param属性**  
为添加以上字段属性，需前后对比本资源与官方版本，更新$PATH/apps/apps-common/src/deepstream_config_file_parser.c以及../include/deepstream_sources.h  
- f、确保以上操作无误后，cd $PATH/apps/sample_apps/deepstream-app 直接 make 进行编译
  
*step3*、以上是针对添加nvdsanalytics数据分析模块修改的代码块，以下步骤是修改如何在gst-plugins/gst-dsexample中支持自定义网络推理逻辑
- a、在$PATH/gst-plugins/gst-dsexample下 添加本资源相应目录下的cuda_c tensorrt_lib两个文件夹，并同步目录下相应代码，**注意各自makefile中库版本以及路径** 
- b、在$PATH/gst-plugins/gst-dsexample下添加本资源相应位置的get_post.h 和get_post.cpp文件   
- c、前后对比本资源与官方版本， 更新$PATH/gst-plugins/gst-dsexample/gstdsexample.cpp  
- d、因配置文件中[dsexample]字段添加了以下几个属性:
  * **batch-size**（拉流source数量）
  * **device-ip**（算法设备ip）
  * **http-port**（http通信端口）
  * **http-trans**（http协议地址）
  * **env-interval**（视频流间隔几帧进行算法处理）  
  为添加以上字段，需前后对比本资源与官方版本，更新$PATH/gst-plugins/gst-dsexample/gstdsexample.h  
  前后对比本资源与官方版本，更新$PATH/apps/apps-common/includes/deepstream_dsexample.h  
  前后对比本资源与官方版本，更新$PATH/apps/apps-common/src/deepstream_config_file_parser.c  
  前后对比本资源与官方版本，更新$PATH/apps/apps-common/src/deepstream_dsexample.c  
  前后对比本资源与官方版本，更新$PATH/apps/apps-common/src/deepstream-yaml/deepstream_dsexample_yaml.cpp
- e、在确保以上代码更新无误后，先在cuda_c下进行make, 再在tensorrt_lib下make, 最后在$PATH/gst-plugins/gst-dsexample/下 make clean,make,sudo make install即可
  
# 测试
目前maxDeepstream除了支持deepstream本身的一些peoplenet, trafficcamnet外，也支持自定义的级联网络逻辑推理，在完成以上代码更新以及编译后，即可进行以下demo的复现，复现拉的是mp4本地视频，要拉rtsp,需要修改配置文件[source]字段type    
- 1、中文车牌检测，支持绿色车牌识别  
模型加载，部分模型经过[nvidia TAO工具](https://docs.nvidia.com/tao/tao-toolkit/index.html）进行微调训练)，模型包括三个，1级模型为nvidia的  trafficcamnet，二级模型为重训练的车牌检测模型，三级模型为重训练的车牌识别模型
- step 1、准备好模型以及测试视频:将192.169.2.126机器的/opt/project/engine/car_lrp中所有文件拷贝至本机/opt/project/engine/car_lrp，测试视频放在/opt/project/demo/car_lrp下。





# 脚本
1、deepstream程序重启脚本，一般是在因网络波动、电压等因素导致的摄像头视频流不稳定需要执行此脚本。脚本位置：  
2、算法机器自启动脚本，关机重启后直接执行deepstream-app程序  

# 注意事项
1、算法机器ip变化了，会导致与后端http通信失败  
2、算法配置文件的流width\height要与视频画面的分辨率保持一致，否则算法推流的流画面可能出现卡顿以及模糊情况  
3、不同的机器设备间，同一个tensorrt引擎engine文件有可能不兼容，最好在每个本机利用其内部的tensorRT环境重新转引擎  
4、前期部署时，[sink]属性最好选择终端显示，要是所有画面不是同时加载，在排除不是算法框架的问题下，需要升级摄像头的驱动，联系相应设备官方，更新至最新即可  
5、deepstream一般会很稳定的持续运行，但也遇到过流画面掉线恢复后，deepstream的fps仍然为0，这里只需要在分析fps异常时，删除缓存文件，重启下  deepstream-app程序即可  
