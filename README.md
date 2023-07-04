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
目前maxDeepstream除了支持deepstream本身的一些peoplenet, trafficcamnet外，也支持自定义的级联网络逻辑推理，在完成以上代码更新以及编译后，即可进行以下demo的复现     
- 1、中文车牌检测  
模型加载，模型包括三个，1级模型为nvidia的trafficcamnet，二级模型为车牌检测模型，三级模型为车牌识别模型  
  * **step1**、准备好模型以及测试视频:将公司192.169.2.126机器的/opt/project/engine/car_lrp中所有文件拷贝至你的机器相同目录下，测试视频放在/opt/project/demo/car_lrp下  
  * **step2**、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/car_lrp下的配置文件拷贝到你的机器相应的位置目录下
  * **step3**、 cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/car_lrp/deepstream_app.txt  
  注意初步加载模型，deepstream会根据你的机器tensorrt版本重新生成.engine序列化引擎文件，将耗费一点时间，等生成完成后，下次重运行时，模型的加载将会很快  
  另一点需要注意的是在摄像头画面中选择的roi区域的角点坐标需要在$PATH/apps/sample_apps/deepstream-app/demo_config/car_lrp/nvdsanalytics_config.txt标出  

- 2、目标拥挤（行人）  
模型加载，模型为nvidia的peoplenet  
  * **step1** 、准备好模型以及测试视频：将公司192.169.2.126机器的/opt/project/engine/crowded中所有模型文件拷贝至你的机器相同目录下，测试视频放在/opt/project/demo/crowded下  
  * **step2** 、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/crowded下的配置文件拷贝到你的机器相应的位置目录下  
  * **step3** 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/crowded/deepstream_app.txt  
  
- 3、目标徘徊（行人）  
模型加载，模型为nvidia的peoplenet  
  * **step1** 、准备好模型以及测试视频：模型仍采取上面2中提到的peoplenet模型，测试视频也是如此，无需重复创建目录
  * **step2** 、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/hover下的配置文件拷贝到你的机器相应的位置目录下  
  * **step3** 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/hover/deepstream_app.txt  

- 4、方向检测（行人）  
模型加载，模型为nvidia的peoplenet  
  * **step1** 、准备好模型以及测试视频：模型仍采取上面2提到的peoplenet模型，测试视频也是如此  
  * **step2** 、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/dir下的配置文件拷贝到你的机器相应的位置目录下  
  * **step3** 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/dir/deepstream_app.txt  

- 5、目标蹦跑（行人）
模型加载，模型为nvidia的peoplenet  
  * **step1** 、准备好模型以及测试视频：模型仍采取上面2提到的peoplenet模型，测试视频放在/opt/project/demo/running下  
  * **step2** 、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/running下的配置文件拷贝到你的机器相应的位置目录下  
  * **step3** 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/running/deepstream_app.txt  

- 6、口罩检测
模型加载，模型为nvidia的peoplenet  
  * **step1** 、准备好模型以及测试视频：模型仍采取上面2提到的peoplenet模型，测试视频放在/opt/project/demo/mask下  
  * **step2** 、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/mask下的配置文件拷贝到你的机器相应的位置目录下  
  * **step3** 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/mask/deepstream_app.txt  

- 7、厨师帽检测（明厨亮灶）
模型加载，模型为nvidia的detectnet_v2  
  * **step1** 、准备好模型以及测试视频：将公司192.169.2.126机器的/opt/project/engine/chefcap中所有模型文件拷贝至你的机器相同目录下，测试视频放在/opt/project/demo/chefcap下  
  * **step2** 、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/chefcap下的配置文件拷贝到你的机器相应的位置目录下  
  * **step3** 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/chefcap/deepstream_app.txt  

- 8、抽烟检测（明厨亮灶）  
模型加载，模型为nvidia的detectnet_v2  
  * **step1** 、准备好模型以及测试视频：将公司192.169.2.126机器的/opt/project/engine/smoking中所有模型文件拷贝至你的机器相同目录下，测试视频仍拉的是7中的视频  
  * **step2** 、准备好算法配置文件，将本资源$PATH/apps/sample_apps/deepstream-app/demo_config/smoking下的配置文件拷贝到你的机器相应的位置目录下  
  * **step3** 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/smoking/deepstream_app.txt  

- 9、人脸识别  
模型加载，模型为retinaface+arcface  
  * **step1** 、准备好模型,需要按照官方指导搭建insightface工程[insightface](https://github.com/xuwangJ/insightface_align),找到insightface_align/detection/retinaface/export_onnx.py找到下面代码行
  ```
  #根据实际拉流的多少，默认拉的是16路摄像头，即batch=16
  parser.add_argument('--input_shape', nargs='+', default=[16, 3, 540, 960], type=int, help='input shape.')
  ```
  执行此脚本导出onnx模型文件  
  
  * **step2** 、安装tensorRT8.x版本，测试用的是TensorRT-8.2.5.1版本，cd bin目录下执行以下命令  
  ```
  #--onnx 是你step1转换得到的onnx模型 --saveEngine是tensorRT转换得到的engine引擎，fp16表示是fp16的推理精度，转换较慢，大概10-20分钟
  ./trtexec --onnx=./r50.onnx --saveEngine=retinaface_fp16.engine --fp16
  ```
  测试视频放至/opt/project/demo/arcface下，根据onnx的batch_size拉取对应摄像头路数目进行人脸识别  

  * **step3** 、进行人脸识别人脸底库数据生成，简单思路可采取pytorh模型，推理经过Align人脸纠正，再经过arcface提取特征生成人脸库csv文件，但为了掌握tensorRT引擎转换以及如何推理引擎文件，以下步骤介绍如何利用tensorRT进行人脸库生成的：  
      - a 、要是涉及到前后端通信，根据实际后端ip以及端口修改/op/project/Algorithm_face.json文件内容  
      - b 、加载peoplenet的engine，可利用2中目标拥挤的demo,将batch-size设置成1,运行demo后会在/opt/project/engine/crowded下会生成resnet34_peoplenet_int8.etlt_b1_gpu0_int8.engine,将其copy到/opt/project/engine/arcface下,随后cd /opt/project/script/peoplenet下进行编译:mkdir build, cd build, cmake .., make（此处默认cmakelist中的一些opencv库，tensorRT库，jsoncpp库你已经安装好了）  
      - c 、以上编译无误后，创建/opt/project/tmp/目录，并在下面创建face_picture和result文件夹，随后在result下分别创建face,landmark,arcface三个文件夹，并将准备注册的所有人脸图片x.jpg  放入face_picture文件夹下，注意chmod打开新创建文件的权限，
      - d 、cd build执行./peoplenet xx.jpg命令，最终会在/opt/project/tmp/result/face下生成xx.jpg人脸图片  
      - e 、对上一步得到的x.jpg人脸图片进行landmark提取并进行align纠正，将192.169.2.126机器的/opt/project/engine/arcface中的landmark_batch_1017.onnx进行tensorRT引擎转换，将其copy至tensorRT的bin目录下，执行以下指令转换得到landmark的engine,并将其copy回/opt/project/engine/arcface下: 
      ```
      ./trtexec --onnx=landmark_batch_1017.onnx --minShapes=input:1x3x112x112 --optShapes=input:128x3x112x112 --maxShapes=input:256x3x112x112 --saveEngine=landmark_b256_dynamic_fp16.engine
      ```  
      - f 、对以上landmark模型进行人脸align, cd /opt/project/script/face_arcface/landmark/ ,进行编译:mkdir build, cd build, cmake .., make即可，然后build目录下执行./landmark xx.jpg，人脸align图片保存在/opt/project/tmp/result/landmark中
      - g 、接着进行arcface转onnx, 在step1中的insightface工程中，找到/recognition/arcface_torch/torch2onnx.py 并执行  
      ```
      #model/r18是模型pt目录位置 --network=r18是arcface训练时选择的backbone,这里选择的是resnet18
      python torch2onnx.py model/r18 --output=./model/arcface_fanghang_r18.onnx --network=r18
      ```  
      有了arcface_fanghang_r18.onnx这onnx模型，再进行tensorRT引擎转换，同样的，将其copy到bin下，执行  
      ```
      ./trtexec --onnx=arcface_fanghang_r18.onnx --minShapes=input:1x3x112x112 --optShapes=input:128x3x112x112 --maxShapes=input:256x3x112x112 --saveEngine=arcface_b256_dynamic_fp16.engine
      ```  
      - h 、得到arcfacce的engine引擎文件后，将其copy至/opt/project/engine/arcface, cd /opt/project/script/face_arcface/arcface/, 进行编译， mkdir build, cd build, cmake .., make即可，然后build目录下执行./arcface xx.jpg，人脸特征向量csv文件保存在/opt/project/config/face.csv里  

  * **step4** 、有了人脸库csv,那么下一步就是拉摄像头rtsp流进行人脸特征提取并与csv中注册的人脸进行相似度计算  
      - a 、先进行gst-dsexample相关库的编译，cd $PATH/gst-plugins/gst-dsexample/cuda_c 进行make; cd $PATH/gst-plugins/gst-dsexample/tensorrt_lib 进行make; cd $PATH/gst-plugins/gst-dsexample/ 进行make, sudo make install  
      在bashrc中配置以下环境变量：  
      ```
      export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-6.1/sources/gst-plugins/gst-dsexample/tensorrt_lib:$LD_LIBRARY_PATH
      ```  
      - b 、cd $PATH/apps/sample_apps/deepstream-app 执行./deepstream-app -c ./demo_config/face_arcface/deepstream_app.txt即可拉15路摄像头画面进行人脸识别
      注意两点：一是人脸识别的预警信息通过http协议传输，预警图片数据会自动存在/opt/project/tmp/face_warningImg下，以摄像头编号以及其在人脸库的id创建文件夹进行保存;二是要想把预警数据包传输给后端，需要查看$PATH/gst-plugins/gst-dsexample/gstdsexample.cpp 第1223  
      ```
      http_post(warning_http, http_string.c_str());
      ```  
      代码是否被注释，要想与后端通信，此处应该取消注释  

- 10、行人reid(后续支持)  
模型加载，模型为nvidia的peoplenet+Fastreid
  具体实现可参照9中的人脸识别方案，利用peoplenet检测出行人bbox框，将这个bbox作为行人reid的特征提取的输入，并将输出的feature_map与底库数据进行相似度计算   



# 脚本
要保证代码在实际部署时能够持续运行，并在一些特殊情况下，系统能够通过算法执行的进程有无从而进行监督并进行自启动运行，固添加了以下几个脚本,脚本在本资源script文件夹下，需要将其复制到系统/opt/project下：  
- 1、摄像头因网络，电压等问题导致掉线后又恢复，为了不影响deepstream程序执行，需要添加摄像头离线后再恢复的deepstream程序自启动脚本。脚本为restart_deepstream.sh，具体脚本被执行是在$PATH/apps/sample_apps/deepstream-app/deepstream_app_main.c 209行，程序根据拉流计算的FPS的异常进行重启动的检查  
- 2、算法机器开机自执行deepstream-app程序，执行脚本：（需要将此脚本配置成ubuntu系统开机自启动，可自行上网查询有关资料，网上介绍的方案都可实现），脚本为ai_run.sh  
- 3、前后端http上传图片并下达算法执行指令，从而算法实现人脸注册、人脸识别，具体脚本如下所述：  
     - a、与后端通信脚本，采取的是python以及web库的方案，根据接受到的json字段信息分别执行不同的算法模块脚本，脚本为http_post.py, 后端通过http://$机器IP:8081/compare_v2接口，向算法机器传输json字段，算法侧http_post.py脚本分析此json字段，从而运行下达的指令  
     - b、人脸注册，即将人脸图片经过人脸检测，人脸landmark align以及arcface人脸识别并最终将feature向量写进csv文件中，脚本为：faceregistry.sh   
     - c、人脸识别，即将后端传输来的json中制定的图片进行人脸相似读计算，脚本为facerecognition.sh，这里需要编译一下/opt/project/script/face_arcface几个源码以及确保/opt/script/Algorithm_face.json中的接口通信信息是正确的，首先将peoplenet的CMakeLists.txt的cuda_add_executable改成cuda_add_executable(peoplenet_arcface peoplenet_arcface.cpp get_post.cpp)，下面的target_link_libraries要生成名为peoplenet_arcface的可执行文件。同样的/opt/project/script/face_arcface/arcFace下的arcface也要执行相似的操作，对arcface_v2.cpp编译从而生成/opt/project/script/face_arcface/peoplenet/build/arcface_recog可执行文件

# 注意事项
- 1、部分脚本执行时，会提示Could not open X Display,是因为你执行的deepstream_app程序用的是终端窗口进行demo展示，需要在[sink]将type=1，关闭x display即可 
- 2、确保与后端通信时，一些算法或者脚本配置的http接口的正确，包括配置了正确的后端ip以及接口端口
- 3、算法配置文件的流width\height要与视频画面的分辨率保持一致，否则算法推流的流画面可能出现卡顿以及模糊情况  
- 4、不同的机器设备间，同一个tensorrt引擎engine文件有可能不兼容，最好在每个本机利用其内部的tensorRT环境重新转引擎  
- 5、前期部署时，[sink]属性最好选择终端显示，要是所有画面不是同时加载，在排除不是算法框架的问题下，需要升级摄像头的驱动，联系相应设备官方，更新至最新即可  
- 6、deepstream一般会很稳定的持续运行，但也遇到过流画面掉线恢复后，deepstream的fps仍然为0，这里只需要在分析fps异常时，删除缓存文件，重启下deepstream-app程序即可，即$PATH/apps/sample_apps/deepstream-app/deepstream_app_main.c 209行所执行的脚本命令

# 后续优化方向
- 1、 实际项目部署时，因对2-4mm毫米的摄像头的安装需求，需要对画面标定纠正，目前纠正采取的是ffmpeg+nginx对推的http流转成http流，并在gis中播放，但出现延迟4-5s的情况，后续可改成采取MaxDeepstream直接转RTSP流  
- 2、 MaxDeepstream中目前对arcface网络输出后处理的anchor regress采取的是cpu，因拉取16+路数，可将这部分采取cuda c优化加速  
- 3、 gst-dsexample在deepstream中，其pipeline处于在较后位置，导致无法使用tracker，是否可以在tracker pipeline前进行数据流转mat并进行tensorrt引擎推理，充分利用检测+跟踪进行算法分析  
