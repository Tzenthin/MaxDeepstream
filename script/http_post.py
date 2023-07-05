
# coding=UTF-8<code>
import os
import web
import json
import socket  

timeout = 40    
socket.setdefaulttimeout(timeout)

args = {
    'token': '9all1981,ljdrnc#',
    'img_save_path': './upload_img',
    'txt_save': './result'
}


urls_v2 = (
    '/', 'index',
    '/compare_v2', 'compare_v2',
)

class index:
    def GET(self):
        return "hello world!"

class compare_v2:
    def POST(self):

        parms = web.data()
        parms = json.loads(parms)
        print(parms)
        code = parms['code']
        env = parms['env']


        if code == "39": #人脸识别事件的编号
            img_str = parms['name']
            if env == "01": #进行人脸注册
                os.system('sh /opt/project/script/faceregistry.sh %s' %img_str)
            if env == "02": #进行人脸注销
                json_p = {
                    "code" : "39",
                    "status": "404",
                    "info": "Face id cancellation is not currently supported"
                }
                return json.dumps(json_p)
            if env == "03": #进行人脸识别
                os.system('sh /opt/project/script/facerecognition.sh %s' %img_str)
        if code == "73": #deepstream进程重启
            if env == "01": #算法重启
                os.system('sh /opt/project/script/restart_deepstream.sh')
            if env == "02": #系统重启
                os.system('reboot')

        else:
            json_p = {
                "code" : "",
                "status": "404",
                "info": "The code unknown"
            }
            return json.dumps(json_p)
            
        json_p = {
            "code" : "",
            "status": "200",
            "info": "http post successful"
            }
        return json.dumps(json_p)

class add:
    def POST(self):
        
        parms = web.data()
        
        parms = json.loads(parms)
        print("parms:",parms)
        img_str0 = parms['num0']
        
        img_str1 = parms['num1']
        # need_face_crop = False
        
        print("执行")
        score=int(img_str0)+int(img_str1)
        # score, msg, face_a, face_b = face_api(img0, img1)
        json_p = {"status": 200,
                "score": score}
        return json.dumps(json_p)

if __name__ == "__main__":
    app = web.application(urls_v2, globals())
    app.run()
