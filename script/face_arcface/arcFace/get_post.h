//get_post.h

#ifndef get_post__h
#define get_post__h

#ifdef __cplusplus
extern "C"
{
#endif

#define MY_HTTP_DEFAULT_URL "http://113.90.117.71"
#define MY_HTTP_DEFAULT_PORT 80
//add by xuwang
#define WARNING_OUT_HTTP "%s:%d/data/api/pyapi/warning/add"
#define WARNING_OUT_HTTP2 "%s:%d/data/api/pyapi/warning/images/activity"
#define REBOOT_OUT_HTTP "%s:%d/data/api/pyapi/camera/switch?status=R"
#define SWITCH_GET_HTTP "%s:%d/data/api/pyapi/activity/switch"
#define SWITCH_GET_HTTP_C "%s:%d/data/api/pyapi/camera/switch?status=R"

// #define TOSERVICES_HTTP_DEFAULT_URL "http://127.0.0.1"
// #define TOSERVICES_HTTP_DEFAULT_URL2 "http://192.168.2.50"
// #define TOSERVICES_HTTP_DEFAULT_PORT 9099
// #define THIS_PC_IP "192.168.2.209"

static int http_tcpclient_create(const char *host, int port);
static void http_tcpclient_close(int socket);
static int http_parse_url(const char *url, char *host, char *file, int *port);
static int http_tcpclient_recv(int socket, char *lpbuff);
static int http_tcpclient_send(int socket, char *buff, int size);
static char *http_parse_result(const char*lpbuf);
char *http_post(const char *url, const char *post_str);
char *http_get(const char *url);





#ifdef __cplusplus
}
#endif



#endif
 