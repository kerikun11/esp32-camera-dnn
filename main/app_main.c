#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "sdkconfig.h"

#include "esp_http_server.h"

#include "Validation_inference.h"
#include "Validation_parameters.h"

#include "app_camera.h"
#include "app_wifi.h"
#include "esp_camera.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

static const char* TAG = "camera";

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t http_server_init();

void* _context = NULL;

void app_main()
{
    app_wifi_main();
    app_camera_main();

    _context = nnablart_validation_allocate_context(Validation_parameters);

    vTaskDelay(100 / portTICK_PERIOD_MS);
    http_server_init();
}

esp_err_t jpg_httpd_handler(httpd_req_t* req)
{
    camera_fb_t* fb = NULL;
    esp_err_t res = ESP_OK;
    size_t fb_len = 0;
    int64_t fr_start = esp_timer_get_time();

    fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    res = httpd_resp_set_type(req, "image/jpeg");
    if (res == ESP_OK) {
        res = httpd_resp_set_hdr(req, "Content-Disposition",
            "inline; filename=capture.jpg");
    }

    if (res == ESP_OK) {
        fb_len = fb->len;
        res = httpd_resp_send(req, (const char*)fb->buf, fb->len);
    }
    esp_camera_fb_return(fb);
    int64_t fr_end = esp_timer_get_time();
    ESP_LOGI(TAG, "JPG: %uKB %ums", (uint32_t)(fb_len / 1024),
        (uint32_t)((fr_end - fr_start) / 1000));
    return res;
}

esp_err_t jpg_stream_httpd_handler(httpd_req_t* req)
{
    size_t _jpg_buf_len;
    uint8_t* _jpg_buf;
    char* part_buf[64];
    uint8_t resized_img[NNABLART_VALIDATION_INPUT0_SIZE];

    static int64_t last_frame = 0;
    if (!last_frame) {
        last_frame = esp_timer_get_time();
    }

    esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }

    float* nn_input_buffer = nnablart_validation_input_buffer(_context, 0);

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
        } else {
            if (fb->format != PIXFORMAT_JPEG) {
                bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
                if (!jpeg_converted) {
                    ESP_LOGE(TAG, "JPEG compression failed");
                    esp_camera_fb_return(fb);
                    res = ESP_FAIL;
                }
            } else {
                _jpg_buf_len = fb->len;
                _jpg_buf = fb->buf;
            }
        }
        if (res == ESP_OK) {
            size_t hlen = snprintf((char*)part_buf, 64, _STREAM_PART, _jpg_buf_len);

            res = httpd_resp_send_chunk(req, (const char*)part_buf, hlen);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char*)_jpg_buf, _jpg_buf_len);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY,
                strlen(_STREAM_BOUNDARY));
        }
        if (fb->format != PIXFORMAT_JPEG) {
            free(_jpg_buf);
        }

        // リサイズ
        stbir_resize_uint8(fb->buf, 160, 120, 0, resized_img, 28, 28, 0, 1);
        esp_camera_fb_return(fb);

        // 白黒反転, 閾値処理
        for (int i = 0; i < NNABLART_VALIDATION_INPUT0_SIZE; i++) {
            uint8_t p = ~(resized_img[i]);

            if (p < 180) {
                p = 0;
            }

            nn_input_buffer[i] = p;
        }

        // Infer image
        int64_t infer_time = esp_timer_get_time();
        nnablart_validation_inference(_context);
        infer_time = (esp_timer_get_time() - infer_time) / 1000;

        // Fetch inference result
        float* probs = nnablart_validation_output_buffer(_context, 0);

        int top_class = 0;
        float top_probability = 0.0f;
        for (int class = 0; class < NNABLART_VALIDATION_OUTPUT0_SIZE; class ++) {
            if (top_probability < probs[class]) {
                top_probability = probs[class];
                top_class = class;
            }
        }

        int64_t fr_end = esp_timer_get_time();
        int64_t frame_time = fr_end - last_frame;
        last_frame = fr_end;
        frame_time /= 1000;
        ESP_LOGI(TAG, "Result %d   Frame-time %ums (Inferrence-time %ums)",
            top_class, (uint32_t)frame_time, (uint32_t)infer_time);
    }

    nnablart_validation_free_context(_context);
    last_frame = 0;
    return res;
}

static esp_err_t http_server_init()
{
    httpd_handle_t server;
    httpd_uri_t jpeg_uri = { .uri = "/jpg",
        .method = HTTP_GET,
        .handler = jpg_httpd_handler,
        .user_ctx = NULL };

    httpd_uri_t jpeg_stream_uri = { .uri = "/",
        .method = HTTP_GET,
        .handler = jpg_stream_httpd_handler,
        .user_ctx = NULL };

    httpd_config_t http_options = HTTPD_DEFAULT_CONFIG();

    ESP_ERROR_CHECK(httpd_start(&server, &http_options));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &jpeg_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &jpeg_stream_uri));

    return ESP_OK;
}
