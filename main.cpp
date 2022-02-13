#include <Arduino.h>
// Tensor flow library
#include "include/TensorFlowLite_ESP32/src/TensorFlowLite_ESP32.h"
// internet conections

#include "WiFi.h"
#include <WiFiClientSecure.h>
#include "include/arduino-mqtt-master/src/MQTTClient.h"
// disable brownout problems
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
// conversor de imagem
#include "img_converters.h"
// TensorFlow Files
#include "tflite_model_files/detection_responder.h"
#include "tflite_model_files/model_settings.h"
#include "tflite_model_files/person_detect_model_data.h"
#include "include/TensorFlowLite_ESP32/src/tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "include/TensorFlowLite_ESP32/src/tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "include/TensorFlowLite_ESP32/src/tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "include/TensorFlowLite_ESP32/src/tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "include/TensorFlowLite_ESP32/src/tensorflow/lite/schema/schema_generated.h"
#include "include/TensorFlowLite_ESP32/src/tensorflow/lite/version.h"
// My classes
#include "cam_config/app_camera_esp.h"
#include "secrets.h"

#define debug(x) Serial.print(x)
#define debugln(x) Serial.println(x)
#define wf WiFi

uint8_t errorCount = 0;
//modificar números detendendo da câmera
const char *ESP32CAM_PUBLISH_TOPIC = "cam/3";
const char *ESP32CAM_SUBSCRIBE_TOPIC = "cam/3/msg";
const char *ESP32_SAVER_PUBLICSH_TOPIC = "saver/detect/pic";

const char *BEGIN_STREAM = "sb";
const char *END_STREAM = "se";

const int writeBufSize = 20 * 1024;
const int readBufSize = 512;

// Gerenciadores de multiprocessamento
TaskHandle_t TaskEncode;
TaskHandle_t TaskConnection;
TaskHandle_t TaskDetection;
SemaphoreHandle_t semaphoroCamera;
QueueHandle_t queueBuffer;
QueueHandle_t queueDetection;

WiFiClientSecure net = WiFiClientSecure();
MQTTClient client = MQTTClient(128, writeBufSize, readBufSize, true);
camera_fb_t *fb;
// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  uint8_t *jpgBuf1 = (uint8_t *)ps_malloc(writeBufSize);
  uint8_t *jpgBuf2 = (uint8_t *)ps_malloc(writeBufSize);
  size_t jpgLen1;
  size_t jpgLen2;

  bool isStreaming = false;
  // bool gotMessageCanSend = false;
  uint8_t person_score = 0;
  uint8_t no_person_score = 0;
  // Tf objects
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 65 * 1024; // 70
  static uint8_t tensor_arena[kTensorArenaSize];
  // Buffer de envio

} // namespace
void messageHandler(String &topic, String &payload)
{
  debugln(payload.c_str());
  if (payload.equals(BEGIN_STREAM))
  {
    isStreaming = true;
    if (TaskDetection != NULL)
      vTaskSuspend(TaskDetection);
  }
  else if (payload.equals(END_STREAM))
  {
    isStreaming = false;
    if (TaskDetection != NULL)
      vTaskResume(TaskDetection);
  }
}

void beginWiFi()
{
  wf.disconnect();
  wf.persistent(false);
  wf.mode(WIFI_STA);
  wf.begin(WL_SSID, WL_PASSWORD);
  // WiFi.begin("TP-Link_8263", "15237033");
  //  WiFi.begin("TEMPERO_DO_POLEN_COMFIBRA", "temperos");
  int countTries = 0;
  while (wf.status() != WL_CONNECTED)
  {
    delay(500);
    debug(".");
    countTries++;
    if (countTries > 20)
      break;
  }
  if (countTries > 20)
  {
    debug("try againgn\n");
    wf.disconnect();
    wf.begin(WL_SSID2, WL_PASSWORD2);
    countTries = 0;
    while (wf.status() != WL_CONNECTED)
    {
      delay(500);
      debug(".");
      countTries++;
      if (countTries > 20)
        ESP.restart();
    }
  }
  debugln("WL CONNECTED");
  // Configure WiFiClientSecure to use the AWS IoT device credentials
  net.setCACert(AWS_CERT_CA);
  net.setCertificate(AWS_CERT_CRT);
  net.setPrivateKey(AWS_CERT_PRIVATE);

  // Connect to the MQTT broker on the AWS endpoint we defined earlier
  client.begin(AWS_IOT_ENDPOINT, 8883, net);
  client.setCleanSession(true);
  client.onMessage(messageHandler);

  debugln("Connecting to AWS");

  while (!client.connect(THINGNAME))
  {
    debug(".");
    delay(100);
  }

  if (!client.connected())
  {
    debugln("AWS Timeout");
    ESP.restart();
    return;
  }
  // Topico de recebimento de mensagens
  client.subscribe(ESP32CAM_SUBSCRIBE_TOPIC);
  debugln("AWS Connected!");
}
bool copyPictureTF()
{
  fb = esp_camera_fb_get();
  if (!fb)
  {
    debugln("buffer nulo");
    return false;
  }
  else
  {
    // debugln(ESP.getMinFreePsram());
    if (fb->width == 96 && fb->height == 96)
    {
      memcpy(input->data.uint8, fb->buf, fb->len);
    }
    else
    {
      int post = 0;
      int puloY = fb->height / 96;
      int startY = (fb->height % 96) / 2;
      int puloX = fb->width / 96;
      int startX = (fb->width % 96) / 2;
      for (int y = startY; y < fb->height - startY; y += puloY)
      {
        for (int x = startX; x < fb->width - startX; x += puloX)
        {
          int getPos = y * fb->width + x;
          input->data.uint8[post] = fb->buf[getPos];
          post++;
        }
      }
    }
  }
  return true;
}

void encodeTask(void *arg)
{
  bool b = true;
  int bufCount;
  bool isThereOne;
  for (;;)
  {
    bufCount = 0;
    if (isStreaming)
    {
      fb = esp_camera_fb_get();
      if (b)
      {
        // Encode buf2
        free(jpgBuf2);
        jpgLen2 = 0;
        frame2jpg(fb, 8, &jpgBuf2, &jpgLen2);
        b = false;
        bufCount = 2;
      }
      else
      {
        // Enconde buf1
        free(jpgBuf1);
        jpgLen1 = 0;
        frame2jpg(fb, 8, &jpgBuf1, &jpgLen1);
        b = true;
        bufCount = 1;
      }
      xQueueSend(queueBuffer, &bufCount, 0);
    }
    else
    {
      xQueueReceive(queueDetection, &isThereOne, pdMS_TO_TICKS(5));
      if (isThereOne)
      {
        debugln("SEND DETECT 2");
        free(jpgBuf1);
        jpgLen1 = 0;
        frame2jpg(fb, 15, &jpgBuf1, &jpgLen1);
        bufCount = 3;
        b = true;
      }
    }
    if (bufCount != 0)
    {
      debug("\t\t\t\t ");
      debugln(bufCount);
    }
    xQueueSend(queueBuffer, &bufCount, portMAX_DELAY);
  }
}
void WiFiTask(void *arg)
{
  bool result = false;
  float imgCount = 0;
  unsigned long timein = millis();
  for (;;)
  {
    client.loop();
    int mBufCount;
    xQueueReceive(queueBuffer, &mBufCount, pdMS_TO_TICKS(1000));
    if (mBufCount == 3)
    {
      // Send to saver board
      result = client.publish(ESP32_SAVER_PUBLICSH_TOPIC, jpgBuf1, jpgLen1, false, 0);
      debug(result ? " S3\n" : " F3\n");
      if (!result)
        ESP.restart();
    }
    else
    {
      if (mBufCount == 2)
      {
        // Send buf 2;
        result = client.publish(ESP32CAM_PUBLISH_TOPIC, jpgBuf2, jpgLen2, false, 0);
        debugln(result ? "2-- " + String(jpgLen2) : " F2");
        if (!result)
        {
          errorCount++;
          if (errorCount > 3)
            ESP.restart();
        }
      }
      else if (mBufCount == 1)
      {
        // Send buf 1;
        result = client.publish(ESP32CAM_PUBLISH_TOPIC, jpgBuf1, jpgLen1, false, 0);
        debugln(result ? "1-- " + String(jpgLen1) : " F1");
        if (!result)
        {
          errorCount++;
          if (errorCount > 3)
            ESP.restart();
        }
      }
      imgCount += 1;
      if (millis() - timein >= 10000)
      {
        debug(imgCount / 10);
        debugln(" F/s");
        timein = millis();
        imgCount = 0;
      }
    }
  }
}
void detectionTask(void *arg)
{
  for (;;)
  {
    bool isThereOne = false;
    // debugln(heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    TfLiteTensor *output = interpreter->output(0);
    int timeIn = millis();
    bool success = true;
    xSemaphoreTake(semaphoroCamera, portMAX_DELAY);
    success = copyPictureTF();
    xSemaphoreGive(semaphoroCamera);
    // Run the model on this input and make sure it succeeds.
    if (!success)
    {
      vTaskDelay(pdMS_TO_TICKS(50));
      return;
    }
    if (kTfLiteOk != interpreter->Invoke())
    {
      error_reporter->Report("Invoke failed.");
    }
    person_score = output->data.uint8[kPersonIndex];
    no_person_score = output->data.uint8[kNotAPersonIndex];
    RespondToDetection(error_reporter, person_score, no_person_score);
    int timeOut = millis() - timeIn;
    Serial.printf("Process time: %d millis.\n", timeOut);
    if (person_score > no_person_score)
    {
      error_reporter->Report("Detectado");
      isThereOne = true;
      debugln("SEND DETEC 1");
      xQueueSend(queueDetection, &isThereOne, portMAX_DELAY);
      isThereOne = false;
      xQueueSend(queueDetection, &isThereOne, portMAX_DELAY);
    }
  }
}

void setup()
{
  Serial.begin(115200);
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // disable brownout detector

  if (!psramInit())
  {
    debugln("PSRAM_INT_FAIL");
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                       tflite::ops::micro::Register_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_AVERAGE_POOL_2D,
      tflite::ops::micro::Register_AVERAGE_POOL_2D());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }
  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  // Inicialize wifi connection
  beginWiFi();
  // Inicia Camera
  if (app_camera_init() < 0)
  {
    debugln("INIT CAMERA ERROR");
    ESP.restart();
  }
  // Configura multiproscessamento
  semaphoroCamera = xSemaphoreCreateMutex();
  queueBuffer = xQueueCreate(1, sizeof(int));
  queueDetection = xQueueCreate(1, sizeof(bool));

  xTaskCreatePinnedToCore(WiFiTask, "WiFiHandler", 5000, NULL, 1, &TaskConnection, 0);
  xTaskCreatePinnedToCore(encodeTask, "encodeHandler", 5000, NULL, 1, &TaskEncode, 0);
  xTaskCreatePinnedToCore(detectionTask, "detectionTask", 15000, NULL, 1, &TaskDetection, 1);
  // isConnected = socket->isConnect();
  debugln("SETUP DONE");
}
void loop()
{
  vTaskDelete(NULL);
};
