#include <Arduino.h>

#include "gesture_model.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


// =====================================================
// HARDWARE PINS
// =====================================================

#define LED_PIN 2
#define LED_PINR 25
#define LED_PING 26
#define LED_PINB 27


// =====================================================
// MODEL SETTINGS
// =====================================================

constexpr int kInputSize  = 63;
constexpr int kOutputSize = 4;

// Tensor memory
constexpr int kTensorArenaSize = 30 * 1024;

uint8_t tensor_arena[kTensorArenaSize];


// =====================================================
// TENSORFLOW LITE VARIABLES
// =====================================================

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// =====================================================
// OPERATION RESOLVER
// =====================================================
//
// Our neural network is:
//
// 63
//  ↓
// Dense(24) + ReLU
//  ↓
// Dense(12) + ReLU
//  ↓
// Dense(4) + Softmax
//
// Therefore we register:
// FullyConnected
// Relu
// Softmax
//

tflite::MicroMutableOpResolver<3> resolver;


// =====================================================
// INPUT DATA
// =====================================================

float features[kInputSize];

String line;


// =====================================================
// GESTURE NAMES
// =====================================================

const char* names[] =
{
  "FIST",
  "ONE",
  "TWO",
  "PALM"
};


// =====================================================
// LED BLINK SETTINGS
// =====================================================

// 0 = OFF
// 1 = SLOW BLINK
// 2 = FAST BLINK
// 3 = ON

int ledMode = 0;

unsigned long lastBlinkTime = 0;

bool ledState = LOW;

const unsigned long SLOW_BLINK = 1000;  // 1 second
const unsigned long FAST_BLINK = 200;   // 200 milliseconds


// =====================================================
// SETUP TINYML
// =====================================================

void setupTinyML()
{
  Serial.println();
  Serial.println("Initializing TinyML...");


  // ---------------------------------------------------
  // Load model
  // ---------------------------------------------------

  model = tflite::GetModel(gesture_model_tflite);

  if (model == nullptr)
  {
    Serial.println("ERROR: Model is NULL!");

    while (1)
    {
      delay(1000);
    }
  }

  Serial.println("Model loaded.");


  // ---------------------------------------------------
  // Check model schema
  // ---------------------------------------------------

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("ERROR: Model schema mismatch!");

    Serial.print("Model version: ");
    Serial.println(model->version());

    Serial.print("Expected version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);

    while (1)
    {
      delay(1000);
    }
  }

  Serial.println("Model schema OK.");


  // ---------------------------------------------------
  // Register required operators
  // ---------------------------------------------------

  if (resolver.AddFullyConnected() != kTfLiteOk)
  {
    Serial.println("ERROR: Could not add FullyConnected");

    while (1)
    {
      delay(1000);
    }
  }


  if (resolver.AddRelu() != kTfLiteOk)
  {
    Serial.println("ERROR: Could not add Relu");

    while (1)
    {
      delay(1000);
    }
  }


  if (resolver.AddSoftmax() != kTfLiteOk)
  {
    Serial.println("ERROR: Could not add Softmax");

    while (1)
    {
      delay(1000);
    }
  }


  Serial.println("TensorFlow operations registered.");


  // ---------------------------------------------------
  // Create interpreter
  // ---------------------------------------------------

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize
  );

  interpreter = &static_interpreter;


  // ---------------------------------------------------
  // Allocate tensors
  // ---------------------------------------------------

  Serial.println("Allocating tensors...");

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("ERROR: AllocateTensors FAILED!");

    Serial.println("Try increasing kTensorArenaSize.");

    while (1)
    {
      delay(1000);
    }
  }

  Serial.println("Tensor allocation successful.");


  // ---------------------------------------------------
  // Get input/output tensors
  // ---------------------------------------------------

  input = interpreter->input(0);
  output = interpreter->output(0);


  if (input == nullptr)
  {
    Serial.println("ERROR: Input tensor is NULL!");

    while (1)
    {
      delay(1000);
    }
  }


  if (output == nullptr)
  {
    Serial.println("ERROR: Output tensor is NULL!");

    while (1)
    {
      delay(1000);
    }
  }


  Serial.println("TinyML ready!");
  Serial.println();
}


// =====================================================
// LED CONTROL
// =====================================================

void updateLED()
{
  unsigned long currentTime = millis();


  // ---------------------------------------------------
  // FIST - LED OFF
  // ---------------------------------------------------

  if (ledMode == 0)
  {
    digitalWrite(LED_PIN, LOW);
    digitalWrite(LED_PINB, HIGH);
    digitalWrite(LED_PING, LOW);
    digitalWrite(LED_PINR, LOW);
    

    ledState = LOW;
  }


  // ---------------------------------------------------
  // PALM - LED ON
  // ---------------------------------------------------

  else if (ledMode == 3)
  {
    digitalWrite(LED_PIN, HIGH);
        digitalWrite(LED_PINB, LOW);
    digitalWrite(LED_PING, HIGH);
    digitalWrite(LED_PINR, LOW);
    ledState = HIGH;
  }


  // ---------------------------------------------------
  // ONE - SLOW BLINK
  // ---------------------------------------------------

  else if (ledMode == 1)
  {
    if (currentTime - lastBlinkTime >= SLOW_BLINK)
    {
      lastBlinkTime = currentTime;

      ledState = !ledState;
    digitalWrite(LED_PINB, LOW);
    digitalWrite(LED_PING, LOW);
    digitalWrite(LED_PINR, HIGH);
      digitalWrite(LED_PIN, ledState);
    }
  }


  // ---------------------------------------------------
  // TWO - FAST BLINK
  // ---------------------------------------------------

  else if (ledMode == 2)
  {
    if (currentTime - lastBlinkTime >= FAST_BLINK)
    {
      lastBlinkTime = currentTime;

      ledState = !ledState;
    digitalWrite(LED_PINB, LOW);
    digitalWrite(LED_PING, LOW);
    digitalWrite(LED_PINR, LOW);
      digitalWrite(LED_PIN, ledState);
    }
  }
}


// =====================================================
// PERFORM GESTURE INFERENCE
// =====================================================

void performGesture()
{
  // ---------------------------------------------------
  // Copy 63 features into model input
  // ---------------------------------------------------

  for (int i = 0; i < kInputSize; i++)
  {
    input->data.f[i] = constrain(
      features[i],
      -2.0f,
      2.0f
    );
  }


  // ---------------------------------------------------
  // Run inference
  // ---------------------------------------------------

  if (interpreter->Invoke() != kTfLiteOk)
  {
    Serial.println("ERROR: Inference failed!");

    return;
  }


  // ---------------------------------------------------
  // Find highest probability
  // ---------------------------------------------------

  int best = 0;

  float bestScore = output->data.f[0];


  for (int i = 1; i < kOutputSize; i++)
  {
    if (output->data.f[i] > bestScore)
    {
      bestScore = output->data.f[i];

      best = i;
    }
  }


  // ---------------------------------------------------
  // Display result
  // ---------------------------------------------------

  Serial.print("Gesture: ");
  Serial.print(names[best]);

  Serial.print(" | Confidence: ");
  Serial.println(bestScore, 3);


  // ---------------------------------------------------
  // Ignore low-confidence predictions
  // ---------------------------------------------------

  if (bestScore < 0.70f)
  {
    Serial.println("Low confidence - LED unchanged.");

    return;
  }


  // ---------------------------------------------------
  // Control LED
  // ---------------------------------------------------

  switch (best)
  {
    case 0:

      // FIST
      // LED OFF

      ledMode = 0;

      digitalWrite(LED_PIN, LOW);

      break;


    case 1:

      // ONE
      // SLOW BLINK

      ledMode = 1;

      lastBlinkTime = millis();

      break;


    case 2:

      // TWO
      // FAST BLINK

      ledMode = 2;

      lastBlinkTime = millis();

      break;


    case 3:

      // PALM
      // LED ON

      ledMode = 3;

      digitalWrite(LED_PIN, HIGH);

      break;
  }
}


// =====================================================
// ARDUINO SETUP
// =====================================================

void setup()
{
  // ---------------------------------------------------
  // LED
  // ---------------------------------------------------

  pinMode(LED_PIN, OUTPUT);

  digitalWrite(LED_PIN, LOW);


  // ---------------------------------------------------
  // Serial
  // ---------------------------------------------------

  Serial.begin(115200);

  delay(1500);


  // ---------------------------------------------------
  // Startup messages
  // ---------------------------------------------------

  Serial.println();
  Serial.println("==============================");
  Serial.println(" ESP32 GESTURE TINYML");
  Serial.println("==============================");


  // ---------------------------------------------------
  // Initialize TinyML
  // ---------------------------------------------------

  setupTinyML();
}


// =====================================================
// ARDUINO LOOP
// =====================================================

void loop()
{
  // Keep LED blinking without blocking
  updateLED();


  while (Serial.available())
  {
    char c = (char)Serial.read();


    // -------------------------------------------------
    // End of feature line
    // -------------------------------------------------

    if (c == '\n')
    {
      int n = 0;


      // ------------------------------------------------
      // Convert Arduino String to char array
      // ------------------------------------------------

      char buffer[900];

      line.toCharArray(
        buffer,
        sizeof(buffer)
      );


      // ------------------------------------------------
      // Split comma-separated values
      // ------------------------------------------------

      char* token = strtok(
        buffer,
        ","
      );


      while (
        token != nullptr &&
        n < kInputSize
      )
      {
        features[n] = atof(token);

        n++;

        token = strtok(
          nullptr,
          ","
        );
      }


      // ------------------------------------------------
      // Check whether we received all 63 features
      // ------------------------------------------------

      if (n == kInputSize)
      {
        performGesture();
      }


      // Clear line

      line = "";
    }


    // -------------------------------------------------
    // Ignore carriage return
    // -------------------------------------------------

    else if (c != '\r')
    {
      if (line.length() < 899)
      {
        line += c;
      }
      else
      {
        // Prevent buffer overflow

        line = "";
      }
    }
  }
}