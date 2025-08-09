// STM32 Nucleo Motor Control Firmware
// Compatible with the existing motor control commands

#include <Arduino.h>  // STM32 Arduino core

// Motor A pins
const int IN1 = D2;  // PA9 
const int IN2 = D3;  // PA10 
const int ENA = D6;  // PA7 (PWM) 

// Motor B pins
const int IN3 = D4;  // PA0 
const int IN4 = D5;  // PA1 
const int ENB = D9;  // PB1 (PWM)

// Motor speeds (0-1023)
int speedA = 512;
int speedB = 512;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("STM32 Nucleo Motor Controller Ready");
  
  // Configure motor control pins
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  
  // Initialize motors to stopped state
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    char motor = command.charAt(0);
    char action = command.charAt(1);
    
    // Process speed commands (e.g., "SA512" or "SB1023")
    if (motor == 'S') {
      motor = command.charAt(1);
      int speed = command.substring(2).toInt();
      
      // Constrain speed to valid range
      speed = constrain(speed, 0, 1023);
      
      // Map 0-1023 to 0-255 for analogWrite
      int pwmSpeed = map(speed, 0, 1023, 0, 255);
      
      if (motor == 'A') {
        speedA = speed;
        analogWrite(ENA, pwmSpeed);
        Serial.print("Motor A speed set to: ");
        Serial.println(speed);
      } else if (motor == 'B') {
        speedB = speed;
        analogWrite(ENB, pwmSpeed);
        Serial.print("Motor B speed set to: ");
        Serial.println(speed);
      }
      return;
    }
    
    // Process direction commands
    int pwmSpeed = 0;
    
    if (motor == 'A') {
      pwmSpeed = map(speedA, 0, 1023, 0, 255);
      
      if (action == '0') { // Stop
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, LOW);
        analogWrite(ENA, 0);
        Serial.println("Motor A: Stop");
      } else if (action == '1') { // Forward
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
        analogWrite(ENA, pwmSpeed);
        Serial.println("Motor A: Forward");
      } else if (action == '2') { // Backward
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, HIGH);
        analogWrite(ENA, pwmSpeed);
        Serial.println("Motor A: Backward");
      }
    } else if (motor == 'B') {
      pwmSpeed = map(speedB, 0, 1023, 0, 255);
      
      if (action == '0') { // Stop
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, LOW);
        analogWrite(ENB, 0);
        Serial.println("Motor B: Stop");
      } else if (action == '1') { // Forward
        digitalWrite(IN3, HIGH);
        digitalWrite(IN4, LOW);
        analogWrite(ENB, pwmSpeed);
        Serial.println("Motor B: Forward");
      } else if (action == '2') { // Backward
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, HIGH);
        analogWrite(ENB, pwmSpeed);
        Serial.println("Motor B: Backward");
      }
    }
  }
}

