// Motor A
const uint8_t IN1 = D1;  // GPIO5
const uint8_t IN2 = D2;  // GPIO4
const uint8_t ENA = D3;  // GPIO0

// Motor B
const uint8_t IN3 = D5;  // GPIO14
const uint8_t IN4 = D6;  // GPIO12
const uint8_t ENB = D7;  // GPIO13

// Default PWM speeds. Many small DC motors won't move at very low duty
// cycles, so start with a mid-range value that can be tuned from the host
// using the `SA`/`SB` commands.
int speedA = 512;
int speedB = 512;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);
  Serial.begin(115200);  // Typical baud rate for ESP8266
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("A")) {
      if (cmd == "A1") {
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
        analogWrite(ENA, speedA);
      } else if (cmd == "A2") {
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, HIGH);
        analogWrite(ENA, speedA);
      } else if (cmd == "A0") {
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, LOW);
        analogWrite(ENA, 0);
      }
    }

    if (cmd.startsWith("B")) {
      if (cmd == "B1") {
        digitalWrite(IN3, HIGH);
        digitalWrite(IN4, LOW);
        analogWrite(ENB, speedB);
      } else if (cmd == "B2") {
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, HIGH);
        analogWrite(ENB, speedB);
      } else if (cmd == "B0") {
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, LOW);
        analogWrite(ENB, 0);
      }
    }

    if (cmd.startsWith("SA")) {
      speedA = constrain(cmd.substring(2).toInt(), 0, 1023);  // ESP8266 PWM range
      analogWrite(ENA, speedA);
    }

    if (cmd.startsWith("SB")) {
      speedB = constrain(cmd.substring(2).toInt(), 0, 1023);
      analogWrite(ENB, speedB);
    }
  }
}

