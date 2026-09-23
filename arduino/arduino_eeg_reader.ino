/*
  arduino_eeg_reader.ino

  Reads the BioAmp EXG Pill's analog output and streams raw ADC values
  over serial, one integer per line, paced at 250Hz -- exactly the format
  eeg_source.py's ArduinoEEGSource expects:

      523
      527
      519
      530
      ...

  Wiring: BioAmp signal output -> Arduino A0
          BioAmp GND -> Arduino GND
          BioAmp VCC -> Arduino 5V (or 3.3V, check your board's pinout)

  Set config.ARDUINO_BAUD_RATE to match BAUD_RATE below (115200 by default
  on the Python side already).
*/

const int EEG_PIN = A0;
const unsigned long SAMPLE_RATE_HZ = 250;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;  // 4000us
const long BAUD_RATE = 115200;

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  analogReadResolution(10);  // no-op on boards without this call (e.g. Uno); safe to leave in
}

void loop() {
  unsigned long now = micros();

  // micros() wraps around after ~70 minutes -- this subtraction is
  // unsigned-safe across that wraparound, unlike a plain now > next check
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = now;
    int value = analogRead(EEG_PIN);
    Serial.println(value);
  }
}
