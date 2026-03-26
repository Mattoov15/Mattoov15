#include <Wire.h>
#include "rgb_lcd.h"

rgb_lcd lcd;

// -------------------- PARAMETRES MESURE --------------------
float VREF = 5.0;

// Pont diviseur (tension)
float R_HAUT = 100000.0;
float R_BAS  = 10000.0;

// Courant
float GAIN_AMP = 6.7;
float Rshunt = 0.6;

// -------------------- FILTRAGE ADC --------------------
#define ALPHA_FILTRE 0.15f  // EMA : 0.05=très lissé/lent, 0.3=plus réactif/bruité
#define NB_SAMPLES   16     // Oversampling : divise le bruit par sqrt(NB_SAMPLES)

// -------------------- VARIABLES --------------------
float tension = 0;
float courant = 0;

float alphaPWM = 0.7;   // PWM FIXE
float alphaPot = 0;
float vitesse = 0;

// Filtre courant (EMA)
float courant_filtre = 0;

// Lecture ADC avec oversampling (moyenne de NB_SAMPLES lectures)
float adcOversample(int pin) {
  long sum = 0;
  for (int i = 0; i < NB_SAMPLES; i++) {
    sum += analogRead(pin);
  }
  return (sum / (float)NB_SAMPLES) * VREF / 1023.0;
}

void setup() {

  lcd.begin(16,2);
  lcd.setRGB(0,0,255);

  pinMode(9, OUTPUT);

  // Configuration PWM 20 kHz
  TCCR1A = 0;
  TCCR1B = 0;

  TCCR1A |= (1 << COM1A1);
  TCCR1A |= (1 << WGM11);

  TCCR1B |= (1 << WGM12) | (1 << WGM13);
  TCCR1B |= (1 << CS10);

  ICR1 = 799;

  OCR1A = alphaPWM * ICR1;
}

void loop() {

  // ----------- POTENTIOMETRE -----------
  int pot = analogRead(A0);
  alphaPot = pot / 1023.0;

  // ----------- VITESSE -----------
  vitesse = alphaPWM * 100;

  // ----------- MESURE TENSION (A1) — oversampling -----------
  float vA1 = adcOversample(A1);
  tension = vA1 * (R_HAUT + R_BAS) / R_BAS;

  // ----------- MESURE COURANT (A2) — oversampling + filtre EMA -----------
  float vA2 = adcOversample(A2);
  courant = vA2 / (GAIN_AMP * Rshunt);

  // Filtre passe-bas exponentiel (EMA)
  courant_filtre = ALPHA_FILTRE * courant + (1.0 - ALPHA_FILTRE) * courant_filtre;

  // ----------- AFFICHAGE -----------
  lcd.setCursor(0,0);
  lcd.print("V:");
  lcd.print(tension,1);
  lcd.print(" I:");
  lcd.print(courant_filtre,2);  // valeur filtrée
  lcd.print("   ");

  lcd.setCursor(0,1);
  lcd.print("del:");
  lcd.print(vitesse,0);
  lcd.print("% P:");
  lcd.print(alphaPot*100,0);
  lcd.print("%  ");

  delay(200);
}
