#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

// ---------- WiFi ----------
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";
const uint16_t UDP_PORT = PORT;
WiFiUDP udp;

// === ADDED: where to send telemetry back ===
IPAddress controllerIP;
uint16_t  controllerPort = 0;
bool      havePeer = false;

// ---------- IO ----------
const int LED_LEFT  = 23; // left wheel sim
const int LED_RIGHT = 21; // right wheel sim

const int PIN_TRIG = 5;
const int PIN_ECHO = 18;
km,n m m
// ---------- Control & safety ----------
const float WALL_THRESH_CM = 15.0;   // max allowed distance to wall
const float DEADMAN_MS     = 500;    // stop if no command in this time
unsigned long lastCmdMs    = 0;

// Robot “intent” from laptop (v, omega). v<0 means retreat.
float cmd_v    = 0.0f;
float cmd_omega= 0.0f;

// ---------- Wall behavior state machine ----------
enum State { IDLE, RETREAT, WALL_GLIDE, BOUNCE };
State state = IDLE;

unsigned long phaseStartMs = 0;
int turnDir = 1;             // +1 or -1
const unsigned long GLIDE_MS  = 800;
const unsigned long BOUNCE_MS = 400;

// === ADDED: telemetry rate control ===
const unsigned long TELEMETRY_PERIOD_MS = 100;  // 10 Hz
unsigned long lastTelemMs = 0;

// ---------- Helpers ----------
float readUltrasonicCM() {
  digitalWrite(PIN_TRIG, LOW); delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  long duration = pulseIn(PIN_ECHO, HIGH, 30000);
  if (duration <= 0) return 9999.0f;
  return duration / 58.0f; // cm
}

void driveSim(float v, float omega) {
  if (v < -0.01f && fabs(omega) < 0.05f) {
    digitalWrite(LED_LEFT, HIGH);
    digitalWrite(LED_RIGHT, HIGH);
  } else if (v > 0.01f && fabs(omega) < 0.05f) {
    digitalWrite(LED_LEFT, HIGH);
    digitalWrite(LED_RIGHT, HIGH);
  } else if (fabs(omega) >= 0.05f) {
    if (omega > 0) { // yaw left
      digitalWrite(LED_LEFT, LOW);
      digitalWrite(LED_RIGHT, HIGH);
    } else {          // yaw right
      digitalWrite(LED_LEFT, HIGH);
      digitalWrite(LED_RIGHT, LOW);
    }
  } else {
    digitalWrite(LED_LEFT, LOW);
    digitalWrite(LED_RIGHT, LOW);
  }
}

void stopSim() {
  digitalWrite(LED_LEFT, LOW);
  digitalWrite(LED_RIGHT, LOW);
}

// === ADDED: stringify state for telemetry ===
const char* stateName(State s) {
  switch (s) {
    case IDLE: return "IDLE";
    case RETREAT: return "RETREAT";
    case WALL_GLIDE: return "WALL_GLIDE";
    case BOUNCE: return "BOUNCE";
    default: return "UNKNOWN";
  }
}

// === ADDED: send telemetry if we know who to reply to ===
void sendTelemetry(float distCm) {
  if (!havePeer) return;
  if (millis() - lastTelemMs < TELEMETRY_PERIOD_MS) return;

  StaticJsonDocument<256> d;
  d["type"] = "telemetry";
  d["dist_cm"] = distCm;
  d["state"] = stateName(state);
  d["v"] = cmd_v;
  d["omega"] = cmd_omega;
  d["lastCmdAge_ms"] = (uint32_t)(millis() - lastCmdMs);
  d["rssi"] = WiFi.RSSI();
  d["uptime_ms"] = (uint32_t)millis();

  udp.beginPacket(controllerIP, controllerPort);
  serializeJson(d, udp);
  udp.endPacket();

  lastTelemMs = millis();
}

// ---------- Setup ----------
void setup() {
  pinMode(LED_LEFT, OUTPUT);
  pinMode(LED_RIGHT, OUTPUT);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println("\nWiFi OK, IP: " + WiFi.localIP().toString());
  udp.begin(UDP_PORT);
  Serial.printf("UDP listening on %u\n", UDP_PORT);

  state = IDLE;
  lastCmdMs = millis();
}

// ---------- Main loop ----------
void loop() {
  // 1) Receive UDP JSON (non-blocking)
  int pktSize = udp.parsePacket();
  if (pktSize > 0) {
    // === ADDED: remember who sent this, so we can reply
    controllerIP = udp.remoteIP();
    controllerPort = udp.remotePort();
    havePeer = true;

    char buf[256];
    int n = udp.read(buf, sizeof(buf) - 1);
    buf[n] = 0;
    StaticJsonDocument<192> d;
    if (deserializeJson(d, buf) == DeserializationError::Ok) {
      const char* cmd = d["cmd"] | "";
      if (!strcmp(cmd, "drive")) {
        cmd_v     = d["v"]    | 0.0;
        cmd_omega = d["omega"]| 0.0;
        lastCmdMs = millis();

        if (cmd_v < -0.01f) state = RETREAT;
        else if (fabs(cmd_v) < 0.01f && fabs(cmd_omega) < 0.01f) state = IDLE;
        else state = IDLE; // MVP: only RETREAT or IDLE
      } else if (!strcmp(cmd, "stop")) {
        cmd_v = 0; cmd_omega = 0; state = IDLE; stopSim();
        lastCmdMs = millis();
      }
    }
  }

  // 2) Dead-man stop
  if (millis() - lastCmdMs > DEADMAN_MS) {
    state = IDLE; cmd_v = 0; cmd_omega = 0;
  }

  // 3) Read ultrasonic (fast)
  float dist = readUltrasonicCM();

  // 4) State machine for “glide → bounce” when too close to wall
  switch (state) {
    case IDLE:
      stopSim();
      break;

    case RETREAT:
      if (dist < WALL_THRESH_CM) {
        turnDir = (turnDir > 0) ? -1 : 1;
        phaseStartMs = millis();
        state = WALL_GLIDE;
      } else {
        driveSim(-0.2f, 0.0f);
      }
      break;

    case WALL_GLIDE: {
      float yaw = 0.8f * (float)turnDir;
      driveSim(0.0f, yaw);
      if (millis() - phaseStartMs > 250) {
        phaseStartMs = millis();
        state = BOUNCE;
      }
      break;
    }

    case BOUNCE: {
      unsigned long dt = millis() - phaseStartMs;
      if (dt <= GLIDE_MS) {
        driveSim(-0.15f, 0.25f * (float)turnDir);
      } else if (dt <= GLIDE_MS + BOUNCE_MS) {
        driveSim(0.0f, -0.8f * (float)turnDir);
      } else {
        state = RETREAT;
      }
      break;
    }
  }

  // === ADDED: send telemetry after state & outputs are updated ===
  sendTelemetry(dist);
}