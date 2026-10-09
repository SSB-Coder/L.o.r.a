"""3D Translucent Water Droplet Visualizer with Holographic Resonance Rings.

Renders a living, crystal-clear 3D spherical water droplet with:
- Physical water refraction (IOR 1.333) and chromatic dispersion
- Fresnel rim reflections (Schlick approximation)
- Crisp specular light glints and internal caustics
- Audio-reactive 3D surface capillary ripples
- Concentric holographic Arc Reactor telemetry rings
- Hardware-accelerated WebGL with headless QPainter fallback
"""

import logging
import math
from typing import override

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

DROPLET_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body, html { width: 100%; height: 100%; overflow: hidden; background: transparent; }
  canvas { display: block; width: 100%; height: 100%; }
</style>
</head>
<body>
<canvas id="dropletCanvas"></canvas>
<script>
(function() {
  const canvas = document.getElementById('dropletCanvas');
  const gl = canvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: false });
  if (!gl) return;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener('resize', resize);
  resize();

  const vs = `
    attribute vec2 position;
    varying vec2 v_uv;
    void main() {
      v_uv = (position + 1.0) * 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fs = `
    precision highp float;
    varying vec2 v_uv;

    uniform float u_time;
    uniform vec2 u_resolution;
    uniform int u_state;      // 0: idle, 1: listening, 2: thinking, 3: speaking, 4: verifying
    uniform float u_audio;    // 0.0 to 1.0
    uniform float u_vscore;   // 0.0 to 1.0

    // Simplex 3D noise approximation
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

    float snoise(vec3 v) {
      const vec2 C = vec2(1.0/6.0, 1.0/3.0);
      const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
      vec3 i  = floor(v + dot(v, C.yyy));
      vec3 x0 = v - i + dot(i, C.xxx);
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min(g.xyz, l.zxy);
      vec3 i2 = max(g.xyz, l.zxy);
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy;
      vec3 x3 = x0 - D.yyy;
      i = mod289(i);
      vec4 p = permute(permute(permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0));
      float n_ = 0.142857142857;
      vec3  ns = n_ * D.wyz - D.xzx;
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_);
      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      vec4 b0 = vec4(x.xy, y.xy);
      vec4 b1 = vec4(x.zw, y.zw);
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
      vec3 p0 = vec3(a0.xy, h.x);
      vec3 p1 = vec3(a0.zw, h.y);
      vec3 p2 = vec3(a1.xy, h.z);
      vec3 p3 = vec3(a1.zw, h.w);
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
      p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }

    void main() {
      vec2 uv = (gl_FragCoord.xy - u_resolution * 0.5) / min(u_resolution.x, u_resolution.y);
      float distToCenter = length(uv);

      // Ray setup for 3D sphere raytracing
      vec3 ro = vec3(0.0, 0.0, 2.2);
      vec3 rd = normalize(vec3(uv, -1.8));

      // Base droplet radius
      float R = 0.52;

      // Gentle surface breathing displacement
      float breath = sin(u_time * 1.5) * 0.015;
      float noiseDisp = snoise(vec3(uv * 2.8, u_time * 0.4)) * 0.02;
      float activeR = R + breath + noiseDisp;

      // Ray-Sphere intersection
      float b = dot(ro, rd);
      float c = dot(ro, ro) - activeR * activeR;
      float disc = b * b - c;

      vec4 finalColor = vec4(0.0);

      if (disc > 0.0) {
        float t = -b - sqrt(disc);
        vec3 hitPos = ro + rd * t;

        // Capillary surface ripples driven by voice audio
        float rippleFreq = 26.0;
        float rippleAmp = 0.01 + u_audio * 0.06;
        if (u_state == 2) { rippleFreq = 38.0; rippleAmp = 0.035; } // Thinking vortex
        if (u_state == 3) { rippleFreq = 32.0; rippleAmp = 0.025 + u_audio * 0.08; } // Speaking cymatics

        float rip = sin(length(hitPos.xy) * rippleFreq - u_time * 6.0) * rippleAmp;

        // Calculated surface normal with liquid ripple perturbation
        vec3 norm = normalize(hitPos);
        norm.x += cos(hitPos.x * rippleFreq - u_time * 6.0) * rippleAmp * 1.5;
        norm.y += sin(hitPos.y * rippleFreq - u_time * 6.0) * rippleAmp * 1.5;
        norm = normalize(norm);

        // Water Optics: Schlick's Fresnel reflection
        // IOR = 1.333 for water, F0 = ((1 - 1.333)/(1 + 1.333))^2 ~ 0.02
        float cosTheta = clamp(dot(-rd, norm), 0.0, 1.0);
        float fresnel = 0.03 + 0.97 * pow(1.0 - cosTheta, 4.0);

        // Chromatic Refraction through translucent water
        vec3 refrR = refract(rd, norm, 1.0 / 1.328);
        vec3 refrG = refract(rd, norm, 1.0 / 1.333);
        vec3 refrB = refract(rd, norm, 1.0 / 1.338);

        // Translucent liquid interior: clear water with cyan/azure depth tint
        vec3 waterBaseColor = vec3(0.02, 0.08, 0.16); // Deep clear ocean back
        if (u_state == 1) waterBaseColor = vec3(0.02, 0.12, 0.18); // Listening aqua
        if (u_state == 2) waterBaseColor = vec3(0.08, 0.05, 0.22); // Thinking violet/indigo
        if (u_state == 3) waterBaseColor = vec3(0.02, 0.14, 0.20); // Speaking bright cyan

        // Refracted background luminosity
        vec3 refrColor = waterBaseColor + vec3(
          length(refrR.xy) * 0.25,
          length(refrG.xy) * 0.45,
          length(refrB.xy) * 0.65
        ) * 0.4;

        // Internal caustic focal glow
        float caustic = pow(max(dot(refrG, vec3(0.0, 0.0, -1.0)), 0.0), 3.0) * 0.55;

        // Specular glints (Key light + secondary rim fill)
        vec3 keyLight = normalize(vec3(0.9, 1.4, 1.6));
        vec3 h1 = normalize(keyLight - rd);
        float spec1 = pow(max(dot(norm, h1), 0.0), 180.0) * 2.2; // Pinpoint sun glint

        vec3 fillLight = normalize(vec3(-1.2, -0.8, 1.0));
        vec3 h2 = normalize(fillLight - rd);
        float spec2 = pow(max(dot(norm, h2), 0.0), 50.0) * 0.5;

        // Fresnel Rim Glow color based on state
        vec3 rimGlowColor = vec3(0.4, 0.85, 1.0); // Translucent electric cyan
        if (u_state == 1) rimGlowColor = vec3(0.3, 1.0, 0.85); // Emerald aqua
        if (u_state == 2) rimGlowColor = vec3(0.6, 0.5, 1.0); // Indigo
        if (u_state == 3) rimGlowColor = vec3(0.45, 0.95, 1.0); // Spoken cyan
        if (u_state == 4) rimGlowColor = vec3(0.2, 0.7, 1.0); // Verifying

        // Compose realistic clear translucent water droplet
        vec3 dropletCol = mix(refrColor, rimGlowColor, fresnel * 0.85);
        dropletCol += vec3(1.0) * spec1; // Pure white specular gleam
        dropletCol += rimGlowColor * spec2;
        dropletCol += rimGlowColor * caustic;

        // Translucent alpha: center is clear and see-through, edge is glassy
        float alpha = clamp(mix(0.28, 0.92, fresnel) + spec1 * 0.8, 0.0, 1.0);
        finalColor = vec4(dropletCol, alpha);
      }

      // HOLOGRAPHIC CONCENTRIC HUD RINGS (Inspired by JARVIS / Arc Reactor reference)
      // Ring 1: Voice Resonance Arc (dist 0.64 - 0.66)
      float r1Dist = abs(distToCenter - 0.65);
      if (r1Dist < 0.02) {
        float angle = atan(uv.y, uv.x);
        float waveRes = sin(angle * 12.0 - u_time * 5.0) * (0.01 + u_audio * 0.04);
        float arcAlpha = smoothstep(0.02, 0.0, r1Dist + waveRes) * (0.4 + u_audio * 0.6);
        vec3 arcColor = mix(vec3(0.1, 0.7, 0.9), vec3(0.4, 0.95, 1.0), u_audio);
        finalColor = mix(finalColor, vec4(arcColor, 1.0), arcAlpha * 0.75);
      }

      // Ring 2: Segmented Radar Tick Ring (dist 0.75 - 0.76)
      float r2Dist = abs(distToCenter - 0.76);
      if (r2Dist < 0.008) {
        float angle2 = atan(uv.y, uv.x) + u_time * 0.2;
        float seg = step(0.4, fract(angle2 * 5.729 / 1.5)); // Tick segments
        float tickAlpha = smoothstep(0.008, 0.0, r2Dist) * seg * 0.35;
        finalColor = mix(finalColor, vec4(0.2, 0.8, 1.0, 1.0), tickAlpha);
      }

      // Ring 3: Verification & Grounding Confidence Arc (dist 0.84 - 0.86)
      float r3Dist = abs(distToCenter - 0.85);
      if (r3Dist < 0.012) {
        float angle3 = (atan(uv.y, uv.x) + 3.14159) / (2.0 * 3.14159); // 0.0 to 1.0
        float filled = step(angle3, clamp(u_vscore, 0.1, 1.0));
        float meterAlpha = smoothstep(0.012, 0.0, r3Dist) * filled * 0.65;
        vec3 meterColor = vec3(0.1, 0.9, 0.7); // Grounded emerald/cyan
        finalColor = mix(finalColor, vec4(meterColor, 1.0), meterAlpha);
      }

      // Soft ambient aura around whole visualizer
      float aura = max(0.0, 1.0 - distToCenter * 1.8);
      vec3 auraCol = vec3(0.02, 0.45, 0.8) * pow(aura, 3.5) * (0.25 + u_audio * 0.35);
      finalColor += vec4(auraCol, aura * 0.2);

      gl_FragColor = finalColor;
    }
  `;

  function createShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  const prog = gl.createProgram();
  gl.attachShader(prog, createShader(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, createShader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);

  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(prog));
    return;
  }
  gl.useProgram(prog);

  const quad = new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

  const posAttr = gl.getAttribLocation(prog, 'position');
  gl.enableVertexAttribArray(posAttr);
  gl.vertexAttribPointer(posAttr, 2, gl.FLOAT, false, 0, 0);

  const uTime = gl.getUniformLocation(prog, 'u_time');
  const uRes = gl.getUniformLocation(prog, 'u_resolution');
  const uState = gl.getUniformLocation(prog, 'u_state');
  const uAudio = gl.getUniformLocation(prog, 'u_audio');
  const uVscore = gl.getUniformLocation(prog, 'u_vscore');

  let stateVal = 0;
  let targetAudio = 0.0;
  let curAudio = 0.0;
  let vscoreVal = 0.94; // Default 94% grounding confidence
  const start = performance.now();

  window.setDropletState = function(stateStr, audio, vscore) {
    if (stateStr === 'idle') stateVal = 0;
    else if (stateStr === 'listening') stateVal = 1;
    else if (stateStr === 'thinking' || stateStr === 'acting') stateVal = 2;
    else if (stateStr === 'speaking') stateVal = 3;
    else if (stateStr === 'verifying') stateVal = 4;
    else stateVal = 0;

    if (audio !== undefined && audio !== null) targetAudio = Math.max(0.0, Math.min(1.0, audio));
    if (vscore !== undefined && vscore !== null) vscoreVal = Math.max(0.0, Math.min(1.0, vscore));
  };

  function render() {
    curAudio += (targetAudio - curAudio) * 0.15; // Smooth audio interpolation
    const t = (performance.now() - start) * 0.001;

    gl.uniform1f(uTime, t);
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.uniform1i(uState, stateVal);
    gl.uniform1f(uAudio, curAudio);
    gl.uniform1f(uVscore, vscoreVal);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
})();
</script>
</body>
</html>
"""


class Droplet3DWidget(QWidget):
    """High-fidelity 3D Translucent Water Droplet and Holographic Telemetry Visualizer."""

    def __init__(self, size: int = 260, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._state_str = "idle"
        self._audio_level = 0.0
        self._truthfulness = 0.94
        self._is_active = False
        self._web_view = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Attempt hardware-accelerated QWebEngineView initialization
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView

            self._web_view = QWebEngineView(self)
            self._web_view.page().setBackgroundColor(Qt.GlobalColor.transparent)
            self._web_view.setStyleSheet("background: transparent; border: none;")
            self._web_view.setHtml(DROPLET_HTML)
            layout.addWidget(self._web_view)
            self._use_webgl = True
        except Exception as e:
            logger.warning("WebEngine unavailable for 3D droplet, falling back to QPainter: %s", e)
            self._use_webgl = False
            self._web_view = None

        # Fallback procedural animation timer (for headless/non-WebGL environments)
        self._phase = 0.0
        self._fallback_timer = QTimer(self)
        self._fallback_timer.setInterval(33)
        self._fallback_timer.timeout.connect(self._fallback_animate)

    @property
    def is_active(self) -> bool:
        return self._is_active

    def set_active(self, active: bool) -> None:
        self._is_active = active
        if not self._use_webgl:
            if active and not self._fallback_timer.isActive():
                self._fallback_timer.start()
            elif not active:
                self._fallback_timer.stop()
                self.update()

    def set_state(self, state: str) -> None:
        self._state_str = state.lower()
        self._is_active = self._state_str in ("listening", "thinking", "acting", "speaking", "verifying")
        if self._web_view:
            js = f"if (window.setDropletState) window.setDropletState('{self._state_str}', {self._audio_level}, {self._truthfulness});"
            self._web_view.page().runJavaScript(js)
        else:
            self.set_active(self._is_active)
            self.update()

    def set_audio_level(self, level: float) -> None:
        self._audio_level = max(0.0, min(1.0, level))
        if self._web_view:
            js = f"if (window.setDropletState) window.setDropletState('{self._state_str}', {self._audio_level}, {self._truthfulness});"
            self._web_view.page().runJavaScript(js)
        else:
            self.update()

    def set_truthfulness(self, score: float) -> None:
        self._truthfulness = max(0.0, min(1.0, score))
        if self._web_view:
            js = f"if (window.setDropletState) window.setDropletState('{self._state_str}', {self._audio_level}, {self._truthfulness});"
            self._web_view.page().runJavaScript(js)
        else:
            self.update()

    def _fallback_animate(self) -> None:
        self._phase += 0.08
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        self.update()

    @override
    def paintEvent(self, event) -> None:
        if self._use_webgl:
            return  # WebGL handles rendering via QWebEngineView

        # Pure QPainter translucent water droplet fallback
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        center = QPointF(w / 2.0, h / 2.0)
        radius = min(w, h) / 2.0 - 24.0

        # Translucent glass/water sphere body
        grad = QRadialGradient(center.x() - radius * 0.35, center.y() - radius * 0.35, radius * 1.35)
        grad.setColorAt(0.0, QColor(255, 255, 255, 230))  # Specular sunlight gleam
        grad.setColorAt(0.2, QColor(224, 242, 254, 180))  # Crystalline water
        grad.setColorAt(0.6, QColor(14, 165, 233, 160))   # Aquatic azure
        grad.setColorAt(0.9, QColor(2, 132, 199, 140))    # Deep ocean
        grad.setColorAt(1.0, QColor(56, 189, 248, 210))   # Luminous Fresnel rim
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius, radius)

        # Concentric Holographic Arc Telemetry Rings
        pen = painter.pen()
        pen.setColor(QColor(56, 189, 248, 120))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius + 12, radius + 12)

        if self._is_active:
            pen.setColor(QColor(14, 165, 233, 190))
            painter.setPen(pen)
            painter.drawEllipse(center, radius + 20, radius + 20)
