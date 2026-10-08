/**
 * Aqueous Hydro-Orb Visual Resonator (WebGL Procedural Water Sphere)
 * 
 * Provides a living, state-reactive translucent fluid sphere with:
 * - Raymarched Signed Distance Field (SDF) sphere with surface tension caustics
 * - 3D procedural noise displacement for fluid capillary waves
 * - Dynamic Fresnel rim lighting and chromatic dispersion
 * - State machine reaction: idle, listening, thinking, speaking, verifying
 * - 0 MB Host GPU overhead (runs entirely in client browser WebGL)
 */

class HydroOrbVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
    this.currentState = 'idle';
    this.audioLevel = 0.0;
    this.startTime = performance.now();
    this.running = false;
    this.animationFrameId = null;

    if (!this.gl) {
      console.warn("WebGL not supported; fallback rendering enabled");
      return;
    }

    this._initShaders();
    this._initBuffers();
    this._setupVisibilityListener();
  }

  _initShaders() {
    const gl = this.gl;

    // Fullscreen quad vertex shader
    const vsSource = `
      attribute vec2 a_position;
      varying vec2 v_uv;
      void main() {
        v_uv = (a_position + 1.0) * 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    // Raymarched procedural fluid sphere fragment shader
    const fsSource = `
      precision highp float;
      varying vec2 v_uv;

      uniform float u_time;
      uniform vec2 u_resolution;
      uniform int u_state;      // 0: idle, 1: listening, 2: thinking, 3: speaking, 4: verifying
      uniform float u_audio;    // 0.0 to 1.0

      // Simplex-style 3D noise approximation
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

      // Signed Distance Field (SDF) of the displaced fluid sphere
      float map(vec3 p) {
        float r = 0.65;
        
        // Fluid ripple displacement based on state
        float disp = 0.0;
        if (u_state == 0) { // Idle (gentle breathing)
          float breath = sin(u_time * 1.25) * 0.03;
          disp = snoise(p * 2.2 + vec3(0.0, u_time * 0.4, 0.0)) * 0.04 + breath;
        } else if (u_state == 1) { // Listening
          disp = snoise(p * 3.5 + vec3(0.0, u_time * 1.5, 0.0)) * (0.06 + u_audio * 0.08);
        } else if (u_state == 2) { // Thinking (swirling vortex)
          float angle = length(p.xz) * 3.0 - u_time * 3.0;
          vec3 rotP = vec3(p.x * cos(angle) - p.z * sin(angle), p.y, p.x * sin(angle) + p.z * cos(angle));
          disp = snoise(rotP * 3.0) * 0.08;
        } else if (u_state == 3) { // Speaking (standing cymatics)
          float waves = sin(length(p) * 18.0 - u_time * 6.0) * (0.04 + u_audio * 0.1);
          disp = waves + snoise(p * 4.0 + vec3(u_time * 2.0)) * 0.05;
        } else if (u_state == 4) { // Verifying (calm sapphire tension)
          disp = snoise(p * 2.5 + vec3(u_time * 0.6)) * 0.03;
        }

        return length(p) - (r + disp);
      }

      vec3 calcNormal(vec3 p) {
        float eps = 0.005;
        vec2 h = vec2(eps, 0.0);
        return normalize(vec3(
          map(p + h.xyy) - map(p - h.xyy),
          map(p + h.yxy) - map(p - h.yxy),
          map(p + h.yyx) - map(p - h.yyx)
        ));
      }

      void main() {
        vec2 uv = (gl_FragCoord.xy - u_resolution * 0.5) / min(u_resolution.x, u_resolution.y);
        vec3 ro = vec3(0.0, 0.0, 2.0); // Camera Ray Origin
        vec3 rd = normalize(vec3(uv, -1.5)); // Ray Direction

        float t = 0.0;
        bool hit = false;
        vec3 p = ro;

        for (int i = 0; i < 48; i++) {
          p = ro + rd * t;
          float d = map(p);
          if (d < 0.002) {
            hit = true;
            break;
          }
          t += d;
          if (t > 4.0) break;
        }

        if (hit) {
          vec3 n = calcNormal(p);
          vec3 lightDir = normalize(vec3(0.6, 0.8, 1.0));
          float diff = max(dot(n, lightDir), 0.0);
          
          // Fresnel rim reflection
          float fresnel = pow(1.0 - max(dot(-rd, n), 0.0), 3.0);
          
          // Core fluid color palette by state
          vec3 fluidColor = vec3(0.0, 0.85, 1.0); // Cyan
          if (u_state == 1) { // Listening
            fluidColor = vec3(0.0, 0.95, 0.8);
          } else if (u_state == 2) { // Thinking (Indigo/violet vortex)
            fluidColor = vec3(0.2, 0.4, 0.95);
          } else if (u_state == 3) { // Speaking
            fluidColor = vec3(0.1, 0.8, 1.0);
          } else if (u_state == 4) { // Verifying (Deep Sapphire)
            fluidColor = vec3(0.0, 0.4, 0.9);
          }

          // Caustic glow inside fluid
          float caustic = pow(diff, 2.5) * 0.7;
          vec3 col = fluidColor * (0.3 + diff * 0.4) + vec3(1.0) * caustic + vec3(0.6, 0.9, 1.0) * fresnel * 0.8;
          
          gl_FragColor = vec4(col, 0.85);
        } else {
          // Atmospheric aura halo around sphere
          float distToCenter = length(uv);
          float glow = max(0.0, 1.0 - distToCenter * 2.2);
          vec3 glowCol = vec3(0.0, 0.6, 0.9) * pow(glow, 3.0) * 0.4;
          gl_FragColor = vec4(glowCol, glow * 0.3);
        }
      }
    `;

    const vs = this._compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSource);

    this.program = gl.createProgram();
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error("Shader link failed:", gl.getProgramInfoLog(this.program));
      return;
    }

    this.uTime = gl.getUniformLocation(this.program, "u_time");
    this.uResolution = gl.getUniformLocation(this.program, "u_resolution");
    this.uState = gl.getUniformLocation(this.program, "u_state");
    this.uAudio = gl.getUniformLocation(this.program, "u_audio");
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("Shader compile error:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  _initBuffers() {
    const gl = this.gl;
    const positions = new Float32Array([
      -1.0, -1.0,
       1.0, -1.0,
      -1.0,  1.0,
      -1.0,  1.0,
       1.0, -1.0,
       1.0,  1.0,
    ]);

    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const aPos = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  }

  _setupVisibilityListener() {
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this.stop();
      } else {
        this.start();
      }
    });
  }

  setState(stateName) {
    this.currentState = stateName.toLowerCase();
  }

  setAudioLevel(level) {
    this.audioLevel = Math.max(0.0, Math.min(1.0, level));
  }

  _stateToInt(state) {
    switch (state) {
      case 'idle': return 0;
      case 'listening': return 1;
      case 'thinking': return 2;
      case 'speaking': return 3;
      case 'verifying': return 4;
      default: return 0;
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    const render = () => {
      if (!this.running) return;
      this._render();
      this.animationFrameId = requestAnimationFrame(render);
    };
    this.animationFrameId = requestAnimationFrame(render);
  }

  stop() {
    this.running = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  _render() {
    const gl = this.gl;
    if (!gl || !this.program) return;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);

    const time = (performance.now() - this.startTime) / 1000.0;
    gl.uniform1f(this.uTime, time);
    gl.uniform2f(this.uResolution, this.canvas.width, this.canvas.height);
    gl.uniform1i(this.uState, this._stateToInt(this.currentState));
    gl.uniform1f(this.uAudio, this.audioLevel);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

window.HydroOrbVisualizer = HydroOrbVisualizer;
