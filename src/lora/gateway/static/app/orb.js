/** Mobile pearl with surface waves and outward thought ripples, rendered on 2D canvas. */
class HydroOrbVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.currentState = 'idle'; this.audioLevel = 0; this.running = false;
    this.animationFrameId = null; this.lastFrame = 0;
    this.reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.dark = window.matchMedia('(prefers-color-scheme: dark)');
    this.frame = now => {
      if (!this.canAnimate()) { this.animationFrameId = null; return; }
      if (now - this.lastFrame >= 33) { this.lastFrame = now; this.draw(now / 1000); }
      this.animationFrameId = requestAnimationFrame(this.frame);
    };
    this.refresh = () => {
      if (this.animationFrameId !== null) cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null; this.draw(0);
      if (this.canAnimate()) this.animationFrameId = requestAnimationFrame(this.frame);
    };
    document.addEventListener('visibilitychange', this.refresh);
    this.reduced.addEventListener('change', this.refresh); this.dark.addEventListener('change', this.refresh);
    this.resizeObserver = new ResizeObserver(this.refresh); this.resizeObserver.observe(this.canvas);
    this.draw(0);
  }
  canAnimate() { return this.ctx && this.running && !document.hidden && !this.reduced.matches && !['sleep','sleeping','off','disconnected'].includes(this.currentState); }
  setState(state) { this.currentState = state; this.canvas.dataset.state = state; this.refresh(); }
  setAudioLevel(value) { this.audioLevel = Math.max(0, Math.min(1, value)); }
  start() { if (this.running) return; this.running = true; this.refresh(); }
  stop() { this.running = false; if (this.animationFrameId !== null) cancelAnimationFrame(this.animationFrameId); this.animationFrameId = null; }
  destroy() { this.stop(); document.removeEventListener('visibilitychange', this.refresh); this.reduced.removeEventListener('change', this.refresh); this.dark.removeEventListener('change', this.refresh); this.resizeObserver.disconnect(); }
  outline(radius, time, amplitude) {
    const c = this.ctx;
    c.beginPath();
    for (let step = 0; step <= 80; step++) {
      const angle = step / 80 * Math.PI * 2;
      const wave = amplitude * (Math.sin(angle * 3 + time * 3.1) + .45 * Math.sin(angle * 5 - time * 2.3));
      const r = radius + wave;
      const x = 120 + Math.cos(angle) * r, y = 116 + Math.sin(angle) * r;
      if (step === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.closePath();
  }
  draw(time) {
    if (!this.ctx) return;
    const size = this.canvas.clientWidth || 124, dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    const px = Math.round(size * dpr);
    if (this.canvas.width !== px || this.canvas.height !== px) { this.canvas.width = px; this.canvas.height = px; }
    const c = this.ctx, dark = this.dark.matches;
    c.setTransform(px / 240, 0, 0, px / 240, 0, 0); c.clearRect(0, 0, 240, 240);
    const quiet = this.reduced.matches || ['sleep','sleeping','off','disconnected'].includes(this.currentState);
    const active = ['thinking','verifying'].includes(this.currentState);
    const speaking = this.currentState === 'speaking' && !quiet;
    const phase = quiet ? 0 : Math.sin(time * (active ? .65 : .32));
    const radius = 81 + (quiet ? 0 : phase * .65);
    // Browser speech has no PCM meter: this is a speaking animation, not an audio waveform.
    const motion = speaking ? 2.8 + this.audioLevel * 1.4 : 0;
    const muted = ['sleep','sleeping','off','disconnected'].includes(this.currentState);
    const base = muted ? '145,151,145' : this.currentState === 'verifying' ? '170,151,116' : '125,155,141';
    if (active && !quiet) {
      for (let ring = 0; ring < 3; ring++) {
        const progress = (time * .32 + ring / 3) % 1;
        c.beginPath(); c.ellipse(120, 116, 84 + progress * 30, 81 + progress * 27, 0, 0, Math.PI * 2);
        c.strokeStyle = `rgba(${base},${(1 - progress) * .45})`;
        c.lineWidth = 1.5; c.stroke();
      }
    }
    c.save(); c.translate(121, 213); c.scale(1, .14);
    const shadow = c.createRadialGradient(0,0,0,0,0,72);shadow.addColorStop(0,dark?'#0004':'#304a3b24');shadow.addColorStop(1,'#0000');
    c.fillStyle=shadow;c.beginPath();c.arc(0,0,72,0,Math.PI*2);c.fill();c.restore();
    c.save(); this.outline(radius, time, motion); c.clip();
    const body=c.createRadialGradient(88,71,3,126,124,107);
    body.addColorStop(0,dark?'#e3edde':'#ffffff');body.addColorStop(.32,dark?'#a4b9a4':'#e7eee2');body.addColorStop(.70,`rgba(${base},.76)`);body.addColorStop(1,dark?'#314e43':'#849c90');
    c.fillStyle=body;c.fillRect(25,20,195,195);
    c.save();c.translate(120,116);c.rotate(-.48 + phase * .018 + (speaking ? Math.sin(time * 2) * .11 : 0));
    const crescent=c.createLinearGradient(-65,-30,55,66);crescent.addColorStop(0,dark?'#f3f6dfbb':'#fffef0dd');crescent.addColorStop(.48,'#e5ecd133');crescent.addColorStop(1,`rgba(${base},.1)`);
    c.fillStyle=crescent;c.beginPath();c.ellipse(0,6,91,39,0,0,Math.PI*2);c.fill();
    c.strokeStyle=dark?'#f4f8e54d':'#ffffffe0';c.lineWidth=.8;c.beginPath();c.ellipse(0,6,91,39,0,Math.PI,Math.PI*2);c.stroke();c.restore();
    if ((active || speaking) && !quiet) {
      for (let wave = 0; wave < 3; wave++) {
        c.beginPath();
        for (let x = 31; x <= 210; x += 3) {
          const y = 103 + wave * 17 + Math.sin(x * .038 + time * (speaking ? 3.4 : .8) + wave) * (speaking ? 9 : 3);
          if (x === 31) c.moveTo(x, y); else c.lineTo(x, y);
        }
        c.strokeStyle = dark ? '#f0f8e655' : '#ffffffa0'; c.lineWidth = speaking ? 1.8 : 1; c.stroke();
      }
    }
    const veil=c.createRadialGradient(96,65,0,96,65,67);veil.addColorStop(0,'#ffffff9c');veil.addColorStop(1,'#ffffff00');c.fillStyle=veil;c.fillRect(20,20,200,200);
    c.restore(); this.outline(radius, time, motion);c.strokeStyle=dark?'#e4eedd80':'#e4eee9';c.lineWidth=1.1;c.stroke();
    c.beginPath();c.arc(120,116,radius-3,3.75,5.05);c.strokeStyle='#ffffffb0';c.lineWidth=1.4;c.stroke();
  }
}
