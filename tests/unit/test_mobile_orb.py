"""Execute the mobile renderer lifecycle without browser/package dependencies."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for mobile renderer checks")
def test_mobile_orb_pauses_for_visibility_motion_and_disconnection():
    source = (Path(__file__).resolve().parents[2] / "src/lora/gateway/static/app/orb.js").read_text(encoding="utf-8")
    harness = """
let draws=0,serial=0;const queued=new Map(),events=new Map(),media=new Map();
const requestAnimationFrame=fn=>{queued.set(++serial,fn);return serial};
const cancelAnimationFrame=id=>queued.delete(id);
const context=new Proxy({}, {get:(obj,key)=>key==='clearRect'?()=>draws++:key.startsWith('create')?()=>({addColorStop(){}}):()=>{}});
const canvas={clientWidth:124,width:240,height:240,dataset:{},getContext:type=>{if(type!=='2d')throw Error('Unexpected WebGL');return context}};
const document={hidden:false,getElementById:()=>canvas,addEventListener:(name,fn)=>events.set(name,fn),removeEventListener:name=>events.delete(name)};
const window={devicePixelRatio:3,matchMedia:q=>{
 if(!media.has(q))media.set(q,{matches:false,listeners:new Set(),addEventListener(name,fn){this.listeners.add(fn)},removeEventListener(name,fn){this.listeners.delete(fn)}});
 return media.get(q);
}};
class ResizeObserver{observe(){}disconnect(){}}
function step(time){const entries=[...queued.values()];queued.clear();entries.forEach(fn=>fn(time))}
function assert(value,message){if(!value)throw Error(message)}
"""
    checks = """
const orb=new HydroOrbVisualizer('orb');orb.start();orb.start();assert(queued.size===1,'Duplicate render loop');
const initial=draws;step(34);assert(draws===initial+1,'Active frame missing');
assert(canvas.width<=124*1.75+1,'Device pixel ratio is unbounded');
document.hidden=true;events.get('visibilitychange')();assert(queued.size===0,'Hidden page still animates');
document.hidden=false;const reduced=media.get('(prefers-reduced-motion: reduce)');reduced.matches=true;events.get('visibilitychange')();assert(queued.size===0,'Reduced motion still animates');
reduced.matches=false;reduced.listeners.forEach(fn=>fn());assert(queued.size===1,'Motion preference did not resume');
orb.setState('disconnected');assert(queued.size===0,'Disconnected renderer still animates');
orb.setState('idle');assert(queued.size===1,'Reconnect did not resume');
orb.destroy();assert(queued.size===0&&!events.has('visibilitychange'),'Renderer leaked after disposal');
"""
    result = subprocess.run(["node", "-e", harness + source + checks], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
