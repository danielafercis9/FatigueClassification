"""Local-only boundary and artifact review. Serves pseudonymous traces, never raw paths."""
from __future__ import annotations
import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse
from pipeline import HERE, atomic_json, initialize, utc_now, validate_annotation

HTML = r'''<!doctype html><meta charset="utf-8"><title>Review active-set boundaries</title>
<style>body{font:15px system-ui;margin:24px;color:#193247;background:#f4f7f9}main{max-width:1400px;margin:auto}canvas{background:white;width:100%;border:1px solid #b5c5ce;touch-action:none}input,select,button,textarea{font:inherit;padding:8px;margin:4px}button{cursor:pointer}textarea{width:95%;height:50px}.note{background:#fff3d1;padding:12px}.row{display:flex;flex-wrap:wrap;align-items:center}#message{font-weight:bold}label{display:inline-block}</style>
<main><h1>Active-set and artifact review</h1><p><a href="/results/" target="_blank">View the latest feature trajectories and processing outputs</a></p>
<p class="note">Restricted local data. Proposals are navigation aids, not verified events. Review the raw waveform and RMS envelope. Mark the first sustained exercise activity and the end of the final exercise activity; do not trim an unusual final contraction merely to improve a trajectory. If uncertain, keep the set pending and document why. Boundaries reviewed from EMG do not independently verify technical failure.</p>
<div class="row"><button onclick="move(-1)">Previous</button><select id="record" onchange="load()"></select><button onclick="move(1)">Next</button><span id="progress"></span></div>
<p id="description"></p><canvas id="trace" width="1320" height="400"></canvas><canvas id="rms" width="1320" height="190"></canvas>
<p>Drag horizontally on either graph to select active bounds. Use exact inputs for refinement. Scroll over a graph to zoom; “Full view” restores the recording.</p>
<div class="row"><label>Start (s)<input id="start" type="number" step="0.001"></label><label>End exclusive (s)<input id="end" type="number" step="0.001"></label><button onclick="proposal()">Load proposal</button><button onclick="resetView()">Full view</button><button onclick="zoomBounds()">Zoom to bounds</button></div>
<div class="row"><label>Status<select id="status"><option value="pending">Pending / uncertain</option><option value="reviewed">Reviewed active interval</option><option value="exclude">Exclude set with reason</option></select></label><label>Reviewer ID<input id="reviewer" placeholder="Use initials or anonymous ID"></label></div>
<label>Review note / exclusion reason<textarea id="note" placeholder="Describe boundary uncertainty, artifact evidence or reason for exclusion. Never use model performance."></textarea></label>
<h3>Artifact intervals to omit from feature calculation</h3><p>Original samples remain unchanged. An interval excludes any overlapping window. Leave empty when no defensible artifact interval has been identified.</p>
<div class="row"><input id="astart" type="number" step="0.001" placeholder="Artifact start (s)"><input id="aend" type="number" step="0.001" placeholder="Artifact end (s)"><input id="areason" placeholder="Observed artifact and reason"><button onclick="addArtifact()">Add interval</button></div><div id="artifacts"></div>
<button onclick="save(false)">Save review</button><button onclick="save(true)">Save and next</button><span id="message"></span>
<p>After reviewing, rerun <code>py -3.13 processing/pipeline.py run</code>. Only reviewed sets enter Figure 2. Save before changing recordings; unsaved edits are not retained.</p></main>
<script>
let inventory=[],annotations={},data=null,spans=[],view=[0,1],drag=null,reviewer='';
const el=id=>document.getElementById(id);
async function init(){let r=await fetch('/api/inventory');inventory=await r.json();let a=await fetch('/api/annotations');annotations=await a.json();el('record').innerHTML=inventory.map(x=>`<option value="${x.recording_id}">${x.recording_id} | ${x.leg} | ${x.set_token}</option>`).join('');load();}
async function load(){let rid=el('record').value;data=await(await fetch('/api/trace/'+rid)).json();let a=annotations.recordings[rid];el('start').value=a.start_s??a.candidate_start_s;el('end').value=a.end_s??a.candidate_end_s;el('status').value=a.status;el('reviewer').value=a.reviewer||reviewer;el('note').value=a.note;spans=structuredClone(a.artifact_intervals||[]);view=[data.first_time_s,data.end_exclusive_s];el('description').textContent=`${rid}: leg ${data.leg}, set token ${data.set_token}; CSV header mV, calibration unknown. Orange = amplitude envelope; blue = raw signal.`;el('message').textContent='';renderArtifacts();draw();progress();}
function progress(){let n=Object.values(annotations.recordings).filter(x=>x.status!=='pending').length;el('progress').textContent=`${n}/${inventory.length} reviewed or documented exclusions`;}
function move(delta){let s=el('record');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+delta));load();}
function proposal(){let a=annotations.recordings[el('record').value];el('start').value=a.candidate_start_s;el('end').value=a.candidate_end_s;el('status').value='pending';draw();}
function resetView(){view=[data.first_time_s,data.end_exclusive_s];draw();}
function zoomBounds(){view=[Math.max(data.first_time_s,+el('start').value-.5),Math.min(data.end_exclusive_s,+el('end').value+.5)];draw();}
function drawPanel(canvas,ts,x,color,label){let c=canvas.getContext('2d'),w=canvas.width,h=canvas.height,L=75,R=w-18,T=25,B=h-35;c.clearRect(0,0,w,h);let ys=x.filter((v,i)=>ts[i]>=view[0]&&ts[i]<=view[1]);if(!ys.length)return;let lo=Math.min(...ys),hi=Math.max(...ys);if(hi===lo){lo-=1;hi+=1;}let px=t=>L+(t-view[0])/(view[1]-view[0])*(R-L),py=v=>B-(v-lo)/(hi-lo)*(B-T);c.fillStyle='#172e41';c.font='13px system-ui';c.fillText(label,L,17);c.strokeStyle='#dce4e8';for(let j=0;j<5;j++){let y=T+(B-T)*j/4;c.beginPath();c.moveTo(L,y);c.lineTo(R,y);c.stroke();c.fillText((hi-(hi-lo)*j/4).toPrecision(3),4,y+4);let t=view[0]+(view[1]-view[0])*j/4;c.fillText(t.toFixed(2),px(t)-15,h-10);}
let a=+el('start').value,b=+el('end').value;c.fillStyle='#2b9f7222';c.fillRect(Math.max(L,px(a)),T,Math.max(0,Math.min(R,px(b))-Math.max(L,px(a))),B-T);for(let [s,z] of spans){c.fillStyle='#db403844';c.fillRect(Math.max(L,px(s)),T,Math.max(0,Math.min(R,px(z))-Math.max(L,px(s))),B-T);}c.save();c.beginPath();c.rect(L,T,R-L,B-T);c.clip();c.strokeStyle=color;c.lineWidth=.8;c.beginPath();let first=true;for(let i=0;i<ts.length;i++){if(ts[i]<view[0]||ts[i]>view[1])continue;let xx=px(ts[i]),yy=py(x[i]);if(first){c.moveTo(xx,yy);first=false;}else c.lineTo(xx,yy);}c.stroke();c.restore();}
function draw(){if(!data)return;for(let id of ['trace','rms']){let cv=el(id);cv.width=Math.max(320,Math.round(cv.clientWidth));cv.height=id==='trace'?300:180;}drawPanel(el('trace'),data.t,data.x,'#126782','Raw EMG (header mV); min/max bins');drawPanel(el('rms'),data.envelope_t,data.envelope,'#b34c16','100 ms RMS (boundary aid only)');}
window.addEventListener('resize',draw);
function timeAt(event,canvas){let rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width;return Math.max(data.first_time_s,Math.min(data.end_exclusive_s,view[0]+(x-75)/(canvas.width-93)*(view[1]-view[0])));}
for(let name of ['trace','rms']){let cv=el(name);cv.addEventListener('pointerdown',e=>{drag=timeAt(e,cv);cv.setPointerCapture(e.pointerId);});cv.addEventListener('pointerup',e=>{if(drag===null)return;let t=timeAt(e,cv);el('start').value=Math.min(drag,t).toFixed(3);el('end').value=Math.max(drag,t).toFixed(3);drag=null;el('status').value='pending';draw();});cv.addEventListener('wheel',e=>{e.preventDefault();let center=timeAt(e,cv),scale=e.deltaY>0?1.3:.77;view=[Math.max(data.first_time_s,center-(center-view[0])*scale),Math.min(data.end_exclusive_s,center+(view[1]-center)*scale)];if(view[1]-view[0]<.01)resetView();else draw();},{passive:false});}
for(let id of ['start','end'])el(id).addEventListener('input',()=>{el('status').value='pending';draw();});
function renderArtifacts(){let host=el('artifacts');host.textContent='';spans.forEach((s,i)=>{let p=document.createElement('p');p.textContent=`${s[0]}–${s[1]} s: ${s[2]} `;let b=document.createElement('button');b.textContent='Remove';b.onclick=()=>{spans.splice(i,1);renderArtifacts();draw();};p.append(b);host.append(p);});}
function addArtifact(){let a=+el('astart').value,b=+el('aend').value,r=el('areason').value.trim();if(!(a<b)||!r){el('message').textContent='Enter ordered artifact times and a reason.';return;}spans.push([a,b,r]);renderArtifacts();draw();}
async function save(next){let rid=el('record').value,old=annotations.recordings[rid];let a={...old,start_s:+el('start').value,end_s:+el('end').value,status:el('status').value,reviewer:el('reviewer').value.trim(),note:el('note').value.trim(),artifact_intervals:spans};let response=await fetch('/api/review/'+rid,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(a)});let result=await response.json();if(!response.ok){el('message').textContent=result.error;return;}annotations.recordings[rid]=result;reviewer=a.reviewer;el('message').textContent='Saved locally.';progress();if(next)move(1);}
init();
</script>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass  # No participant paths or review notes in terminal logs.
    def send_json(self,obj,status=200):
        encoded=json.dumps(obj,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(encoded))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(encoded)
    def do_GET(self):
        if self.headers.get('Host') not in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']:
            return self.send_json({'error':'Only loopback hosts allowed'},403)
        path=urlparse(self.path).path
        if path.startswith('/results/'):
            latest=HERE/'outputs/latest_run.json'
            if not latest.exists(): return self.send_json({'error':'Run the processing pipeline first.'},404)
            metadata=json.loads(latest.read_text()); folder=(HERE.parent/metadata['path'].replace('\\','/')).resolve()
            if not folder.is_relative_to((HERE/'outputs').resolve()): return self.send_json({'error':'Invalid output location'},400)
            files={p.name:p for p in folder.glob('*.png')}
            for name in ['feature_trajectories','preview_feature_trajectories']:
                for p in (folder/name).glob('*.png'): files[name+'/'+p.name]=p
            for name in ['window_features.csv','feature_dictionary.csv','recording_processing_log.csv','run_manifest.json']:
                if (folder/name).exists(): files[name]=folder/name
            key=path.removeprefix('/results/')
            if not key:
                options=''.join(f'<option value="{html.escape(k)}">{html.escape(k)}</option>' for k in files if k.endswith('.png'))
                first=next((k for k in files if k.endswith('.png')),'')
                pending=metadata['status']['pending_recordings']
                content=(f'<!doctype html><meta charset="utf-8"><title>Section 4 results</title><style>body{{font:16px system-ui;margin:24px}}select,img{{width:100%}}select{{padding:12px}}</style><h1>Section 4 outputs</h1><p>Run {metadata["run_id"]}. {pending} sets pending boundary review. PREVIEW files are not final report figures. All outputs are restricted local data.</p><select onchange="document.getElementById(\'plot\').src=this.value">'+options+f'</select><img id="plot" src="{html.escape(first)}" alt="Feature trajectories"><p><a href="window_features.csv">Window feature table</a> | <a href="feature_dictionary.csv">Feature definitions</a> | <a href="recording_processing_log.csv">Processing log</a> | <a href="/">Boundary review</a></p>').encode()
                self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content); return
            if key in files:
                content=files[key].read_bytes(); self.send_response(200)
                self.send_header('Content-Type','image/png' if key.endswith('.png') else 'text/csv' if key.endswith('.csv') else 'application/json')
                self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content); return
            return self.send_json({'error':'Unknown result'},404)
        if path=='/':
            content=HTML.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content); return
        if path=='/api/inventory': return self.send_json(json.loads((HERE/'review_data/inventory.json').read_text()))
        if path=='/api/annotations': return self.send_json(json.loads((HERE/'annotations.json').read_text()))
        if path.startswith('/api/trace/'):
            rid=path.removeprefix('/api/trace/')
            ids={x['recording_id'] for x in json.loads((HERE/'review_data/inventory.json').read_text())}
            if rid in ids: return self.send_json(json.loads((HERE/'review_data'/f'{rid}.json').read_text()))
        self.send_json({'error':'Not found'},404)
    def do_POST(self):
        if self.headers.get('Host') not in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']:
            return self.send_json({'error':'Only loopback hosts allowed'},403)
        # Reject cross-origin requests; service binds only to loopback.
        allowed=f'http://{self.headers.get("Host", "")}'
        if self.headers.get('Origin') not in [None,allowed]: return self.send_json({'error':'Cross-origin write rejected'},403)
        path=urlparse(self.path).path
        rid=path.removeprefix('/api/review/') if path.startswith('/api/review/') else ''
        annotations=json.loads((HERE/'annotations.json').read_text())
        if rid not in annotations['recordings']: return self.send_json({'error':'Unknown recording'},404)
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<100000: raise ValueError('Invalid payload size.')
            supplied=json.loads(self.rfile.read(size))
            # Identity/checksum/proposal fields cannot be edited by the UI payload.
            a=dict(annotations['recordings'][rid])
            for key in ['status','start_s','end_s','reviewer','note','artifact_intervals']: a[key]=supplied[key]
            info=json.loads((HERE/'review_data'/f'{rid}.json').read_text())
            validate_annotation(a,info); a['updated_utc']=utc_now()
            if a['status']=='reviewed': a['evidence']='Manual EMG boundary review; technical failure per user testimony, not independently verified.'
            # Append full review history before replacing current annotations.
            history=HERE/'review_data/review_history.jsonl'
            with history.open('a',encoding='utf-8') as stream: stream.write(json.dumps(dict(recording_id=rid,annotation=a))+'\n')
            annotations['recordings'][rid]=a; atomic_json(HERE/'annotations.json',annotations)
            self.send_json(a)
        except (ValueError,KeyError,TypeError) as exc: self.send_json({'error':str(exc)},400)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--port',type=int,default=8765); args=parser.parse_args()
    if not (HERE/'review_data/inventory.json').exists(): initialize()
    print(f'Boundary review: http://127.0.0.1:{args.port}/ (local only). Ctrl+C stops.',flush=True)
    try: HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
    except KeyboardInterrupt: pass


if __name__=='__main__': main()
