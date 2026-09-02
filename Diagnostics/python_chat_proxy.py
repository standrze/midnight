#!/usr/bin/env python3
"""Small same-origin browser UI for the MLX-VLM OpenAI-compatible server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


BACKEND = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "/models/gemma"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ABSlayer Chat</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#080b0f;color:#e7edf5}
main{max-width:900px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:center}
.status{color:#65d68b}.msg{white-space:pre-wrap;background:#171d25;border:1px solid #293342;border-radius:12px;padding:16px;margin:14px 0;line-height:1.45}
.user{background:#10223a}textarea{box-sizing:border-box;width:100%;min-height:110px;background:#171d25;color:#fff;border:2px solid #2176d2;border-radius:12px;padding:14px;font:inherit}
.actions{display:flex;justify-content:space-between;margin-top:10px}button{border:0;border-radius:8px;padding:10px 18px;background:#2878db;color:#fff;font-weight:600}button.secondary{background:#323b48}button:disabled{opacity:.5}
</style></head><body><main><div class="top"><h2>ABSlayer · Gemma 4 E2B</h2><span class="status">online</span></div>
<section id="chat"></section><textarea id="input" placeholder="Message the model… (Enter to send, Shift+Enter for a new line)"></textarea>
<div class="actions"><button class="secondary" id="clear">Clear</button><button id="send">Send</button></div></main>
<script>
const chat=document.querySelector('#chat'), input=document.querySelector('#input'), send=document.querySelector('#send');
let messages=[];
function bubble(role,text){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;chat.appendChild(d);return d}
async function submit(){const text=input.value.trim();if(!text||send.disabled)return;input.value='';send.disabled=true;
 messages.push({role:'user',content:text});bubble('user',text);const out=bubble('assistant','');
 try{const r=await fetch('/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages})});
  if(!r.ok)throw new Error(await r.text());const reader=r.body.getReader(),dec=new TextDecoder();let buf='',answer='';
  while(true){const {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});let p;
   while((p=buf.indexOf('\n'))>=0){const line=buf.slice(0,p).trim();buf=buf.slice(p+1);if(!line.startsWith('data:'))continue;
    const raw=line.slice(5).trim();if(raw==='[DONE]')continue;try{const j=JSON.parse(raw);const s=j.choices?.[0]?.delta?.content||'';answer+=s;out.textContent=answer;window.scrollTo(0,document.body.scrollHeight)}catch{}}
  } messages.push({role:'assistant',content:answer});
 }catch(e){out.textContent='Error: '+e.message}finally{send.disabled=false;input.focus()}}
send.onclick=submit;document.querySelector('#clear').onclick=()=>{messages=[];chat.replaceChildren()};
input.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submit()}};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            incoming = json.loads(self.rfile.read(length))
            payload = json.dumps({
                "model": MODEL,
                "messages": incoming["messages"],
                "max_tokens": 2048,
                "temperature": 0.7,
                "stream": True,
            }).encode()
            request = urllib.request.Request(BACKEND, payload, {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read(4096):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (KeyError, ValueError, urllib.error.URLError) as exc:
            body = str(exc).encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8081), Handler).serve_forever()
