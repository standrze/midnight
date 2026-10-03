#!/usr/bin/env python3
"""Synthetic local SSE server for harness semantics; never runs an inference model."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def json_response(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/v1/models':
            self.json_response({'data':[{'id':self.server.model}]})
        elif self.path == '/v1/runtime':
            self.json_response({'phase':'ready', 'fixture':True, 'contextLength':8192, 'prefillStepSize':512})
        else:
            self.json_response({'error':'unknown route'}, 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not self.server.busy.acquire(blocking=False):
            self.json_response({'error':'draining'}, 429)
            return
        try:
            self.respond(payload)
        except (BrokenPipeError, ConnectionResetError):
            time.sleep(.08)
        finally:
            self.server.busy.release()

    def respond(self, payload):
        responses = self.path == '/v1/responses'
        messages = payload.get('messages', payload.get('input', []))
        if payload.get('previous_response_id'):
            messages = self.server.histories[payload['previous_response_id']] + messages
        text = json.dumps(messages)
        marker = 'NOVA284' if 'NOVA284' in text else 'READY' if 'READY' in text else 'ORBIT731'
        cancel = 'Cancellation request' in text
        answer = ('A bicycle needs regular cleaning and careful inspection. '*40 if 'Decode request' in text or cancel else marker)
        usage = {'prompt_tokens':len(text.split()), 'completion_tokens':len(answer.split())+1,
                 'prompt_tokens_details':{'cached_tokens':0}}
        if 'INVALID_TOKENS' in text:
            usage['completion_tokens'] = '2'
        systems = [m['content'] for m in messages if m.get('role')=='system']
        prefix = hashlib.sha256(json.dumps(systems or messages).encode()).hexdigest()
        if prefix in self.server.prefixes:
            usage['prompt_tokens_details']['cached_tokens'] = max(1,usage['prompt_tokens']-8)
        self.server.prefixes.add(prefix)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        factor = float(os.getenv('FIXTURE_FACTOR', '1'))
        def emit(value):
            self.wfile.write(('data: '+(value if isinstance(value,str) else json.dumps(value))+'\r\n\r\n').encode())
            self.wfile.flush()
        response_id = 'resp_'+str(len(self.server.histories))
        if responses:
            emit({'type':'response.created','response':{'id':response_id}})
        else:
            emit({'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})
        time.sleep(.015*factor)
        if 'REASONING' in text and not responses:
            emit({'choices':[{'index':0,'delta':{'reasoning_content':'Think.'},'finish_reason':None}]})
        chunks = [answer[i:i+4] for i in range(0,len(answer),4)]
        for chunk in chunks:
            if responses:
                emit({'type':'response.output_text.delta','delta':chunk})
            else:
                emit({'choices':[{'index':0,'delta':{'content':chunk},'finish_reason':None}]})
            time.sleep((.002 if not cancel else .02)*factor)
        time.sleep(.005*factor)
        if responses:
            self.server.histories[response_id] = messages + [{'role':'assistant','content':answer}]
            converted = {'input_tokens':usage['prompt_tokens'], 'output_tokens':usage['completion_tokens'], 'input_tokens_details':usage['prompt_tokens_details']}
            emit({'type':'response.completed','response':{'id':response_id,'status':'completed','usage':converted,
                    'output':[{'id':'generated-id','type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':answer}]}]}})
        else:
            emit({'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})
            if 'MISSING_USAGE' not in text:
                emit({'choices':[], 'usage':usage})
            emit('[DONE]')


def make_server(port=0, model='bench'):
    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.model, server.histories, server.prefixes = model, {}, set()
    server.busy = threading.Lock()
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--name',default='bench')
    parser.add_argument('--config')
    parser.add_argument('--model')
    args = parser.parse_args()
    make_server(args.port,args.name).serve_forever()
