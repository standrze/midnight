#!/usr/bin/env python3
"""Compare cache-enabled and cache-disabled local servers, including overlap.
Usage: python3 Scripts/SmokeSharedPromptCache.py [warm-url] [cold-url]
Start the cold server with MODEL_RUNNER_PREFIX_CACHE_MIB=0.
"""
import concurrent.futures
import json
import sys
import urllib.request

from midnight_api_auth import authorization_headers, json_headers

warm = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:18089'
cold = sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:18090'
models = {}
for base in (warm, cold):
    with urllib.request.urlopen(urllib.request.Request(base + '/v1/models', headers=authorization_headers())) as response:
        models[base] = json.load(response)['data'][0]['id']
system = 'Answer the final arithmetic question briefly. ' + ('The garden has green trees, blue flowers and stone paths. ' * 40)
def request(base, question, prompt=system, responses=False):
    messages = [{'role':'system','content':prompt},{'role':'user','content':question}]
    payload = {'model':models[base],'temperature':0}
    payload.update({'input':messages,'max_output_tokens':32} if responses else {'messages':messages,'max_tokens':32})
    route = '/v1/responses' if responses else '/v1/chat/completions'
    with urllib.request.urlopen(urllib.request.Request(base+route, data=json.dumps(payload).encode(), headers=json_headers()), timeout=180) as response:
        return json.load(response)
def content(value):
    return value['choices'][0]['message']
def cached(value):
    return value['usage']['prompt_tokens_details']['cached_tokens']
questions = ['What is 3 plus 3?', 'What is 4 plus 4?', 'What is 5 plus 5?', 'What is 6 plus 6?']
request(warm, questions[0])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    actual = list(pool.map(lambda q: request(warm,q), questions))
expected = [request(cold,q) for q in questions]
for q, a, e in zip(questions, actual, expected):
    assert content(a) == content(e), (q,a,e)
    assert cached(a) >= 128 and cached(e) == 0, (a,e)
    assert a['usage']['prompt_tokens'] == e['usage']['prompt_tokens']
    print(q, content(a), 'cached:',cached(a))
# Different first tokens cannot inherit another client's cache.
changed = 'A completely different instruction: ' + system
miss = request(warm,questions[0],changed)
assert cached(miss) == 0
assert content(miss) == content(request(cold,questions[0],changed))
request(warm,questions[0])
response = request(warm,questions[1],responses=True)
assert response['usage']['input_tokens_details']['cached_tokens'] >= 128, response
print('PASS: four simultaneous clients, independent cold parity, prefix isolation, Chat and Responses usage')
