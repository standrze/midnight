#!/usr/bin/env python3
"""Pair saved generated-evaluation scores without re-executing any candidate code."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys


def info(path):
    path=Path(path); raw=path.read_bytes()
    return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer',type=Path,required=True)
    parser.add_argument('--scores',action='append',required=True,metavar='LABEL=PATH')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20260904)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Output exists; preserve earlier evidence')
    if not 100<=args.draws<=100000: raise ValueError('Draws must be100...100000')
    scorer_info=info(args.scorer)
    spec=importlib.util.spec_from_file_location('frozen_generated_scorer',args.scorer)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    models,inputs,common={},[],None
    for item in args.scores:
        label,sep,path=item.partition('=')
        if not sep or not scorer.LABEL.fullmatch(label) or label in models: raise ValueError('Unique LABEL=PATH required')
        source=json.loads(Path(path).read_text())
        if source.get('format')!=1 or label not in source.get('models',{}): raise ValueError('Unexpected saved score schema/label')
        p=source['provenance']
        if p['scorer']['sha256']!=scorer_info['sha256']: raise ValueError('Saved scores were produced by a different scorer revision')
        identity={k:p[k] if k=='sandbox' else p[k]['sha256'] for k in ('tasks','private_answers','sandbox')}
        if common is None:common=identity
        elif identity!=common:raise ValueError('Tasks, private answers or sandbox policy differ across saved scores')
        model=source['models'][label]
        rows=model['samples']
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate scored IDs')
        if scorer.summarize(rows)!=model['overall']:raise ValueError('Saved totals disagree with per-sample scores')
        if models and {(r['id'],r['category']) for r in rows}!={(r['id'],r['category']) for r in next(iter(models.values()))['samples']}:
            raise ValueError('Saved sample IDs/categories differ')
        models[label]=model;inputs.append({'label':label,**info(path)})
    if len(models)<2:raise ValueError('At least two saved score reports required')
    comparisons={f'{b}_minus_{a}':scorer.paired_comparison(models[a],models[b],args.draws,args.seed)
                 for a,b in itertools.combinations(models,2)}
    result={'format':1,'purpose':'Paired task-accuracy comparison from retained scored reports; no Docker or generation rerun.',
            'provenance':{'inputs':inputs,'scorer':scorer_info,'reproducer':info(__file__),'shared_identity':common},
            'bootstrap':{'draws':args.draws,'seed':args.seed,'unit':'paired task; overall stratified by category, record-weighted'},
            'models':{label:{'overall':m['overall'],'categories':m['categories'],'settings':m['settings']} for label,m in models.items()},
            'comparisons':comparisons,
            'limitations':['Strict primary scores are reused unchanged. Retrieval content/tag diagnostics do not alter pass criteria.',
                           'Intervals describe this finite public regression suite; contamination, source correlation, run/numerical variation and multiple comparisons are not corrected.',
                           'No timing claim or universal quality ranking follows.']}
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
