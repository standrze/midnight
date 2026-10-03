#!/usr/bin/env python3
"""Summarize an xctrace time-profile XML export; percentages describe sampled CPU weight."""
import argparse
import collections
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def analyze(path, start=0.0, end=None):
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end <= start)):
        raise ValueError('Require 0 <= start < end')
    root = ET.parse(path).getroot()
    ids = {element.get('id'): element for element in root.iter() if element.get('id')}
    def resolve(element):
        if element is None:
            return None
        seen = set()
        while element.get('ref'):
            key = element.get('ref')
            if key in seen or key not in ids:
                raise ValueError('Invalid or cyclic xctrace reference: ' + key)
            seen.add(key)
            element = ids[key]
        return element
    leaf = collections.Counter()
    inclusive = collections.Counter()
    samples = missing = 0
    weight_total = missing_weight = 0
    first = last = None
    for row in root.iter('row'):
        time = resolve(row.find('sample-time'))
        weight = resolve(row.find('weight'))
        if time is None or weight is None:
            raise ValueError('Expected time-profile rows with sample-time and weight')
        seconds = int(time.text) / 1e9
        if seconds < start or (end is not None and seconds >= end):
            continue
        ns = int(weight.text)
        if ns <= 0:
            raise ValueError('Sample weights must be positive')
        first = seconds if first is None else min(first, seconds)
        last = seconds if last is None else max(last, seconds)
        tagged = resolve(row.find('tagged-backtrace'))
        stack = resolve(tagged.find('backtrace')) if tagged is not None else None
        names = [resolve(frame).get('name', '<unknown>') for frame in stack] if stack is not None else []
        if not names:
            missing += 1
            missing_weight += ns
            continue
        samples += 1
        weight_total += ns
        leaf[names[0]] += ns
        for name in set(names):
            inclusive[name] += ns
    if not samples:
        raise ValueError('No symbolized stack samples in the requested interval')
    def table(counter):
        return [{'symbol': key, 'sampled_cpu_ms': value / 1e6,
                 'percent_of_symbolized_weight': 100 * value / weight_total}
                for key, value in counter.most_common()]
    return {'schema': 'midnight_time_profile_v1', 'interval_seconds': {'start': start, 'end': end},
            'first_sample_seconds': first, 'last_sample_seconds': last,
            'symbolized_samples': samples, 'missing_stack_samples': missing,
            'symbolized_cpu_ms': weight_total / 1e6, 'missing_stack_cpu_ms': missing_weight / 1e6,
            'leaf': table(leaf), 'inclusive': table(inclusive),
            'limitations': ['Sampled CPU weights are not wall-time percentages or GPU utilization.',
                            'Inclusive entries overlap; recursive frames count once per sample.',
                            'Profiling perturbs execution. Use separate unprofiled comparisons for speed claims.',
                            'No automatic classification of loading, prefill or decode; select a known interval.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float)
    args = parser.parse_args()
    result = analyze(args.input, args.start, args.end)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
