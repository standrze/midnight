#!/usr/bin/env python3
"""Validate opt-in Metal timing samples and summarize queue-local observations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def stats(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'sum_seconds': sum(values), 'mean_seconds': statistics.mean(values),
            'median_seconds': statistics.median(values), 'p95_seconds': values[max(0, math.ceil(.95 * len(values)) - 1)],
            'maximum_seconds': max(values)}


def validate(report):
    require(report.get('format') == 1 and report.get('kind') == 'midnight_metal_command_timing', 'Wrong telemetry schema')
    require(report.get('clock') == 'system_mach_seconds' and report.get('snapshot') == 'atexit_without_gpu_wait', 'Unknown clocks or snapshot semantics')
    for key in ('capacity', 'skip_command_buffers', 'submitted_command_buffers', 'incomplete_sample_count',
                'unsampled_due_to_capacity', 'waits_seen', 'queues_created', 'queue_metadata_overflow'):
        require(integer(report.get(key)), f'Invalid {key}')
    require(1 <= report['capacity'] <= 65536, 'Invalid bounded capacity')
    commands, queues, waits = (report.get(k) for k in ('commands', 'queues', 'waits'))
    require(all(isinstance(rows, list) for rows in (commands, queues, waits)), 'Missing observation lists')
    eligible = max(0, report['submitted_command_buffers'] - report['skip_command_buffers'])
    slots = min(eligible, report['capacity'])
    require(len(commands) + report['incomplete_sample_count'] == slots, 'Completed/pending samples do not reconcile')
    require(report['unsampled_due_to_capacity'] == eligible - slots, 'Capacity overflow does not reconcile')
    require(len(waits) <= min(report['waits_seen'], report['capacity']), 'Too many retained waits')
    require(len(queues) <= min(report['queues_created'], 128), 'Too many retained queue identities')
    require(report['queue_metadata_overflow'] == max(0, report['queues_created'] - 128), 'Queue overflow does not reconcile')
    seen, generations = set(), set()
    for row in commands:
        for key in ('sequence', 'queue_key', 'operations', 'referenced_bytes', 'status'):
            require(integer(row.get(key)), f'Invalid command {key}')
        require(row['sequence'] not in seen, 'Duplicate command sequence')
        seen.add(row['sequence'])
        require(report['skip_command_buffers'] < row['sequence'] <= report['skip_command_buffers'] + slots, 'Sequence outside retained sample range')
        for key in ('commit_start', 'commit_return', 'gpu_start', 'gpu_end', 'callback_observed'):
            require(number(row.get(key)), f'Nonfinite/negative timestamp {key}')
        require(row['commit_start'] > 0 and row['commit_return'] >= row['commit_start'], 'Nonmonotonic host commit clock')
        require(row['callback_observed'] >= row['commit_start'], 'Callback precedes submission')
        require(row['status'] in (4, 5), 'Completion record has nonfinal Metal status')
        require(type(row.get('gpu_timing_valid')) is bool, 'Missing GPU timing validity')
        require(not row['gpu_timing_valid'] or 0 < row['gpu_start'] <= row['gpu_end'], 'Invalid reported GPU interval')
    for row in queues:
        for key in ('generation', 'queue_key'):
            require(integer(row.get(key), 1), f'Invalid queue {key}')
        require(type(row.get('stream_index')) is int and number(row.get('created')), 'Invalid queue metadata')
        require(row['generation'] not in generations, 'Duplicate queue generation')
        generations.add(row['generation'])
    for row in waits:
        require(integer(row.get('queue_key'), 1) and number(row.get('start')) and number(row.get('end')), 'Invalid wait record')
        require(row['end'] >= row['start'] > 0, 'Nonmonotonic existing wait interval')
    return commands, queues, waits


def analyze(report, sequence_from=None, sequence_to=None):
    commands, queues, waits = validate(report)
    if sequence_from is not None:
        commands = [r for r in commands if r['sequence'] >= sequence_from]
    if sequence_to is not None:
        commands = [r for r in commands if r['sequence'] <= sequence_to]
    require(commands, 'No completed samples in selected range')
    identities = defaultdict(list)
    for row in queues:
        identities[row['queue_key']].append(row)
    grouped, unmatched = defaultdict(list), 0
    for row in commands:
        preceding = [q for q in identities[row['queue_key']] if q['created'] <= row['commit_start']]
        if not preceding:
            unmatched += 1
            continue
        queue = max(preceding, key=lambda q: q['created'])
        grouped[(queue['generation'], queue['stream_index'])].append(row)
    # Missing records can hide an intervening buffer; never call its neighbors adjacent.
    gaps_available = (report['incomplete_sample_count'] == 0 and report['queue_metadata_overflow'] == 0
                      and len(queues) == report['queues_created'] and unmatched == 0)
    gpu = [r for r in commands if r['gpu_timing_valid']]
    negative_order_counts = Counter()

    def ordered_delta(rows, left, right, name):
        values = []
        for row in rows:
            difference = row[right] - row[left]
            if difference >= 0:
                values.append(difference)
            else:
                negative_order_counts[name] += 1
        return stats(values)

    metrics = {
        'cpu_commit_call': ordered_delta(commands, 'commit_start', 'commit_return', 'commit_call'),
        'commit_to_gpu_start': ordered_delta(gpu, 'commit_start', 'gpu_start', 'commit_to_gpu_start'),
        'gpu_execution_interval': ordered_delta(gpu, 'gpu_start', 'gpu_end', 'gpu_execution_interval'),
        'gpu_end_to_observer_callback': ordered_delta(gpu, 'gpu_end', 'callback_observed', 'gpu_end_to_callback'),
        'commit_to_observer_callback': ordered_delta(commands, 'commit_start', 'callback_observed', 'commit_to_callback'),
        'existing_synchronize_wait_all_retained': stats([r['end'] - r['start'] for r in waits]),
    }
    by_queue = []
    for (generation, stream), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['commit_start'])
        entry = {'queue_generation': generation, 'stream_index': stream, 'sample_count': len(rows)}
        if gaps_available:
            cpu_gaps, gpu_gaps, late_submissions, overlaps = [], [], [], []
            for previous, current in zip(rows, rows[1:]):
                difference = current['commit_start'] - previous['commit_return']
                if difference >= 0:
                    cpu_gaps.append(difference)
                else:
                    negative_order_counts['queue_cpu_between_commits'] += 1
                if previous['gpu_timing_valid'] and current['gpu_timing_valid']:
                    difference = current['gpu_start'] - previous['gpu_end']
                    gpu_gaps.append(max(0, difference))
                    overlaps.append(max(0, -difference))
                    late_submissions.append(max(0, current['commit_start'] - previous['gpu_end']))
            entry.update(cpu_between_commit_calls=stats(cpu_gaps), queue_gpu_gap=stats(gpu_gaps),
                         queue_gpu_overlap=stats(overlaps), submit_after_previous_gpu_end=stats(late_submissions))
        by_queue.append(entry)
    return {'format': 1, 'status': 'analyzed_observations', 'selected_commands': len(commands),
            'selected_sequence_range': [min(r['sequence'] for r in commands), max(r['sequence'] for r in commands)],
            'status_counts': dict(Counter(str(r['status']) for r in commands)), 'valid_gpu_samples': len(gpu),
            'invalid_gpu_samples': len(commands) - len(gpu), 'commands_without_queue_identity': unmatched,
            'adjacent_queue_gap_metrics_available': gaps_available, 'negative_order_counts': dict(negative_order_counts),
            'metrics': metrics, 'queues': by_queue,
            'coverage': {key: report[key] for key in ('capacity', 'skip_command_buffers', 'submitted_command_buffers',
                'incomplete_sample_count', 'unsampled_due_to_capacity', 'waits_seen', 'queues_created', 'queue_metadata_overflow')},
            'limitations': [
                'This is opt-in instrumented observation, not profiler evidence from an uninstrumented runtime. Measure instrumentation overhead separately.',
                'A command-buffer GPU interval can include event dependencies and stalls. It is not shader busy time or hardware utilization.',
                'Queue-local gaps do not imply global GPU idle time; other queues and processes can run concurrently.',
                'CPU time between commits includes encoding, model/Python/Swift work, allocator work and existing waits. It does not identify a specific bottleneck.',
                'No ordering between completion handlers is assumed. End-to-observer delay measures callback arrival, not a pure driver wake-up cost.',
                'Only existing CommandEncoder::synchronize waitUntilCompleted calls are timed; other waits are not covered. This wait includes all completion handlers, so observation itself can increase its duration.',
                'Wait statistics cover all retained wait records, even when command sequence filtering is requested.',
                'Bounded sampling keeps a contiguous initial range after the optional skip count; it is not a random or automatically phase-labeled sample.',
                'The exit snapshot adds no GPU wait, may precede dependency destructors, and explicitly reports missing or overflowed observations.',
                'Referenced bytes are MLX bookkeeping per buffer, not measured memory traffic. GPU sums can overlap and must not be interpreted as utilization.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sequence-from', type=int)
    parser.add_argument('--sequence-to', type=int)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = analyze(json.loads(raw), args.sequence_from, args.sequence_to)
    result['provenance'] = {'input': str(args.report.resolve()), 'input_sha256': hashlib.sha256(raw).hexdigest(),
                            'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Analyzed {result['selected_commands']} completed command-buffer observations")


if __name__ == '__main__':
    main()
