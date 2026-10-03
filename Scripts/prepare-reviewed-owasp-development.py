#!/usr/bin/env python3
"""Freeze reviewed OWASP development tasks before generation; never run Java."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile

SPEC = importlib.util.spec_from_file_location(
    'owasp_proposals', Path(__file__).with_name('prepare-owasp-review-proposals.py'))
proposals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proposals)

HELPERS = '''Verified benchmark context (only numbered servlet lines may be cited):
HTTP headers, parameter names/values and cookies are attacker controlled.
SeparateClassRequest.getTheParameter returns request.getParameter; getTheCookie
returns the named cookie value; getTheValue always returns the constant "bar".
Configured ThingFactory selects Thing2. Thing1 and Thing2 preserve input;
Thing2 returns an empty string for null. They do not sanitize input.
Utils.TESTFILES_DIR is a trusted base directory, not a containment check.
Utils.getOSCommandString(x) returns "cmd.exe /c " + x + " " on Windows,
and x + " " on Unix. getOSCommandArray(x) returns ["cmd.exe", "/c", x]
on Windows or ["sh", "-c", x] on Unix. An empty command string remains empty
on Unix. getInsecureOSCommandString selects the benchmark's trusted script:
insecureCmd.sh executes `eval $FOO`; insecureCmd.bat executes `%FOO%`.
Database, classpath, configuration, and ambient environment are trusted except
explicit servlet-controlled values passed into them. Evaluate whether the
scoped vulnerability exists on ANY supported Windows/Unix configuration.
Comments and suggestive names are not proof; follow active code and data flow.
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, review, output):
    if output.exists():
        raise ValueError('Frozen output exists; choose a new directory')
    with tempfile.TemporaryDirectory() as temporary:
        verified = Path(temporary) / 'proposals'
        proposals.prepare(source, verified)
        mappings = [json.loads(line) for line in
                    (verified / 'private-source-mapping.jsonl').read_text().splitlines()]
    review_files = ['path-and-command-reviewed.json', 'sql-reviewed.json',
                    'randomness-excluded.json']
    review_data = {name: (review / name).read_bytes() for name in review_files}
    cases = {}
    for name in review_files[:2]:
        document = json.loads(review_data[name])
        if document['review_status'] != 'reviewed' or set(cases) & set(document['cases']):
            raise ValueError('Review missing or duplicated')
        cases.update(document['cases'])
    excluded = json.loads(review_data[review_files[2]])
    expected_excluded = {m['original_case'] for m in mappings if m['publisher_label_row'][3] == '330'}
    if (excluded['status'] != 'excluded_before_model_generation'
            or excluded['generation_started'] is not False
            or set(excluded['excluded_cases']) != expected_excluded):
        raise ValueError('Unverified exclusion')
    selected = [m for m in mappings if m['original_case'] not in expected_excluded]
    if set(cases) != {m['original_case'] for m in selected}:
        raise ValueError('Every included case needs exactly one source review')
    helper_hashes = {}
    revision = json.loads((source / 'source-manifest.json').read_text())['revision']
    for name in ['manifest.json', 'resources-manifest.json']:
        raw = (source / 'helpers' / name).read_bytes()
        manifest = json.loads(raw)
        if manifest['revision'] != revision:
            raise ValueError('Helper revision differs')
        helper_hashes[name] = sha(raw)
        for record in manifest['files']:
            data = (source / 'helpers' / record['file']).read_bytes()
            if len(data) != record['bytes'] or sha(data) != record['sha256']:
                raise ValueError('Helper changed: ' + record['file'])
            helper_hashes[record['file']] = sha(data)
    seed = 20261001
    random.Random(seed).shuffle(selected)
    tasks, answers, private = [], [], []
    for index, mapping in enumerate(selected, 1):
        original = mapping['original_case']
        annotation = cases[original]
        row = mapping['publisher_label_row']
        vulnerable = row[2] == 'true'
        lines = (source / (original + '.java')).read_text().splitlines()
        evidence, anchors = annotation['evidence'], annotation['anchors']
        for values in [evidence, anchors]:
            if (not isinstance(values, list) or any(type(n) is not int or not 1 <= n <= len(lines)
                    or not lines[n - 1].strip() or lines[n - 1].strip().startswith('//')
                    for n in values) or len(values) != len(set(values))):
                raise ValueError('Invalid reviewed evidence: ' + original)
        if not set(anchors) <= set(evidence) or bool(anchors) != vulnerable or bool(evidence) != vulnerable:
            raise ValueError('Review disagrees with publisher verdict: ' + original)
        neutral = f'Case{index:04d}'
        text = '\n'.join(lines).replace(original, neutral)
        identifier = f'owasp-reviewed-development-{index:04d}'
        prompt = (f'Review only {proposals.SCOPES[row[3]]} (CWE-{row[3]}) in the numbered Java servlet. '
                  'Safe means no vulnerability in that scope, not a complete audit. '
                  'Return only JSON with exactly verdict ("safe" or "vulnerable"), '
                  f'cwes (array of canonical strings such as "CWE-{row[3]}", never numeric IDs), '
                  'and evidence_lines (array of distinct source-line integers). Safe requires empty arrays. '
                  'Vulnerable requires the scoped CWE and at least one line showing SQL construction/execution, '
                  'file access/construction, or command execution involved in the vulnerability. '
                  'You may also cite active source-to-sink data-flow lines, but do not cite comments, '
                  'unrelated code, or helper-context line numbers.\n\n' + HELPERS + '\n'
                  + '\n'.join(f'{n}: {line}' for n, line in enumerate(text.splitlines(), 1)))
        tasks.append({'id': identifier, 'category': 'cybersecurity', 'prompt': prompt,
                      'metadata': {'split': 'development', 'family': 'owasp-cwe-' + row[3],
                                   'source_revision': revision}})
        answers.append({'id': identifier, 'kind': 'security-json',
                        'verdict': 'vulnerable' if vulnerable else 'safe',
                        'cwes': ['CWE-' + row[3]] if vulnerable else [],
                        'evidence_lines': evidence, 'evidence_anchor_lines': anchors,
                        'line_count': len(lines), 'evidence_review_status': 'reviewed'})
        private.append({**mapping, 'id': identifier, 'source_review': annotation,
                        'transformed_source_sha256': sha(text.encode())})
    payloads = {name: ''.join(json.dumps(row) + '\n' for row in rows).encode()
                for name, rows in [('tasks.jsonl', tasks), ('answers.jsonl', answers),
                                   ('private-source-mapping.jsonl', private)]}
    provenance = {'status': 'frozen_reviewed_development_before_generation',
                  'task_count': len(tasks), 'seed': seed, 'revision': revision,
                  'builder_sha256': sha(Path(__file__).read_bytes()),
                  'proposal_verifier_sha256': sha(Path(proposals.__file__).read_bytes()),
                  'source_manifest_sha256': sha((source / 'source-manifest.json').read_bytes()),
                  'review_sha256': {name: sha(data) for name, data in review_data.items()},
                  'helper_sha256': helper_hashes,
                  'files': {name: sha(data) for name, data in payloads.items()},
                  'limitations': ['Public synthetic development set; unknown training exposure and correlated templates.',
                                 'Publisher supplies verdict/CWE; primary evaluator reviewed evidence and helper scope. No independent evidence reviewer.',
                                 'Scoped CWE is disclosed; source comments and names may give label-correlated hints.',
                                 'All CWE-330 cases excluded for insufficient-unpredictability label/scope mismatch before generation.',
                                 'Shuffled IDs avoid ordering by label; source logic and line numbers preserved.',
                                 'No final holdout or release promotion claim.']}
    provenance['schema_clarification'] = ('Canonical CWE string format is explicit. Introduced after v2 '
        '31B outputs exposed numeric-label ambiguity; v2 results remain unchanged. '
        'All recipes must run this new corpus for a new matched comparison.')
    output.mkdir(parents=True)
    for name, data in payloads.items():
        (output / name).write_bytes(data)
        if name != 'tasks.jsonl':
            (output / name).chmod(0o600)
    (output / 'LICENSE').write_bytes((source / 'LICENSE').read_bytes())
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.review_dir, args.output_dir), indent=2))
