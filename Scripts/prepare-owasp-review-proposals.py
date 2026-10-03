#!/usr/bin/env python3
"""Prepare a frozen OWASP review development set with unapproved evidence proposals.

Publisher verdict/CWE labels are verified against the pinned CSV. Evidence lines
are mechanically proposed and require review. Proposed keys intentionally use
an unsupported scoring kind so they cannot be mistaken for approved ground truth.
Only public tasks go to generation; no network or code execution occurs here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

PATTERNS = {
    '22': r'\b(?:File|FileInputStream|FileOutputStream|FileReader|FileWriter)\s*\(|\b(?:readAllBytes|newInputStream|newOutputStream)\s*\(',
    '78': r'\.exec\s*\(|\bProcessBuilder\s*\(|\.start\s*\(',
    '89': r'\b(?:prepareStatement|prepareCall|executeQuery|executeUpdate|execute|queryForObject|queryForList|batchUpdate|update)\s*\(|String\s+sql\b',
    '330': r'\b(?:Random|SecureRandom)\b|\brandom\s*\(|\bnext(?:Int|Long|Float|Double|Bytes)\s*\(',
}
SCOPES = {'22': 'path traversal', '78': 'OS command injection',
          '89': 'SQL injection', '330': 'insufficient unpredictability of random values'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, output):
    if output.exists():
        raise ValueError('Use a new output directory; frozen proposals are never overwritten')
    manifest = json.loads((source / 'source-manifest.json').read_text())
    selection = json.loads((source / 'selection.json').read_text())
    if selection['revision'] != manifest['revision'] or selection['repository'] != manifest['repository']:
        raise ValueError('Source selection and manifest revisions disagree')
    labels_raw = (source / 'expectedresults-1.2.csv').read_bytes()
    license_raw = (source / 'LICENSE').read_bytes()
    if digest(labels_raw) != manifest['expected_labels_sha256'] or digest(license_raw) != manifest['license_sha256']:
        raise ValueError('Pinned source labels or license changed')
    labels = {row[0]: row for row in csv.reader(line for line in labels_raw.decode().splitlines()
                                               if line and not line.startswith('#'))}
    files = {record['file']: record for record in manifest['files']}
    tasks, proposed, mapping = [], [], []
    seen = set()
    for index, selected in enumerate(selection['rows'], 1):
        original, _, vulnerable, cwe = selected
        if selected != labels.get(original) or original in seen or cwe not in PATTERNS:
            raise ValueError('Invalid/duplicate source label selection')
        seen.add(original)
        identity = files[original + '.java']
        raw = (source / identity['file']).read_bytes()
        if digest(raw) != identity['sha256'] or len(raw) != identity['bytes']:
            raise ValueError('Pinned Java source changed: ' + original)
        text = raw.decode()
        neutral = f'Case{index:04d}'
        # Rename only this case ID consistently, including servlet/header keys.
        # Preserve line numbering, license notice and all logic/constants.
        transformed = text.replace(original, neutral)
        evidence = [n for n, line in enumerate(transformed.splitlines(), 1)
                    if re.search(PATTERNS[cwe], line)]
        identifier = f'owasp-review-development-{index:04d}'
        prompt = (f'Review only {SCOPES[cwe]} (CWE-{cwe}) in the numbered Java servlet below. '
                  'HTTP request headers, parameters and cookies are attacker controlled. '
                  'Benchmark database/OS/path helper methods supply trusted infrastructure; '
                  'do not infer additional authorization or sanitization outside the shown logic. '
                  'Safe means no vulnerability in this declared scope, not a complete audit. '
                  'Return only JSON with verdict ("safe" or "vulnerable"), cwes (array), '
                  'and evidence_lines (array of source-line integers). Safe requires empty '
                  'arrays; vulnerable requires the scoped CWE and one or more relevant lines.\n\n'
                  + '\n'.join(f'{n}: {line}' for n, line in enumerate(transformed.splitlines(), 1)))
        tasks.append({'id': identifier, 'category': 'cybersecurity', 'prompt': prompt,
                      'metadata': {'split': 'development', 'family': 'owasp-cwe-' + cwe,
                                   'source_revision': manifest['revision']}})
        proposed.append({'id': identifier, 'kind': 'security-json-proposal',
                         'verdict': 'vulnerable' if vulnerable == 'true' else 'safe',
                         'cwes': ['CWE-' + cwe] if vulnerable == 'true' else [],
                         'evidence_lines': evidence if vulnerable == 'true' else [],
                         'line_count': len(transformed.splitlines()),
                         'evidence_review_status': 'pending',
                         'scope_and_helper_review_status': 'pending'})
        mapping.append({'id': identifier, 'original_case': original, 'publisher_label_row': selected,
                        'source_identity': identity, 'transformed_source_sha256': digest(transformed.encode()),
                        'candidate_sink_lines': evidence})
    if len(tasks) != manifest['source_cases'] or len(files) != len(tasks):
        raise ValueError('Selection must cover every frozen source exactly once')
    payloads = {name: ''.join(json.dumps(row) + '\n' for row in rows).encode()
                for name, rows in [('tasks.jsonl', tasks), ('proposed-answers.jsonl', proposed),
                                   ('private-source-mapping.jsonl', mapping)]}
    provenance = {'status': 'prepared_pending_evidence_and_scope_review', 'task_count': len(tasks),
                  'repository': manifest['repository'], 'revision': manifest['revision'],
                  'publisher_labels_sha256': digest(labels_raw), 'builder_sha256': digest(Path(__file__).read_bytes()),
                  'files': {name: digest(data) for name, data in payloads.items()},
                  'limitations': ['Public synthetic development benchmark with possible training exposure.',
                                  'Publisher supplies verdict/CWE; evidence proposals and prompt/helper assumptions require review.',
                                  'Scoped CWE is disclosed; this measures within-scope review, not unconstrained taxonomy discovery.',
                                  'Case IDs are renamed consistently; source license, line numbering and logic remain intact.',
                                  'Grouped by CWE conservatively; structural source-family independence is not established.',
                                  'Proposed answers cannot be scored by evaluate-generated.py until reviewed and explicitly approved.']}
    output.mkdir(parents=True)
    for name, data in payloads.items():
        with (output / name).open('xb') as stream:
            stream.write(data)
        if name != 'tasks.jsonl':
            (output / name).chmod(0o600)
    (output / 'LICENSE').write_bytes(license_raw)
    with (output / 'provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.source_dir, args.output_dir), indent=2))
    except (OSError, ValueError, KeyError) as error:
        print(f'OWASP preparation failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
