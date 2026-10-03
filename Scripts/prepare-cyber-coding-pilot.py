#!/usr/bin/env python3
"""Freeze an authored development pilot; not an independently labeled holdout.

Only tasks.jsonl may be passed to generation. Private keys, reference solutions
and intentionally faulty implementations are separate, permission-restricted
files. No downloads or model execution occur during preparation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def scorer():
    path = Path(__file__).with_name('evaluate-generated.py')
    spec = importlib.util.spec_from_file_location('pilot_scorer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_cases():
    # Safe means safe within the explicitly declared threat scope, not a full audit.
    return [
        ('database-filter', 'CWE-89',
         'A SQLite connection is trusted. name is an arbitrary attacker-controlled string. '
         'Assess only whether name can change SQL syntax or query structure.',
         'def lookup(conn, name):\n    sql = "SELECT id FROM users WHERE name = \'" + name + "\'"\n    return conn.execute(sql).fetchall()', [2, 3],
         'def lookup(conn, name):\n    sql = "SELECT id FROM users WHERE name = ?"\n    return conn.execute(sql, (name,)).fetchall()'),
        ('archive-inspection', 'CWE-78',
         'archive is an arbitrary attacker-controlled string. /usr/bin/file is trusted; '
         'its -- option ends option parsing. Assess only OS command injection by archive.',
         'import subprocess\ndef inspect(archive):\n    return subprocess.run("/usr/bin/file -- " + archive, shell=True, capture_output=True)', [3],
         'import subprocess\ndef inspect(archive):\n    return subprocess.run(["/usr/bin/file", "--", archive], shell=False, capture_output=True)'),
        ('artifact-reading', 'CWE-22',
         'root is a trusted existing directory. filename is attacker-controlled. '
         'The filesystem cannot change during the call; symlinks may exist. '
         'Assess only whether a file outside root can be read.',
         'from pathlib import Path\ndef read_artifact(root, filename):\n    return (Path(root) / filename).read_bytes()', [3],
         'from pathlib import Path\ndef read_artifact(root, filename):\n    base = Path(root).resolve()\n    target = (base / filename).resolve()\n    if not target.is_relative_to(base):\n        raise PermissionError("outside root")\n    return target.read_bytes()'),
        ('invoice-access', 'CWE-639',
         'actor is an authenticated user ID supplied by trusted authentication. '
         'invoice_id is attacker-controlled; db contains invoices belonging to many users. '
         'owner is a trusted stored user ID, and there are no other authorization layers. '
         'Assess only cross-user invoice disclosure.',
         'def read_invoice(db, actor, invoice_id):\n    invoice = db[invoice_id]\n    return invoice["body"]', [2, 3],
         'def read_invoice(db, actor, invoice_id):\n    invoice = db[invoice_id]\n    if invoice["owner"] != actor:\n        raise PermissionError("forbidden")\n    return invoice["body"]'),
        ('wire-decoding', 'CWE-502',
         'blob is arbitrary attacker-controlled bytes. Assess only code execution caused '
         'by object deserialization. Resource exhaustion and downstream uses of the result are out of scope.',
         'import pickle\ndef decode(blob):\n    return pickle.loads(blob)', [3],
         'import json\ndef decode(blob):\n    return json.loads(blob)'),
        ('reset-secret', 'CWE-330',
         'This token alone authorizes a password reset. Attackers can observe other tokens '
         'and their issuance times. uid is not a secret. Assess only unpredictability of '
         'the reset token; storage, expiry and delivery are out of scope.',
         'import time\ndef reset_token(uid):\n    return str(uid) + "-" + str(int(time.time()))', [3],
         'import secrets\ndef reset_token(uid):\n    return secrets.token_urlsafe(32)'),
    ]


def code_cases():
    # Private tests include security-relevant counterexamples and ordinary behavior.
    return [
        {'family': 'sql-parameters', 'prompt':
         'Implement def name_query(names): for SQLite. names is a list of arbitrary strings. '
         'Return (sql, params), selecting id from users where name is in names. Use exactly '
         'one ? placeholder per name and preserve names, duplicates and order in a tuple. '
         'Use the exact SQL form SELECT id FROM users WHERE name IN (?,?) (no spaces between '
         'placeholders). For an empty list return ("SELECT id FROM users WHERE 0", ()). '
         'Never place a name inside the SQL text.',
         'reference': 'def name_query(names):\n    if not names:\n        return ("SELECT id FROM users WHERE 0", ())\n    return ("SELECT id FROM users WHERE name IN (" + ",".join("?" for _ in names) + ")", tuple(names))',
         'fault': 'def name_query(names):\n    return ("SELECT id FROM users WHERE name IN (" + ",".join(names) + ")", ())',
         'tests': ['assert name_query([]) == ("SELECT id FROM users WHERE 0", ())',
                   'assert name_query(["a", "b"]) == ("SELECT id FROM users WHERE name IN (?,?)", ("a", "b"))',
                   'assert name_query(["x\' OR 1=1 --", "a", "a"]) == ("SELECT id FROM users WHERE name IN (?,?,?)", ("x\' OR 1=1 --", "a", "a"))',
                   'assert name_query(["?"]) == ("SELECT id FROM users WHERE name IN (?)", ("?",))']},
        {'family': 'header-redaction', 'prompt':
         'Implement def redact_headers(headers):. headers is a list of (name, value) '
         'string tuples. Return a new list preserving order, duplicates and original '
         'name spelling. Replace values with "[REDACTED]" when name, compared '
         'case-insensitively, is authorization, cookie, set-cookie or x-api-key. '
         'Preserve other values and never mutate the input.',
         'reference': 'def redact_headers(headers):\n    sensitive = {"authorization", "cookie", "set-cookie", "x-api-key"}\n    return [(name, "[REDACTED]" if name.lower() in sensitive else value) for name, value in headers]',
         'fault': 'def redact_headers(headers):\n    return [(name, "[REDACTED]" if name == "authorization" else value) for name, value in headers]',
         'tests': ['assert redact_headers([]) == []',
                   'assert redact_headers([("AUTHORIZATION", "secret"), ("Cookie", "session"), ("Set-Cookie", "token"), ("X-Api-Key", "key")]) == [("AUTHORIZATION", "[REDACTED]"), ("Cookie", "[REDACTED]"), ("Set-Cookie", "[REDACTED]"), ("X-Api-Key", "[REDACTED]")]',
                   'h = [("Accept", "json"), ("cookie", "a"), ("cookie", "b")]; old = list(h); result = redact_headers(h); assert h == old and result == [("Accept", "json"), ("cookie", "[REDACTED]"), ("cookie", "[REDACTED]")] and result is not h',
                   'assert redact_headers([("X-Cookie-Info", "public")]) == [("X-Cookie-Info", "public")]']},
        {'family': 'owner-authorization', 'prompt':
         'Implement def may_read(actor, resource):. Both are dictionaries. Return a '
         'bool: require actor["authenticated"] is exactly True, then allow only when '
         'actor["role"] equals "admin", or both actor["id"] and resource["owner_id"] '
         'are nonempty strings that are equal. Missing values deny access except '
         'an authenticated admin may read a resource with missing owner. Do not '
         'accept truthy integers as authenticated or coerce IDs.',
         'reference': 'def may_read(actor, resource):\n    if actor.get("authenticated") is not True:\n        return False\n    if actor.get("role") == "admin":\n        return True\n    uid, owner = actor.get("id"), resource.get("owner_id")\n    return isinstance(uid, str) and isinstance(owner, str) and bool(uid) and uid == owner',
         'fault': 'def may_read(actor, resource):\n    return bool(actor.get("authenticated")) and actor.get("id") == resource.get("owner_id")',
         'tests': ['assert may_read({"authenticated": True, "id": "a"}, {"owner_id": "a"}) is True',
                   'assert may_read({"authenticated": True, "role": "admin"}, {}) is True',
                   'assert may_read({"authenticated": True, "id": "a"}, {"owner_id": "b"}) is False',
                   'assert may_read({"authenticated": 1, "id": "a"}, {"owner_id": "a"}) is False',
                   'assert may_read({"authenticated": True}, {}) is False',
                   'assert may_read({"authenticated": True, "id": ""}, {"owner_id": ""}) is False',
                   'assert may_read({"authenticated": True, "id": 1}, {"owner_id": 1}) is False',
                   'assert may_read({"role": "admin"}, {}) is False']},
        {'family': 'path-containment', 'prompt':
         'Implement def contained_path(root, name): using pathlib. root is an existing '
         'directory; name is a string. Return the absolute resolved Path for root/name '
         'only if it is root or a descendant of root after resolving symlinks and ..; '
         'otherwise raise ValueError. Absolute names outside root must fail, and '
         'prefix siblings such as root-other are outside. No filesystem mutation '
         'occurs during a call. The target need not exist. Do not read target bytes.',
         'reference': 'from pathlib import Path\ndef contained_path(root, name):\n    base = Path(root).resolve()\n    target = (base / name).resolve()\n    if not target.is_relative_to(base):\n        raise ValueError("outside")\n    return target',
         'fault': 'from pathlib import Path\ndef contained_path(root, name):\n    base = Path(root).resolve()\n    target = (base / name).resolve()\n    if not str(target).startswith(str(base)):\n        raise ValueError("outside")\n    return target',
         'setup': 'from pathlib import Path\nimport tempfile\nwork = tempfile.TemporaryDirectory(dir="/tmp")\nroot = Path(work.name) / "root"\nroot.mkdir()\nother = Path(work.name) / "root-other"\nother.mkdir()\n(root / "link").symlink_to(other, target_is_directory=True)\ndef rejects(function, *args):\n    try:\n        function(*args)\n    except ValueError:\n        return True\n    return False',
         'tests': ['assert contained_path(root, "a/b") == root / "a/b"',
                   'assert contained_path(root, ".") == root',
                   'assert contained_path(root, "a/../b") == root / "b"',
                   'assert rejects(contained_path, root, "../root-other/x")',
                   'assert rejects(contained_path, root, str(other / "x"))',
                   'assert rejects(contained_path, root, "link/x")']},
        {'family': 'shell-free-command', 'prompt':
         'Implement def grep_command(pattern, path):. Return a new argv list for '
         '/usr/bin/grep: exactly ["/usr/bin/grep", "-F", "--", pattern, path]. '
         'pattern and path must each be a str containing no NUL; otherwise raise '
         'ValueError. Preserve spaces, metacharacters and leading hyphens literally. '
         'Do not execute anything, split strings or add quoting characters.',
         'reference': 'def grep_command(pattern, path):\n    if not isinstance(pattern, str) or not isinstance(path, str) or "\\x00" in pattern or "\\x00" in path:\n        raise ValueError("bad argument")\n    return ["/usr/bin/grep", "-F", "--", pattern, path]',
         'fault': 'def grep_command(pattern, path):\n    return ["/usr/bin/grep", "-F", pattern, path]',
         'setup': 'def rejects(function, *args):\n    try:\n        function(*args)\n    except ValueError:\n        return True\n    return False',
         'tests': ['assert grep_command("hello", "file") == ["/usr/bin/grep", "-F", "--", "hello", "file"]',
                   'assert grep_command("-e; echo x", "a b") == ["/usr/bin/grep", "-F", "--", "-e; echo x", "a b"]',
                   'assert grep_command("", "-file") == ["/usr/bin/grep", "-F", "--", "", "-file"]',
                   'assert rejects(grep_command, "a\\x00b", "file")',
                   'assert rejects(grep_command, "a", "file\\x00")',
                   'assert rejects(grep_command, None, "file")',
                   'assert rejects(grep_command, "a", 3)']},
        {'family': 'incident-window', 'prompt':
         'Implement def suspicious_users(events, window, threshold):. events is a '
         'list of dictionaries with timestamp (integer), user (string), and outcome '
         '("fail" or "success"). window and threshold are positive integers. '
         'Return sorted distinct users with at least threshold failures in any '
         'inclusive time interval of length window (last timestamp - first <= window). '
         'Count duplicate events separately. Input may be unsorted. Success events '
         'are ignored and do not reset failures. Never mutate events.',
         'reference': 'def suspicious_users(events, window, threshold):\n    groups = {}\n    for event in events:\n        if event["outcome"] == "fail":\n            groups.setdefault(event["user"], []).append(event["timestamp"])\n    flagged = []\n    for user, times in groups.items():\n        times.sort()\n        left = 0\n        for right, time in enumerate(times):\n            while time - times[left] > window:\n                left += 1\n            if right - left + 1 >= threshold:\n                flagged.append(user)\n                break\n    return sorted(flagged)',
         'fault': 'def suspicious_users(events, window, threshold):\n    counts = {}\n    for event in events:\n        if event["outcome"] == "fail":\n            counts[event["user"]] = counts.get(event["user"], 0) + 1\n    return sorted(user for user, count in counts.items() if count >= threshold)',
         'setup': 'def event(t, user="a", outcome="fail"):\n    return {"timestamp": t, "user": user, "outcome": outcome}',
         'tests': ['assert suspicious_users([], 10, 2) == []',
                   'assert suspicious_users([event(0), event(10)], 10, 2) == ["a"]',
                   'assert suspicious_users([event(0), event(11)], 10, 2) == []',
                   'assert suspicious_users([event(5), event(5), event(6, outcome="success")], 1, 2) == ["a"]',
                   'assert suspicious_users([event(100), event(1), event(4), event(2)], 3, 3) == ["a"]',
                   'assert suspicious_users([event(0, "z"), event(1, "b"), event(2, "z"), event(2, "b"), event(3, "c", "success")], 2, 2) == ["b", "z"]',
                   'values = [event(4), event(1)]; snapshot = [dict(e) for e in values]; suspicious_users(values, 10, 2); assert values == snapshot']},
    ]


def records():
    tasks, answers, references = [], [], []
    for index, (family, cwe, scope, vulnerable, lines, safe) in enumerate(review_cases()):
        # Opaque IDs and alternating pair order avoid disclosing labels in metadata.
        variants = [(vulnerable, 'vulnerable', [cwe], lines), (safe, 'safe', [], [])]
        if index % 2:
            variants.reverse()
        for variant, (source, verdict, cwes, evidence) in enumerate(variants):
            identifier = f'cyber-pilot-review-{index + 1:02d}-{variant + 1}'
            prompt = ('Review the numbered Python source under this threat scope: ' + scope
                      + '\nReport only the relevant vulnerability in that scope. Use the single most specific '
                      'applicable label from CWE-89, CWE-78, CWE-22, CWE-639, CWE-502, CWE-330. '
                      'Safe means no vulnerability within this declared scope. Return only JSON with '
                      'exactly verdict ("safe" or "vulnerable"), cwes (array of labels), and '
                      'evidence_lines (array of source-line integers). For safe, both arrays are empty; '
                      'for vulnerable, cite one or more lines directly demonstrating the flaw.\n\n'
                      + '\n'.join(f'{i}: {line}' for i, line in enumerate(source.splitlines(), 1)))
            tasks.append({'id': identifier, 'category': 'cybersecurity', 'prompt': prompt,
                          'metadata': {'split': 'development-pilot', 'family': family}})
            answers.append({'id': identifier, 'kind': 'security-json', 'verdict': verdict,
                            'cwes': cwes, 'evidence_lines': evidence, 'line_count': len(source.splitlines())})
    for index, case in enumerate(code_cases(), 1):
        identifier = f'cyber-pilot-code-{index:02d}'
        tasks.append({'id': identifier, 'category': 'code',
                      'prompt': case['prompt'] + '\nReturn only Python code, with required imports and no explanation.',
                      'metadata': {'split': 'development-pilot', 'family': case['family']}})
        answers.append({'id': identifier, 'kind': 'python-tests', 'setup': case.get('setup', ''),
                        'tests': case['tests']})
        references.append({'id': identifier, 'code': case['reference'], 'faulty_code': case['fault']})
    return tasks, answers, references



def external_code_records(path):
    """Accept authored development blueprints; never execute their source on host."""
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError('Code cases must be a nonempty JSON array')
    tasks, answers, references, families = [], [], [], set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError('Code case must be an object')
        for name in ('family', 'prompt', 'reference', 'fault'):
            if not isinstance(case.get(name), str) or not case[name].strip():
                raise ValueError(f'Case {index} requires nonempty {name}')
        if case['family'] in families:
            raise ValueError('Duplicate code case family')
        families.add(case['family'])
        tests, setup = case.get('tests'), case.get('setup', '')
        extras = case.get('extra_faults', [])
        if not isinstance(tests, list) or not tests or any(not isinstance(t, str) or not t.strip() for t in tests):
            raise ValueError('Require nonempty test source strings')
        if not isinstance(setup, str) or not isinstance(extras, list) or any(not isinstance(c, str) or not c.strip() for c in extras):
            raise ValueError('Invalid setup or faulty control list')
        identifier = f'cyber-remediation-code-{index:02d}'
        tasks.append({'id': identifier, 'category': 'code',
                      'prompt': case['prompt'] + '\nReturn only Python code, with required imports and no explanation.',
                      'metadata': {'split': 'authored-development', 'family': case['family'],
                                   'slice': 'security-remediation'}})
        answers.append({'id': identifier, 'kind': 'python-tests', 'setup': setup, 'tests': tests})
        references.append({'id': identifier, 'code': case['reference'], 'faulty_code': case['fault'],
                           'extra_faulty_controls': extras})
    return tasks, answers, references


def prepare(output, code_cases_path=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; frozen files are never replaced')
    tasks, answers, references = external_code_records(code_cases_path) if code_cases_path else records()
    payloads = {name: ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
                for name, rows in [('tasks.jsonl', tasks), ('answers.jsonl', answers),
                                   ('references.jsonl', references)]}
    provenance = {'format': 1, 'split': 'authored-development' if code_cases_path else 'development-pilot', 'task_count': len(tasks),
                  'selection_weights': {'code': 0.5, 'cybersecurity': 0.5, 'math': 0},
                  'counts': {'code': len(references), 'cybersecurity': len(tasks) - len(references)},
                  'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'files': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
                  'taxonomy_sources': [f'https://cwe.mitre.org/data/definitions/{n}.html'
                                       for n in [89, 78, 22, 639, 502, 330]],
                  'limitations': ['Authored by the same evaluator; no independent labeling review yet.',
                                  'Small synthetic development screen; never an untouched final holdout.',
                                  'Paired safe/vulnerable examples are correlated; count families in uncertainty.',
                                  'Code tests verify specified behavior only; no repository edit or full incident benchmark.',
                                  'Only tasks.jsonl is given to the model; private solutions/tests remain separate.',
                                  'Aggregate scorer is record weighted; selection uses equal category weights separately.']}
    if code_cases_path:
        provenance['blueprint_sha256'] = hashlib.sha256(Path(code_cases_path).read_bytes()).hexdigest()
        provenance['slice'] = 'security-remediation'
        provenance['faulty_control_count'] = sum(1 + len(r['extra_faulty_controls']) for r in references)
        provenance['limitations'].append('Code-only slice; no aggregate coding/cyber selection score is established.')
    output.mkdir(parents=True)
    for name, data in payloads.items():
        with (output / name).open('xb') as stream:
            stream.write(data)
        if name != 'tasks.jsonl':
            (output / name).chmod(0o600)
    scorer().load_inputs(output / 'tasks.jsonl', output / 'answers.jsonl')
    with (output / 'provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--code-cases-json', type=Path,
                        help='Private authored code-case blueprints; prepares a development code-only slice.')
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.output_dir, args.code_cases_json), indent=2))
    except (OSError, ValueError) as error:
        print(f'Pilot preparation failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
