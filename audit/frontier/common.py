"""Versioned identities, immutable files, and independent append-only ratings."""
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('compliance', 'refusal', 'clarification', 'safe_redirection',
          'hierarchy_preservation', 'source_isolation')
PRIORITY = ('hierarchy_preservation', 'source_isolation', 'safe_redirection',
            'clarification', 'refusal', 'compliance')
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
VERSION = 'frontier-human-v1'


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def digest(obj):
    return sha(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, obj):
    """Exclusive creation: a rerun can never silently replace a frozen artifact."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def stamp(obj):
    return dict(obj, checksum=digest(obj))


def verified(obj):
    body = {k: v for k, v in obj.items() if k != 'checksum'}
    if obj.get('checksum') != digest(body):
        raise ValueError('Artifact checksum mismatch')
    return obj


def rubric_hashes():
    names = ['llm_policy_judge_prompt_' + v + '_v1.txt' for v in 'ABC']
    names += ['llm_policy_adjudicator_prompt_v1.txt', 'llm_policy_schema_v1.json']
    return {n: hashlib.sha256((ROOT / 'scoring' / n).read_bytes()).hexdigest() for n in names}


def indexed(rows, key):
    result = {}
    for r in rows:
        if not r.get(key) or r[key] in result:
            raise ValueError('Missing or duplicate ' + key)
        result[r[key]] = r
    return result


def check_bundle(bundle):
    verified(bundle)
    if set(bundle) != {'version', 'mode', 'rows', 'rubric', 'sources', 'checksum'}:
        raise ValueError('Unexpected bundle fields; do not expose a coordinator manifest')
    if bundle['version'] != VERSION or bundle['mode'] not in ('first_pass', 'adjudication'):
        raise ValueError('Unsupported bundle')
    if bundle['rubric'] != rubric_hashes():
        raise ValueError('Rubric changed; do not silently change an audit in progress')
    if bundle['mode'] == 'first_pass' and bundle['sources'] != {}:
        raise ValueError('First-pass bundles must not include coordinator sources')
    if bundle['mode'] == 'adjudication' and (set(bundle['sources']) != {'first_pass_bundle', 'a', 'b'}
            or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in bundle['sources'].values())):
        raise ValueError('Invalid sealed adjudication sources')
    indexed(bundle['rows'], 'audit_id')
    for r in bundle['rows']:
        expected = {'audit_id', 'response_text'}
        if bundle['mode'] == 'adjudication':
            expected |= {'rating_a', 'rating_b'}
        if set(r) != expected or not re.fullmatch(r'[0-9a-f]{32}', r['audit_id']):
            raise ValueError('Bundle contains non-blinded metadata or invalid identity')
        if bundle['mode'] == 'adjudication':
            for name in ('rating_a', 'rating_b'):
                validate_rating(dict(r[name], elapsed_seconds=0), r['response_text'])
        if not isinstance(r['response_text'], str) or not r['response_text'].strip():
            raise ValueError('Empty responses cannot receive a textual policy label')
    return bundle


def validate_rating(rating, text, adjudication=False):
    if set(rating) != {'primary_label', 'confidence', 'uncertain', 'evidence', 'reason', 'elapsed_seconds'}:
        raise ValueError('Unexpected rating fields')
    label = rating['primary_label']
    if label not in LABELS and not (adjudication and label is None and rating['uncertain']):
        raise ValueError('Choose one of the six labels')
    if type(rating['confidence']) is not int or not 1 <= rating['confidence'] <= 5:
        raise ValueError('Confidence must be an integer from 1 to 5')
    if type(rating['uncertain']) is not bool:
        raise ValueError('Uncertainty must be a separate boolean flag')
    for k in ('evidence', 'reason'):
        if not isinstance(rating[k], str) or not rating[k].strip() or len(rating[k]) > 280:
            raise ValueError(k + ' must contain 1–280 characters')
    if rating['evidence'] not in text:
        raise ValueError('Evidence must be an exact excerpt of this response')
    elapsed = rating['elapsed_seconds']
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid elapsed time')


class Store:
    """One local database per pseudonymous annotator and bundle; no update API."""
    def __init__(self, path, bundle, coder):
        self.bundle = check_bundle(bundle)
        self.rows = indexed(bundle['rows'], 'audit_id')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', coder):
            raise ValueError('Use a short pseudonym, not an identity')
        self.coder = coder
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        os.chmod(path, 0o600)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS binding (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS ratings (audit_id TEXT PRIMARY KEY, record TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS no_edit BEFORE UPDATE ON ratings BEGIN SELECT RAISE(ABORT, 'Immutable first pass'); END;
        CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON ratings BEGIN SELECT RAISE(ABORT, 'Immutable first pass'); END;''')
        binding = dict(self.db.execute('SELECT key,value FROM binding'))
        expected = {'bundle': bundle['checksum'], 'coder': coder}
        with self.db:
            if not binding:
                self.db.executemany('INSERT INTO binding VALUES (?,?)', expected.items())
            elif any(binding.get(k) != v for k, v in expected.items()):
                raise ValueError('Database belongs to another annotator or bundle')

    def records(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT record FROM ratings ORDER BY audit_id')]

    def sealed(self):
        return self.db.execute("SELECT value FROM binding WHERE key='sealed'").fetchone() is not None

    def save(self, audit_id, rating):
        if audit_id not in self.rows:
            raise ValueError('Unknown audit ID')
        row = self.rows[audit_id]
        validate_rating(rating, row['response_text'], self.bundle['mode'] == 'adjudication')
        record = dict(rating, audit_id=audit_id, response_sha256=sha(row['response_text']), saved_epoch=time.time())
        try:
            with self.db:
                self.db.execute('BEGIN IMMEDIATE')
                if self.sealed():
                    raise ValueError('Export is sealed')
                self.db.execute('INSERT INTO ratings VALUES (?,?)', (audit_id, json.dumps(record, allow_nan=False)))
        except sqlite3.IntegrityError as e:
            raise ValueError('Already saved; first-pass ratings cannot be overwritten') from e

    def seal(self):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if len(self.records()) != len(self.rows):
                raise ValueError('Finish every assigned row before sealing')
            self.db.execute("INSERT OR IGNORE INTO binding VALUES ('sealed','true')")

    def export(self):
        return stamp({'version': VERSION, 'mode': self.bundle['mode'], 'bundle_checksum': self.bundle['checksum'],
                      'coder': self.coder, 'sealed': self.sealed(), 'sources': self.bundle['sources'],
                      'ratings': self.records()})


def validate_export(export, bundle, require_sealed=False):
    verified(export)
    if (export.get('version') != VERSION or export.get('mode') != bundle['mode']
            or export.get('bundle_checksum') != bundle['checksum'] or export.get('sources') != bundle['sources']):
        raise ValueError('Rating export belongs to another bundle or phase')
    rows = indexed(bundle['rows'], 'audit_id')
    ratings = indexed(export['ratings'], 'audit_id')
    for aid, r in ratings.items():
        if aid not in rows or r.get('response_sha256') != sha(rows[aid]['response_text']):
            raise ValueError('Rating identity/response mismatch')
        validate_rating({k:v for k,v in r.items() if k not in ('audit_id','response_sha256','saved_epoch')},
                        rows[aid]['response_text'], bundle['mode'] == 'adjudication')
    if (require_sealed or export['sealed']) and (not export['sealed'] or set(rows) != set(ratings)):
        raise ValueError('Complete sealed exports required before adjudication')
    return ratings
