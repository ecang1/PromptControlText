"""Offline CLI: verified frames, immutable samples, and canonical judge adapters."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import (ROOT, VERSION, MODEL, LABELS, sha, digest, read, immutable,
                     stamp, verified, indexed, rubric_hashes, check_bundle, validate_export)


def build_frame(run):
    # Import only offline case construction; never instantiate a client or ledger.
    from frontier.main_run import main_cases, validate_main
    manifest = read(ROOT / 'artifacts/frontier/manifest.json')
    config = read(ROOT / 'configs/frontier-main.json')
    validate_main(config, manifest, read(ROOT / 'configs/frontier-pilot.json'))
    expected = {c['request_id']: c for c in main_cases(manifest, config)}
    run = Path(run)
    progress = read(run / 'progress.json')
    metadata = indexed([json.loads(x) for x in (run / 'requests.jsonl').read_text().splitlines()], 'request_id')
    if not set(metadata) <= set(expected):
        raise ValueError('Foreign requests in publication')
    rows, excluded, seen, files = [], Counter(), set(), {}
    for path in sorted((run / 'responses').glob('*.jsonl')):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            r = json.loads(line); rid = r['request_id']; case = r['case']; result = r['result']
            if rid in seen or rid not in expected or case != expected[rid]:
                raise ValueError('Duplicate or changed frozen case')
            seen.add(rid)
            m = metadata[rid]
            if m['state'] != 'done' or m['response_text_sha256'] != sha(result['text']):
                raise ValueError('Result has not been verified as complete')
            if m['outcome'] != result['outcome'] or m['request_body_sha256'] != case['request_body_sha256']:
                raise ValueError('Result metadata mismatch')
            if result['outcome'] not in ('text', 'truncated') or not result['text'].strip():
                excluded[result['outcome']] += 1
                continue
            item = case['item']; condition = case['condition']
            rows.append({'request_id': rid, 'response_text': result['text'], 'response_sha256': sha(result['text']),
                         'provider': case['model']['provider'], 'benchmark': item['benchmark'],
                         'stratum': item['stratum'], 'item_id': item['item_id'],
                         'parent_id': item['benchmark'] + ':' + str(item.get('group_id') or item['item_id']),
                         'condition_id': condition['condition_id'], 'family': condition['family'],
                         'truncated': bool(result.get('truncated'))})
    if len(seen) != progress['completed_cases']:
        raise ValueError('Published full-record coverage differs from progress')
    complete = progress['main']['complete'] and len(seen) == len(expected) == 10950
    return stamp({'version': VERSION, 'complete': complete, 'expected_cases': len(expected),
                  'accounted_cases': len(seen), 'excluded_nontext': dict(excluded), 'rows': sorted(rows, key=lambda r:r['request_id']),
                  'provenance': {'progress': digest(progress), 'metadata': hashlib.sha256((run/'requests.jsonl').read_bytes()).hexdigest(),
                                 'record_files': files, 'manifest': digest(manifest), 'config': digest(config)},
                  'rubric': rubric_hashes()})


def rank(seed, key):
    return sha(seed + '\0' + key)


def package(rows, component, seed, frame, settings, sources=None):
    data = []
    for row in rows:
        r = dict(row)
        r['audit_id'] = rank(seed + ':opaque', r['request_id'])[:32]
        r['component'] = component
        data.append(r)
    data.sort(key=lambda r:rank(seed + ':order', r['audit_id']))
    indexed(data, 'audit_id'); indexed(data, 'request_id')
    m = stamp({'version': VERSION, 'frame_checksum': frame['checksum'], 'frame_complete': frame['complete'],
               'component': component, 'seed': seed, 'settings': settings, 'sources': sources or {},
               'rubric': frame['rubric'], 'rows': data})
    b = stamp({'version': VERSION, 'mode': 'first_pass', 'rubric': frame['rubric'], 'sources': {},
               'rows': [{'audit_id': r['audit_id'], 'response_text': r['response_text']} for r in data]})
    check_bundle(b)
    return m, b


def write_package(path, pair):
    # Fail before writing if the directory already exists, even if empty.
    p = Path(path); p.mkdir(parents=True, exist_ok=False)
    immutable(p/'manifest.private.json', pair[0]); immutable(p/'bundle.json', pair[1])


def practice(frame, seed, n=30):
    verified(frame)
    if n < 1:
        raise ValueError('Practice size must be positive')
    groups = defaultdict(list)
    for r in frame['rows']:
        groups[r['provider']].append(r)
    selected, used = [], set()
    # Round-robin by provider; exact duplicate texts appear only once in practice.
    queues = {p:sorted(rs,key=lambda r:rank(seed,r['request_id'])) for p,rs in groups.items()}
    while len(selected) < n:
        advanced = False
        for p in sorted(queues):
            while queues[p] and queues[p][0]['response_sha256'] in used:
                queues[p].pop(0)
            if queues[p] and len(selected) < n:
                r = queues[p].pop(0);selected.append(r);used.add(r['response_sha256']);advanced=True
        if not advanced:
            raise ValueError('Insufficient distinct practice responses')
    return package(selected, 'practice', seed, frame, {'n':n, 'purpose':'unscored timing and calibration'})


def exclude(rows, manifests):
    ids, texts = set(), set()
    for m in manifests:
        verified(m)
        ids.update(r['request_id'] for r in m['rows']);texts.update(r['response_sha256'] for r in m['rows'])
    return [r for r in rows if r['request_id'] not in ids and r['response_sha256'] not in texts]


def allocation(sizes, n):
    """One per nonempty condition, then proportional remaining-capacity apportionment."""
    if n < len(sizes) or n > sum(sizes.values()):
        raise ValueError('Sample must cover each available condition and fit the frame')
    out = {k:1 for k in sizes}; remaining = n-len(sizes)
    capacity = sum(v-1 for v in sizes.values())
    if remaining:
        quota = {k:remaining*(v-1)/capacity for k,v in sizes.items()}
        for k,v in quota.items():out[k] += int(v)
        left = n-sum(out.values())
        for k in sorted(sizes, key=lambda k:(-(quota[k]-int(quota[k])), k))[:left]:out[k] += 1
    return out


def representative(frame, calibration, seed, per_provider=100):
    verified(frame);verified(calibration)
    if not frame['complete']:
        raise ValueError('Scored sampling requires the final complete generation frame')
    if calibration['component'] != 'practice':
        raise ValueError('Expected immutable practice manifest')
    frame_by_id = indexed(frame['rows'], 'request_id')
    for r in calibration['rows']:
        if r['request_id'] not in frame_by_id or frame_by_id[r['request_id']]['response_sha256'] != r['response_sha256']:
            raise ValueError('Practice does not map to this generation frame')
    eligible = exclude(frame['rows'], [calibration]); selected=[]; strata=[];coverage={}
    for provider in sorted({r['provider'] for r in eligible}):
        groups=defaultdict(list)
        for r in eligible:
            if r['provider']==provider:groups[r['condition_id']].append(r)
        counts=allocation({k:len(v) for k,v in groups.items()},per_provider)
        for condition, rs in sorted(groups.items()):
            n=counts[condition];N=len(rs)
            strata.append({'provider':provider,'condition_id':condition,'N':N,'n':n})
            for r in sorted(rs,key=lambda r:rank(seed,r['request_id']))[:n]:
                selected.append(dict(r,inclusion_probability=n/N,weight=N/n))
        chosen=[r for r in selected if r['provider']==provider]
        coverage[provider]={'conditions':len({r['condition_id'] for r in chosen}),
                            'benchmark_strata':dict(Counter(r['stratum'] for r in chosen)),
                            'missing_benchmark_strata':sorted({r['stratum'] for r in eligible if r['provider']==provider}-{r['stratum'] for r in chosen})}
    return package(selected,'representative',seed,frame,
                   {'per_provider':per_provider,'eligible_N':len(eligible),'excluded_practice_texts':len(frame['rows'])-len(eligible),
                    'strata':strata,'coverage':coverage,
                    'estimand':'Text responses in the final frozen study, excluding every exact practice-text match; not all benchmark items.'},
                   {'practice':calibration['checksum']})


def check_judges(judges, frame):
    verified(frame);verified(judges)
    if judges['model'] != MODEL or judges['rubric'] != frame['rubric'] or judges['frame_checksum'] != frame['checksum']:
        raise ValueError('Canonical judge model, rubric or frame mismatch')
    expected=indexed(frame['rows'],'request_id');rows=indexed(judges['rows'],'request_id')
    if set(rows)!=set(expected):raise ValueError('Canonical judge mapping must cover every text request; use null labels for unresolved rows')
    for rid,r in rows.items():
        if r['response_sha256']!=expected[rid]['response_sha256'] or r['primary_label'] not in (*LABELS,None):
            raise ValueError('Invalid canonical identity/label')
        if not 0<=r['confidence']<=1 or type(r['needs_review']) is not bool:
            raise ValueError('Invalid canonical confidence/review flag')
        if len(r['votes'])!=3 or any(x not in (*LABELS,None) for x in r['votes']):
            raise ValueError('Expected A/B/C votes, with null for parse errors')
    return rows


def targeted(frame, calibration, representative_manifest, judges, seed, n=60):
    if n<0:raise ValueError('Targeted size cannot be negative')
    j=check_judges(judges,frame);verified(representative_manifest)
    if representative_manifest['frame_checksum']!=frame['checksum'] or representative_manifest['component']!='representative':
        raise ValueError('Representative sample belongs to another frame')
    if representative_manifest['sources'].get('practice')!=calibration['checksum']:
        raise ValueError('Practice mismatch')
    eligible=exclude(frame['rows'],[calibration,representative_manifest]);counts=Counter(x['primary_label'] for x in j.values() if x['primary_label'])
    candidates=[]
    for r in eligible:
        q=j[r['request_id']];why=[]
        if q['primary_label'] and counts[q['primary_label']]/len(j)<.05:why.append('rare_label')
        if q['needs_review'] or q['confidence']<.8 or q['primary_label'] is None:why.append('uncertainty')
        if len(set(q['votes']))>1:why.append('ABC_disagreement')
        if why:candidates.append(dict(r,target_reasons=why))
    chosen=sorted(candidates,key=lambda r:rank(seed,r['request_id']))[:n]
    for r in chosen:r.update(inclusion_probability=len(chosen)/len(candidates),weight=None)
    return package(chosen,'targeted',seed,frame,{'maximum_n':n,'candidate_N':len(candidates),'rare_threshold':.05,'confidence_threshold':.8,
                   'estimand':'Conditional diagnostic queue only; never combine with representative estimates'},
                   {'practice':calibration['checksum'],'representative':representative_manifest['checksum'],'judges':judges['checksum']})


def judge_input(frame, out):
    verified(frame)
    if not frame['complete']:raise ValueError('Use final frame for canonical judge handoff')
    unique={sha(r['response_text'].strip()):r['response_text'].strip() for r in frame['rows']}
    with Path(out).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['item_id','model_output']);w.writeheader()
        for key,text in sorted(unique.items()):w.writerow({'item_id':key,'model_output':text})
    # No inference is launched. The existing judge strips outer whitespace; raw human text is retained.


def import_judge(frame, directory):
    verified(frame)
    if not frame['complete']:raise ValueError('Canonical import requires the final frame')
    directory=Path(directory);m=read(directory/'manifest.json')
    if m['model']!=MODEL or m['schema_version']!='v1' or m['prompt_set_version']!='v1':
        raise ValueError('Not canonical Llama 3.1 8B v1 outputs')
    for variant in 'ABC':
        if (directory/('judge_prompt_'+variant+'.txt')).read_text().strip()!=(ROOT/'scoring'/('llm_policy_judge_prompt_'+variant+'_v1.txt')).read_text().strip():
            raise ValueError('Judge prompt changed')
    if (directory/'adjudicator_prompt.txt').read_text().strip()!=(ROOT/'scoring/llm_policy_adjudicator_prompt_v1.txt').read_text().strip():
        raise ValueError('Adjudicator prompt changed')
    with (directory/'labeled.csv').open(newline='') as f: labeled=list(csv.DictReader(f))
    labels=indexed(labeled,'item_id');expected={sha(r['response_text'].strip()) for r in frame['rows']}
    if set(labels)!=expected:raise ValueError('Judge handoff identity coverage mismatch')
    votes=defaultdict(dict)
    with (directory/'judge_votes.csv').open(newline='') as f:
        for v in csv.DictReader(f):
            key=v['row_hash'];variant=v['judge_prompt_variant']
            if key not in expected or variant not in 'ABC' or len(variant)!=1 or variant in votes[key] or v['judge_model']!=MODEL:
                raise ValueError('Duplicate, foreign or noncanonical judge vote')
            votes[key][variant]=v['primary_label'] if v['primary_label'] in LABELS else None
    rows=[]
    for r in frame['rows']:
        key=sha(r['response_text'].strip());v=labels[key]
        if sha(v['model_output'])!=key or v['llm_judge_model']!=MODEL or v['llm_adjudicator_model']!=MODEL:
            raise ValueError('Judge output text/model mismatch')
        if set(votes[key])!=set('ABC'):raise ValueError('Missing A/B/C vote evidence')
        label=v['llm_policy_label'] if v['llm_policy_label'] in LABELS else None
        rows.append({'request_id':r['request_id'],'response_sha256':r['response_sha256'],'primary_label':label,
                     'confidence':float(v['llm_confidence'] or 0),'needs_review':v['llm_needs_human_audit'].lower()=='true' or label is None,
                     'votes':[votes[key][x] for x in 'ABC']})
    obj=stamp({'model':MODEL,'frame_checksum':frame['checksum'],'rubric':frame['rubric'],'rows':rows,
               'provenance':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file() and p.suffix in ('.csv','.txt','.json')}})
    check_judges(obj,frame);return obj


def adjudication_bundle(bundle, a, b):
    check_bundle(bundle)
    if bundle['mode']!='first_pass' or a['coder']==b['coder']:raise ValueError('Two distinct independent first passes required')
    ra=validate_export(a,bundle,True);rb=validate_export(b,bundle,True)
    rows=[]
    # Review every row, including agreement: accept the consensus explicitly or mark unresolved.
    for r in bundle['rows']:
        public=lambda rating:{k:rating[k] for k in ['primary_label','confidence','uncertain','evidence','reason']}
        rows.append(dict(r,rating_a=public(ra[r['audit_id']]),rating_b=public(rb[r['audit_id']])))
    return stamp({'version':VERSION,'mode':'adjudication','rubric':bundle['rubric'],
                  'sources':{'first_pass_bundle':bundle['checksum'],'a':a['checksum'],'b':b['checksum']},'rows':rows})


def make_kit(bundle_path, out):
    import shutil
    bundle = check_bundle(read(bundle_path))
    p = Path(out); p.mkdir(parents=True, exist_ok=False)
    immutable(p/'bundle.json', bundle)
    names = ['audit/__init__.py', 'audit/frontier/__init__.py', 'audit/frontier/common.py',
             'audit/frontier/dashboard.py', 'audit/frontier/dashboard.html']
    names += ['scoring/'+name for name in rubric_hashes()]
    for name in names:
        target=p/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    shutil.copyfile(ROOT/'docs/frontier-audit/ANNOTATOR_GUIDE.md',p/'ANNOTATOR_GUIDE.md')
    (p/'START.txt').write_text('Read ANNOTATOR_GUIDE.md. Use your own copy of this folder.\n'
        'Run: python3 -m audit.frontier.dashboard --bundle bundle.json --database .local/ratings.sqlite --coder YOUR_PSEUDONYM\n'
        'Open http://127.0.0.1:8765. Keep the same database and pseudonym to resume.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('frame');p.add_argument('--run',default='artifacts/frontier/runs/2026-10-08-main');p.add_argument('--out',required=True)
    for name in ['practice','representative','targeted']:
        p=sub.add_parser(name);p.add_argument('--frame',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',required=True)
        if name=='practice':p.add_argument('--n',type=int,default=30)
        else:p.add_argument('--practice',required=True)
        if name=='representative':p.add_argument('--per-provider',type=int,default=100)
        if name=='targeted':
            p.add_argument('--representative',required=True);p.add_argument('--judges',required=True);p.add_argument('--n',type=int,default=60)
    p=sub.add_parser('judge-input');p.add_argument('--frame',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('import-judge');p.add_argument('--frame',required=True);p.add_argument('--directory',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('kit');p.add_argument('--bundle',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('adjudication');p.add_argument('--bundle',required=True);p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args();cmd=args.command
    if cmd=='frame':immutable(args.out,build_frame(args.run))
    elif cmd=='practice':write_package(args.out,practice(read(args.frame),args.seed,args.n))
    elif cmd=='representative':write_package(args.out,representative(read(args.frame),read(args.practice),args.seed,args.per_provider))
    elif cmd=='targeted':write_package(args.out,targeted(read(args.frame),read(args.practice),read(args.representative),read(args.judges),args.seed,args.n))
    elif cmd=='judge-input':judge_input(read(args.frame),args.out)
    elif cmd=='kit':make_kit(args.bundle,args.out)
    elif cmd=='import-judge':immutable(args.out,import_judge(read(args.frame),args.directory))
    else:immutable(args.out,adjudication_bundle(read(args.bundle),read(args.a),read(args.b)))
    print('Created immutable artifact:',args.out)

if __name__=='__main__':main()
