"""Two independent first passes, separate adjudication, and optional canonical labels."""
import argparse
import random
import math
from collections import Counter, defaultdict

from .common import LABELS, VERSION, digest, read, immutable, verified, indexed, check_bundle, validate_export
from .sampling import adjudication_bundle, check_judges


def agreement(a, b, weights=None):
    if len(a)!=len(b):raise ValueError('Unequal rating lengths')
    weights=[1.0]*len(a) if weights is None else weights
    if len(weights)!=len(a) or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0 for w in weights):raise ValueError('Invalid weights')
    if any(x not in LABELS for x in a+b):raise ValueError('Unknown policy')
    total=sum(weights)
    if not total:return {'n':0,'agreement':None,'kappa':None,'degenerate':'no paired ratings'}
    pa=Counter();pb=Counter();matched=0
    for x,y,w in zip(a,b,weights):pa[x]+=w;pb[y]+=w;matched+=w*(x==y)
    observed=matched/total;expected=sum(pa[x]*pb[x] for x in LABELS)/total**2
    return {'n':len(a),'agreement':observed,'kappa':(observed-expected)/(1-expected) if expected<1-1e-12 else None,
            'degenerate':'constant identical marginals; kappa undefined' if expected>=1-1e-12 else None}


def confusion(gold, predicted, weights=None):
    if len(gold)!=len(predicted):raise ValueError('Unequal label lengths')
    weights=[1.0]*len(gold) if weights is None else weights
    if len(weights)!=len(gold) or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0 for w in weights):raise ValueError('Invalid weights')
    matrix={a:{b:0.0 for b in LABELS} for a in LABELS};raw={a:{b:0 for b in LABELS} for a in LABELS}
    for g,p,w in zip(gold,predicted,weights):
        if g not in LABELS or p not in LABELS:raise ValueError('Unknown policy')
        matrix[g][p]+=w;raw[g][p]+=1
    classes={}
    for label in LABELS:
        support=sum(matrix[label].values());positive=sum(matrix[g][label] for g in LABELS)
        count=sum(raw[label].values());pred_count=sum(raw[g][label] for g in LABELS)
        classes[label]={'human_n':count,'judge_n':pred_count,'weighted_human_total':support,
                        'precision':matrix[label][label]/positive if positive else None,
                        'recall':matrix[label][label]/support if support else None,
                        'flag':'absent human class' if not count else ('rare human class (<10)' if count<10 else None)}
    total=sum(weights)
    return {'n':len(gold),'accuracy':sum(matrix[x][x] for x in LABELS)/total if total else None,
            'orientation':'rows = adjudicated human; columns = canonical judge',
            'weighted_matrix':matrix,'count_matrix':raw,'per_class':classes}


def quantile(values, p):
    values=sorted(values)
    if not values:return None
    at=(len(values)-1)*p;lo=int(at);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def scores(rows):
    paired=[r for r in rows if r['a'] is not None and r['b'] is not None]
    resolved=[r for r in rows if r['gold'] is not None and r['judge'] is not None]
    ag=agreement([r['a'] for r in paired],[r['b'] for r in paired],[r['w'] for r in paired])
    cf=confusion([r['gold'] for r in resolved],[r['judge'] for r in resolved],[r['w'] for r in resolved])
    result={'human_agreement':ag['agreement'],'human_kappa':ag['kappa'],'judge_accuracy':cf['accuracy']}
    for label,v in cf['per_class'].items():
        result[label+'_precision']=v['precision'];result[label+'_recall']=v['recall']
    return result


def cluster_intervals(rows, repetitions=1000, seed=20261008):
    if repetitions<1:return {'reason':'bootstrap disabled','intervals':{}}
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[r['benchmark']][r['parent']].append(r)
    if not groups or any(len(v)<2 for v in groups.values()):
        return {'reason':'fewer than two sampled parents in at least one benchmark; no reliable cluster interval','intervals':{}}
    rng=random.Random(seed);values=defaultdict(list)
    for _ in range(repetitions):
        sample=[]
        for benchmark in sorted(groups):
            parents=sorted(groups[benchmark])
            for parent in rng.choices(parents,k=len(parents)):sample.extend(groups[benchmark][parent])
        for k,v in scores(sample).items():
            if v is not None:values[k].append(v)
    return {'method':'Approximate percentile bootstrap of benchmark parents within benchmark, jointly across providers and conditions; inclusion weights retained. Not exact finite-population design intervals.',
            'seed':seed,'repetitions':repetitions,'parents_by_benchmark':{k:len(v) for k,v in groups.items()},
            'intervals':{k:{'lower':quantile(v,.025),'upper':quantile(v,.975),'valid_replicates':len(v),
                            'warning':'many degenerate replicates' if len(v)<.9*repetitions else None} for k,v in values.items()}}


def analyze(manifest, bundle, a, b, adjudication=None, judges=None, frame=None, repetitions=1000, seed=20261008):
    verified(manifest);check_bundle(bundle)
    if manifest['component'] not in ('practice','representative','targeted'):
        raise ValueError('Unknown sample component')
    if bundle['mode']!='first_pass' or a['coder']==b['coder']:raise ValueError('Independent pseudonyms required')
    m=indexed(manifest['rows'],'audit_id');br=indexed(bundle['rows'],'audit_id')
    if set(m)!=set(br) or any(m[k]['response_text']!=br[k]['response_text'] for k in m):
        raise ValueError('Manifest/bundle identity mismatch')
    ra=validate_export(a,bundle);rb=validate_export(b,bundle);adj={}
    if adjudication is not None:
        adj_bundle=adjudication_bundle(bundle,a,b)
        if adjudication['coder'] in (a['coder'],b['coder']):raise ValueError('Use a distinct adjudicator pseudonym')
        adj=validate_export(adjudication,adj_bundle)
    j={}
    if judges is not None:
        if frame is None or manifest['frame_checksum']!=frame['checksum']:raise ValueError('Judge comparison requires the matching frame')
        j=check_judges(judges,frame)
    rows=[]
    for aid,r in m.items():
        w=r.get('weight') if manifest['component']=='representative' else 1.0
        if (w is None or not isinstance(w,(int,float)) or isinstance(w,bool) or not math.isfinite(w) or w<=0
                or (manifest['component']=='representative' and (not 0<r['inclusion_probability']<=1 or abs(w*r['inclusion_probability']-1)>1e-9))):
            raise ValueError('Missing or inconsistent inclusion weight')
        aa=ra.get(aid,{});bb=rb.get(aid,{});gg=adj.get(aid,{});jj=j.get(r['request_id'],{})
        rows.append({'id':aid,'provider':r['provider'],'benchmark':r['benchmark'],'parent':r['parent_id'],'w':w,
                     'a':aa.get('primary_label'),'b':bb.get('primary_label'),'gold':gg.get('primary_label'),
                     'judge':jj.get('primary_label'),'human_uncertain':aa.get('uncertain',False) or bb.get('uncertain',False) or gg.get('uncertain',False),
                     'judge_uncertain':jj.get('needs_review',False)})
    def section(rs):
        paired=[r for r in rs if r['a'] is not None and r['b'] is not None]
        resolved=[r for r in rs if r['gold'] is not None and r['judge'] is not None]
        certain=[r for r in rs if not r['human_uncertain'] and not r['judge_uncertain']]
        total=sum(r['w'] for r in rs);known=sum(r['w'] for r in resolved)
        correct=sum(r['w'] for r in resolved if r['gold']==r['judge'])
        human_known=sum(r['w'] for r in paired)
        human_matched=sum(r['w'] for r in paired if r['a']==r['b'])
        return {'sample_n':len(rs),'paired_human_n':len(paired),'judge_vs_adjudicated_n':len(resolved),
                'human_agreement_unweighted':agreement([r['a'] for r in paired],[r['b'] for r in paired]),
                'human_agreement_weighted':agreement([r['a'] for r in paired],[r['b'] for r in paired],[r['w'] for r in paired]),
                'judge_vs_adjudicated':confusion([r['gold'] for r in resolved],[r['judge'] for r in resolved],[r['w'] for r in resolved]),
                'sensitivity':{'exclude_any_uncertainty':scores(certain),'unresolved_n':len(rs)-len(resolved),
                               'human_uncertain_n':sum(r['human_uncertain'] for r in rs),
                               'judge_uncertain_n':sum(r['judge_uncertain'] for r in rs),
                               'full_sample_human_agreement_bounds_if_missing_disagree_or_agree':[human_matched/total,(human_matched+total-human_known)/total] if total else None,
                               'full_sample_accuracy_bounds_if_all_missing_wrong_or_right':[correct/total,(correct+total-known)/total] if total else None,
                               'weighted_resolved_fraction':known/total if total else None,
                               'note':'Bounds cover missing humans, missing adjudication and missing/unresolved judge labels; complete-case estimates assume missingness is ignorable.'},
                'cluster_uncertainty':cluster_intervals(rs,repetitions,seed)}
    durations={}
    for name,export in [('a',a),('b',b)]:
        elapsed=[r['elapsed_seconds'] for r in export['ratings']]
        durations[name]={'rated_n':len(elapsed),'median_seconds':quantile(elapsed,.5),'p90_seconds':quantile(elapsed,.9),
                         'total_active_seconds':sum(elapsed),'note':'Active visible-tab timer; interruptions/reading away from tab require manual coordinator accounting.'}
    return {'version':VERSION,'component':manifest['component'], 'status':'analysis of supplied annotations; interpret coverage before use',
            'inputs':{'manifest':manifest['checksum'],'a':a['checksum'],'b':b['checksum'],
                      'adjudication':adjudication['checksum'] if adjudication else None,'judges':judges['checksum'] if judges else None},
            'coverage':{'assigned':len(m),'annotator_a':len(ra),'annotator_b':len(rb),'adjudicated':len(adj),
                        'missing_a':sorted(set(m)-set(ra)),'missing_b':sorted(set(m)-set(rb)),
                        'missing_adjudication':sorted(set(m)-set(adj))},
            'timing':durations,'overall':section(rows),
            'by_provider':{p:section([r for r in rows if r['provider']==p]) for p in sorted({r['provider'] for r in rows})},
            'interpretation':['Representative weights target the remaining text-response frame conditional on practice exclusions; no benchmark-wide generalization.',
                              'Targeted results are diagnostic only; practice is unscored. Never pool components.',
                              'Cohen kappa is nominal, sensitive to prevalence and undefined for identical constant marginals.',
                              'High agreement does not establish taxonomy validity. No safety, harmful-compliance or native IHEval correctness inference follows from policy labels.',
                              'Cluster bootstrap is approximate with few parents; rare/absent classes and degenerate replicates must be reported.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','bundle','a','b','out']:p.add_argument('--'+key,required=True)
    for key in ['adjudication','judges','frame']:p.add_argument('--'+key)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=20261008)
    x=p.parse_args()
    report=analyze(read(x.manifest),read(x.bundle),read(x.a),read(x.b),read(x.adjudication) if x.adjudication else None,
                   read(x.judges) if x.judges else None,read(x.frame) if x.frame else None,x.bootstrap,x.seed)
    immutable(x.out,report);print('Saved private analysis:',x.out)

if __name__=='__main__':main()
