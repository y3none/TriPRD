"""Reproduce completion-v6 report-level comparisons without model weights."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tarfile

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'results/TriPRD_completion_v6/completion_v6_reports.tar.gz'
OUT=ROOT/'experiments/completion_v6_result_analysis_2026-10-27.json'
METRICS=('S_alpha','F_beta_w','E_phi_mean','E_phi_max','E_phi_adaptive','MAE')
ORDER=('COD10K','CAMO','NC4K','CHAMELEON')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def main():
    with tarfile.open(INPUT,'r:gz') as archive:
        files={member.name.removeprefix('./'):archive.extractfile(member).read()
               for member in archive.getmembers() if member.isfile()}
    reports={stage:json.loads(files[stage+'/comparison.json']) for stage in ('teacher','source','cod')}
    checks={}
    for stage,r in reports.items():
        checks[stage]={
          'contract_hash':r['contract_sha256']==digest(r['protocol']),
          'history_copies':r['history']==json.loads(files[stage+'/history.json']),
          'epoch_sequence':[x['epoch'] for x in r['history']]==list(range(1,r['completed_epochs']+1)),
          'code_hashes':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in r['protocol']['source_sha256'].items())}
        assert all(checks[stage].values()),(stage,checks[stage])
    cod=reports['cod']; protocol=cod['protocol']
    assert Counter(k.split('/',1)[0] for k in protocol['cod_train'])=={'COD10K':3040,'CAMO':1000}
    assert len(set(protocol['cod_train']))==4040 and not protocol['cod_val']
    assert cod['best_epoch']==cod['completed_epochs']==100 and cod['test_status']=='complete'
    assert cod['test']==json.loads(files['cod/test_metrics.json'])
    expected={'COD10K':2026,'CAMO':250,'NC4K':4121,'CHAMELEON':76}
    for name,n in expected.items():
        assert cod['test'][name]['count']==cod['test'][name]['official']['num_samples']==n
    benchmark={d:{k:cod['test'][d]['official'][k] for k in METRICS} for d in ORDER}
    text=(ROOT/'previous_performance.tex').read_text(encoding='utf-8')
    e2row=text.split('E2Net\n')[-1].split('\\bottomrule')[0]
    numbers=[float(x) for x in re.findall(r'(?<!\d)\.\d{3}(?!\d)',e2row)]
    assert len(numbers)==16,len(numbers)
    e2={d:dict(zip(('S_alpha','alphaE_unverified','F_beta_w','MAE'),numbers[i*4:i*4+4]))
        for i,d in enumerate(('NC4K','COD10K','CAMO','CHAMELEON'))}
    deltas={d:{k:benchmark[d][k]-e2[d][k] for k in ('S_alpha','F_beta_w','MAE')} for d in ORDER}
    stages={}
    for stage,r in reports.items():
        h=r['history']; first,last=h[0]['train'],h[-1]['train']
        stages[stage]={
          'epochs':len(h),'selected_epoch':r['best_epoch'],'selection_metric':r['selection_metric'],
          'steps':sum(x['train']['steps'] for x in h),
          'optimizer_steps':sum(x['train']['optimizer_steps'] for x in h),
          'amp_skips':sum(x['train']['amp_skipped_steps'] for x in h),
          'elapsed_minutes':(datetime.fromisoformat(r['finished_at'])-datetime.fromisoformat(r['started_at'])).total_seconds()/60,
          'first_train':first,'last_train':last,
          'selected_epoch_train':h[r['best_epoch']-1]['train']}
        if stage!='cod':
            stages[stage].update(first_validation=h[0]['validation']['loss'],best_validation=-r['best_score'],last_validation=h[-1]['validation']['loss'])
    source=reports['source']; hs=source['history']
    kd=[]
    for row in hs:
        t=row['train']; weighted=t['mask_kd']+.2*t['boundary_kd']; gt=t['total']-weighted
        kd.append({'epoch':row['epoch'],'weighted_kd':weighted,'gt_loss':gt,'kd_fraction_total':weighted/t['total'],'kd_over_gt':weighted/gt})
    hc=cod['history']; get=lambda e,k:hc[e-1]['train'][k]
    diagnostics={
      'teacher_best_to_last_val_relative_increase':stages['teacher']['last_validation']/stages['teacher']['best_validation']-1,
      'student_best_to_last_val_relative_increase':stages['source']['last_validation']/stages['source']['best_validation']-1,
      'source_kd':kd,
      'cod_epochs_1_to_5':[{k:row[k] for k in ('epoch','phase','train')} for row in hc[:5]],
      'warmup_coarse_loss_relative_reduction':1-get(3,'supervised_coarse')/get(1,'supervised_coarse'),
      'warmup_final_mask_relative_reduction':1-get(3,'supervised_mask')/get(1,'supervised_mask'),
      'first_joint_final_mask_relative_reduction':1-get(4,'supervised_mask')/get(3,'supervised_mask'),
      'last_final_vs_coarse_loss_relative_reduction':1-get(100,'supervised_mask')/get(100,'supervised_coarse')}
    result={'date':'2026-10-27','input':str(INPUT.relative_to(ROOT)),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            'input_files':len(files),'checks':checks,'benchmark':benchmark,'stages':stages,'diagnostics':diagnostics,
            'e2net_reference':{'source':'previous_performance.tex','kind':'user_record_not_matched_rerun','rows':e2,'v6_minus_reference':deltas,'E_not_compared':'alphaE variant of user E2Net row not independently verified'},
            'gpus':json.loads(files['completion_pipeline.json'])['gpus'],
            'checkpoint_sha256_reported':cod['checkpoint_sha256'],
            'limitations':['No checkpoint or prediction tensors available: report-level audit only',
                           'v4/v5 internal validation cannot be subtracted from official v6 test scores',
                           'No matched v6 RGB-only/no-transfer official result: multimodal net gain is unidentified',
                           'Full COD run has no validation curve: no claim about optimal epoch or test overfitting',
                           'FNR is image-and-threshold-average, not fraction of missed objects; boundary metrics are null']}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    destination=(ROOT/'results/TriPRD_completion_v6/reports').resolve()
    for name,payload in files.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe report path')
        target=destination.joinpath(*rel.parts).resolve()
        if not target.is_relative_to(destination): raise ValueError('Report path escapes destination')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=payload: raise ValueError('Existing extracted report differs: '+name)
        target.write_bytes(payload)
    print(json.dumps({'checks':checks,'e2net_deltas':deltas,'diagnostics':{k:v for k,v in diagnostics.items() if k not in {'source_kd','cod_epochs_1_to_5'}},'output':str(OUT)},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
