#!/usr/bin/env python3
"""Package v6 with unchanged v4/v5 dependencies and a content manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
ADDITIONS=("README_COMPLETION_V6.md","requirements-completion-v6.txt",
           "models/triprd_completion.py","models/triprd_completion_loss.py",
           "tools/run_triprd_completion.py","tools/completion_parallel.py",
           "tools/run_triprd_completion_pipeline.py","tools/run_triprd_completion.sh",
           "tools/evaluate_triprd_completion.py","tools/package_triprd_completion.py",
           "tests/test_completion_model.py","tests/test_completion_loss.py",
           "tests/test_completion_runner.py","tests/test_completion_protocol.py",
           "tests/test_completion_parallel.py","tests/test_completion_pipeline.py","tests/check_completion_dinov3.py",
           "literature/cod_transfer_redesign_2026-10-26.md")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'deliverables/triprd_completion_v6_2026-10-26.zip')
    a=p.parse_args(argv)
    with zipfile.ZipFile(ROOT/'deliverables/triprd_staged_v5_2026-10-25.zip') as old:
        previous=json.loads(old.read('MANIFEST_STAGED_V5.json'))
        payloads={name:old.read(name) for name in old.namelist()}
        for name,expected in previous['files'].items():
            if name.startswith('MANIFEST_'):
                continue
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected['sha256']:
                raise ValueError(f'Frozen v4/v5 dependency changed: {name}')
    payloads.update({name:(ROOT/name).read_bytes() for name in ADDITIONS})
    manifest={'method':'triprd_completion_v6','files':{
        name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in payloads.items()},
        'excluded':['datasets/images','generic weights','teacher/student checkpoints','results','dinov3 checkout'],
        'accuracy_evidence':'Functional validation only; no real-dataset COD gain or SOTA claim.',
        'training_route':['new multimodal completion teacher','RGB source student','full4040 COD','RGB-only test']}
    payloads['MANIFEST_COMPLETION_V6.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'w') as archive:
        for name,data in payloads.items():
            info=zipfile.ZipInfo(name,(2026,10,26,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if name.endswith('.sh') else 0o100644)<<16
            archive.writestr(info,data)
    with zipfile.ZipFile(a.out) as archive:
        if archive.testzip() is not None or any(archive.read(name)!=data for name,data in payloads.items()):
            raise RuntimeError('Archive integrity verification failed')
    print(json.dumps({'archive':str(a.out.resolve()),'entries':len(payloads),'bytes':a.out.stat().st_size,
                      'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
