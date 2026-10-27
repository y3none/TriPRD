#!/usr/bin/env python3
"""Manual real-architecture smoke check, with random local DINOv3 weights.

No download or dataset is used. This verifies shapes/gradients, not accuracy.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import torch
from models.triprd_completion import CompletionStudent, CompletionTeacher
from models.triprd_completion_loss import supervised_loss


def has_gradient(module):
    return any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dino-repo',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--size',type=int,default=384)
    p.add_argument('--base-channels',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    torch.set_num_threads(a.threads)
    result={'scope':'Official local DINOv3 random-weight architecture/gradient smoke check; no accuracy evidence',
            'pretrained':False,'device':a.device,'image_size':a.size,'base_channels':a.base_channels,
            'torch_version':str(torch.__version__),'roles':{}}
    for role,cls in (('student',CompletionStudent),('teacher',CompletionTeacher)):
        torch.manual_seed(31)
        started=time.perf_counter()
        model=cls(model_variant='dinov3_vitb16',base_channels=a.base_channels,pretrained=False,
                  dino_repo=a.dino_repo,train_backbone_blocks=4).to(a.device).train()
        rgb=torch.randn(1,3,a.size,a.size,device=a.device)
        mask=torch.zeros(1,1,a.size,a.size,device=a.device)
        mask[:,:,a.size//4:3*a.size//4,a.size//3:2*a.size//3]=1
        if role=='teacher':
            output=model(rgb,torch.rand(1,1,a.size,a.size,device=a.device),
                         torch.rand(1,1,a.size,a.size,device=a.device))
        else:
            output=model(rgb,task='cod')
        loss=supervised_loss(output,mask)['total']
        loss.backward()
        blocks=list(model.encoder.backbone.blocks)
        checks={'output_shape':list(output['logits'].shape)==[1,1,a.size,a.size],
                'finite_loss':bool(torch.isfinite(loss)),
                'finite_gradients':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                'encoder_tail_gradients':all(has_gradient(block) for block in blocks[-4:]),
                'early_encoder_frozen':all(not p.requires_grad and p.grad is None for block in blocks[:-4] for p in block.parameters()),
                'refiner_gradients':has_gradient(model.refiner),
                'real_rgb_detail_gradients':has_gradient(model.detail)}
        result['roles'][role]={'checks':checks,'loss':float(loss.detach()),
                              'stage_shapes':[list(x.shape) for x in output['stage_logits']],
                              'parameters':sum(p.numel() for p in model.parameters()),
                              'seconds':time.perf_counter()-started}
        del model,output,loss,rgb,mask
        gc.collect()
    result['status']='passed' if all(all(v['checks'].values()) for v in result['roles'].values()) else 'failed'
    serialized=json.dumps(result,ensure_ascii=False,indent=2)
    print(serialized)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(serialized+'\n',encoding='utf-8')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
