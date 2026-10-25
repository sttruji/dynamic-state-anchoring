#!/usr/bin/env python3
"""Verify unmodified checkpoint inference before marker resize or anchor wrapping."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PIN='e0ef1887404adbf6c531d2cac7804694321f87bd'


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def smoke_checkpoint(backbone,revision,output,device='auto',tiny=False,max_new_tokens=16):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from dsa.models.backbone import BackboneAdapter
    from dsa.data.tokenize import TinyTokenizer
    from dsa.devices import mps_memory_snapshot, resolve_device, synchronize
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    if not tiny and (len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision)):
        raise ValueError('Resolve and pin an immutable checkpoint revision before inference')
    device=str(resolve_device(device))
    requested={'backbone':backbone,'revision':revision,'tiny':tiny,
               'max_new_tokens':max_new_tokens,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
               'device':device,'device_name':torch.cuda.get_device_name(device) if device=='cuda' else device,
               'device_helpers_sha256':hashlib.sha256((ROOT/'dsa/devices.py').read_bytes()).hexdigest(),
               'cuda':torch.version.cuda}
    output=Path(output)
    if output.exists():
        prior=json.loads(output.read_text())
        if prior.get('identity')==requested and prior.get('passed'):
            return {**prior,'reused_verified_smoke':True}
    torch.manual_seed(7)
    if tiny:
        tokenizer=TinyTokenizer()
        model=BackboneAdapter.tiny(len(tokenizer)).model
    else:
        tokenizer=AutoTokenizer.from_pretrained(backbone,revision=revision,use_fast=True)
        model=AutoModelForCausalLM.from_pretrained(backbone,revision=revision,attn_implementation='eager')
        if '<SA>' in tokenizer.get_vocab():raise ValueError('Original checkpoint has an unexpected marker collision')
    model=model.to(device).eval()
    original_vocab=model.get_input_embeddings().num_embeddings
    text='What is 2 plus 3?\n'
    inputs=torch.tensor([tokenizer.encode(text,add_special_tokens=False)],device=device)
    mask=torch.ones_like(inputs)
    synchronize(device)
    start=time.perf_counter()
    with torch.no_grad():
        logits=model(input_ids=inputs,attention_mask=mask,use_cache=False,return_dict=True).logits
        if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Unmodified checkpoint produced nonfinite logits')
        generated=model.generate(inputs,attention_mask=mask,max_new_tokens=max_new_tokens,do_sample=False,
                                 use_cache=False,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    synchronize(device)
    if model.get_input_embeddings().num_embeddings!=original_vocab:raise AssertionError('Smoke test changed model vocabulary')
    continuation=tokenizer.decode(generated[0,inputs.shape[1]:].tolist(),skip_special_tokens=True)
    report={'schema_version':'dsa_unmodified_checkpoint_smoke_v1','passed':True,'backbone':backbone,'revision':revision,
            'tiny':tiny,'identity':requested,'device':device,'input_question':text,'generated_continuation':continuation,
            'original_vocab_size':original_vocab,'vocabulary_changed':False,'anchor_wrapper_used':False,
            'finite_logits':True,'elapsed_seconds':time.perf_counter()-start,
            **mps_memory_snapshot(device),
            'accuracy_claim':'Inference plumbing only; this single question is not a benchmark result.',
            'model_weights_downloaded':not tiny,'hf_home':os.environ.get('HF_HOME')}
    atomic_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone',default='yingfanbot/gsm-cot-gpt2')
    parser.add_argument('--revision',default=PIN)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--max-new-tokens',type=int,default=16)
    parser.add_argument('--tiny',action='store_true')
    args=parser.parse_args()
    report=smoke_checkpoint(args.backbone,args.revision,args.output,args.device,args.tiny,args.max_new_tokens)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
