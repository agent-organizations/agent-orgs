"""Start one local multi-LoRA vLLM server, only with --execute."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import datetime as dt
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def command(family):
    work=ROOT/'Experiment Codes'/family
    cfg=json.loads((work/'config/model.json').read_text())
    registry=json.loads((work/'config/adapter_registry.json').read_text())
    model=(work/cfg['base_model_dir']).resolve()
    args=[sys.executable,'-m','vllm.entrypoints.openai.api_server',
        '--model',str(model),'--tokenizer',str(model),'--tokenizer-mode','slow',
        '--served-model-name',family.lower()+'-base','--dtype','bfloat16',
        '--quantization','bitsandbytes','--load-format','bitsandbytes',
        '--max-model-len',str(cfg['context_tokens']),'--gpu-memory-utilization','0.88',
        '--enable-lora','--lora-dtype','bfloat16','--max-lora-rank','16',
        '--max-loras',str(len(registry)),'--max-cpu-loras',str(len(registry)),
        '--max-num-seqs','32' if family=='Mistral' else '16',
        '--max-num-batched-tokens','4096','--no-enable-prefix-caching',
        '--generation-config','vllm','--host','127.0.0.1','--port','8000',
        '--disable-log-requests']
    if family=='Qwen':args+=['--trust-remote-code']
    args+=['--lora-modules']+[row['alias']+'='+str(ROOT/row['serving_path']) for row in registry]
    return args,registry

def start(family,execute=False):
    args,registry=command(family)
    if not execute:
        # Personal paths do not enter checked-in dry-run output.
        print(json.dumps([arg.replace(str(ROOT),'YOUR PATHWAY').replace(sys.executable,'python') for arg in args],indent=2))
        return None
    work=ROOT/'Experiment Codes'/family
    for row in json.loads((ROOT/'Models'/family/'base_manifest.json').read_text())['files']:
        path=ROOT/row['path']
        if not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Fetch or verify base file: '+row['path'])
    for row in registry:
        if sha(ROOT/row['serving_path']/'adapter_model.safetensors')!=row['weights_sha256']:raise ValueError('Adapter hash mismatch: '+row['alias'])
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8000))==0:raise RuntimeError('Port8000 already has a server; stop it deliberately before switching families.')
    runtime=work/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1'}
    with (runtime/'vllm.log').open('a') as log:
        process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    record={'pid':process.pid,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'command':args,
        'family':family,'endpoint':'http://127.0.0.1:8000','base_model_copies':1,
        'aliases':[r['alias'] for r in registry]}
    (runtime/'server_process.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Started server PID',process.pid,'; wait for /health before running requests.')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['Mistral','Qwen'],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    start(a.model,a.execute)
