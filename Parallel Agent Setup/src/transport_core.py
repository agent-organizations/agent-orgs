"""Isolated, resumable requests to one immutable multi-LoRA vLLM server.

This is infrastructure code. It contains no experiment-specific choice rules,
article materials, endorsement policies, or repair loops.
"""
from __future__ import annotations
import asyncio
import codecs
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
import httpx

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)

def runtime_digest(root):
    root=Path(root)
    server=json.loads((root/'config/server_process.json').read_text())
    return digest({'server_command':server['command'],
      'environment_lock':(root/'config/requirements.lock.txt').read_text(),
      'base_model_manifest':json.loads((root/'config/base_model_manifest.json').read_text()),
      'client_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

def request_digest(payload, alias, runtime):
    return digest({'payload':payload,'adapter':alias,'runtime':runtime})

async def journaled_lines(response, path):
    decoder=codecs.getincrementaldecoder('utf-8')()
    pending=''
    with Path(path).open('ab') as journal:
        async for chunk in response.aiter_bytes():
            journal.write(chunk);journal.flush()
            pending+=decoder.decode(chunk)
            while '\n' in pending:
                line,pending=pending.split('\n',1)
                yield line.removesuffix('\r')
        pending+=decoder.decode(b'',final=True)
        if pending:yield pending.removesuffix('\r')

def validate_job(job, aliases, context_limit=16384):
    if job['model'] not in aliases:raise ValueError('Unregistered adapter: '+job['model'])
    ids=job['input_ids']
    if not ids or any(type(x) is not int or x<0 for x in ids):raise ValueError('Explicit token IDs required')
    if len(ids)+job['max_tokens']>context_limit:raise ValueError('Context budget exceeded; truncation is forbidden')
    if not 0<=job['temperature']<=2:raise ValueError('Invalid temperature')
    if not 0<=job['seed']<2**32:raise ValueError('Explicit 32-bit request seed required')
    if not job['call_id'].replace('_','').replace('-','').isalnum():raise ValueError('Unsafe call ID')
    return {
      'model':job['model'],'prompt':ids,'temperature':job['temperature'],
      'max_tokens':job['max_tokens'],'seed':job['seed'],'n':1,
      'top_p':1.0,'top_k':-1,'min_p':0.0,'repetition_penalty':1.0,
      'presence_penalty':0.0,'frequency_penalty':0.0,
      'stop_token_ids':job.get('stop_token_ids',[]),'stop':job.get('stop',[]),'include_stop_str_in_output':True,
      'skip_special_tokens':False,'logprobs':5,'return_tokens_as_token_ids':True,'return_token_ids':True,
      'min_tokens':job.get('min_tokens',0),
      'stream':True,'stream_options':{'include_usage':True}}

async def execute_one(client, semaphore, job, root, aliases, endpoint):
    payload=validate_job(job, aliases)
    runtime=runtime_digest(root)
    signature=request_digest(payload,aliases[job['model']],runtime)
    path=Path(root)/'logs/calls'/(job['call_id']+'.json')
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['request_sha256']!=signature:raise ValueError('A saved call ID has different inputs')
        if saved['status']=='complete':return {**saved,'resumed_from_disk':True}
        raise RuntimeError('Previous call failed; explicit new run ID required: '+job['call_id'])
    path.parent.mkdir(parents=True,exist_ok=True)
    claim=path.with_suffix('.claim')
    fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    queued=time.perf_counter()
    record={'call_id':job['call_id'],'kind':job.get('kind', 'research_call'),
      'status':'queued','queued_utc':stamp(),'request_sha256':signature,
      'request':payload,'adapter':aliases[job['model']],'runtime_sha256':runtime,
      'serialized_input':job['serialized'],'input_tokens':len(job['input_ids']),
      'output_text':'','output_token_ids':[],'logprob_token_ids':[],
      'server_prompt_token_ids':None,'events':[],'usage':None,'received_done':False,
      'finish_reason':None,'stop_reason':None}
    atomic_json(path,record)
    async with semaphore:
        start=time.perf_counter();record.update(status='running',started_utc=stamp(),client_queue_seconds=start-queued)
        atomic_json(path,record)
        print('REQUEST_START', job['call_id'], 'INPUT_TOKENS', len(job['input_ids']), flush=True)
        first=None
        try:
            async with client.stream('POST',endpoint+'/v1/completions',json=payload) as response:
                if response.status_code!=200:
                    await response.aread()
                    raise RuntimeError(f'HTTP {response.status_code}: {response.text[:2000]}')
                async for line in journaled_lines(response,path.with_suffix('.sse')):
                    if not line.startswith('data:'):continue
                    raw=line[5:].lstrip()
                    if raw=='[DONE]':
                        record['received_done']=True;break
                    event=json.loads(raw);record['events'].append(event)
                    if event.get('error'):raise RuntimeError(str(event['error']))
                    if event.get('usage') is not None:record['usage']=event['usage']
                    for choice in event.get('choices',[]):
                        if choice.get('prompt_token_ids') is not None:
                            if record['server_prompt_token_ids'] is not None and record['server_prompt_token_ids']!=choice['prompt_token_ids']:
                                raise RuntimeError('Server prompt token IDs changed within one request')
                            record['server_prompt_token_ids']=choice['prompt_token_ids']
                        record['output_token_ids'].extend(choice.get('token_ids') or [])
                        text=choice.get('text','')
                        if text and first is None:first=time.perf_counter()
                        record['output_text']+=text
                        tokens=(choice.get('logprobs') or {}).get('tokens') or []
                        for token in tokens:
                            if token.startswith('token_id:'):
                                record['logprob_token_ids'].append(int(token.split(':',1)[1]))
                        if choice.get('finish_reason') is not None:record['finish_reason']=choice['finish_reason']
                        if choice.get('stop_reason') is not None:record['stop_reason']=choice['stop_reason']
            if record['usage'] is None:raise RuntimeError('Missing server token usage')
            if record['usage']['prompt_tokens']!=record['input_tokens']:raise RuntimeError('Server input token count mismatch')
            if record['server_prompt_token_ids']!=job['input_ids']:raise RuntimeError('Server prompt token IDs mismatch')
            if not record['received_done'] or record['finish_reason'] not in ['stop','length']:raise RuntimeError('Incomplete completion stream')
            record['output_tokens']=record['usage']['completion_tokens']
            record['output_ids_match_usage']=len(record['output_token_ids'])==record['output_tokens']
            if not record['output_ids_match_usage']:raise RuntimeError('Output token IDs/usage mismatch')
            if record['logprob_token_ids']!=record['output_token_ids']:raise RuntimeError('Logprob and output token IDs mismatch')
            record['status']='complete'
        except asyncio.CancelledError:
            record.update(status='interrupted',error='Request task cancelled; explicit recovery is required')
            raise
        except Exception as exc:
            record.update(status='failed',error=repr(exc))
        finally:
            end=time.perf_counter()
            record.update(finished_utc=stamp(),elapsed_seconds=end-start,
              total_seconds=end-queued,time_to_first_text_seconds=None if first is None else first-start)
            atomic_json(path,record)
            claim.unlink(missing_ok=True)
            print('REQUEST',job['call_id'],record['status'],'INPUT',record['input_tokens'],
                  'OUTPUT',record.get('output_tokens'),'QUEUE_SECONDS',round(record['client_queue_seconds'],3),
                  'GENERATION_SECONDS',round(record['elapsed_seconds'],3),
                  'TOTAL_SECONDS',round(record['total_seconds'],3),flush=True)
    return record

async def monitor_memory(path, stop):
    with Path(path).open('a') as file:
        while not stop.is_set():
            process=await asyncio.create_subprocess_exec('nvidia-smi',
              '--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits',
              stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            out,err=await process.communicate()
            sample={'utc':stamp(),'gpu':out.decode().strip(),'error':err.decode().strip()}
            sample['meminfo']=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:')))
            file.write(json.dumps(sample)+'\n');file.flush()
            try:await asyncio.wait_for(stop.wait(),timeout=1.0)
            except asyncio.TimeoutError:pass

async def run_batch(jobs, concurrency, root, run_name, aliases, endpoint='http://127.0.0.1:8000'):
    invocation_start=time.perf_counter()
    root=Path(root)
    if len({j['call_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate call IDs')
    for job in jobs:validate_job(job,aliases)
    runtime=runtime_digest(root)
    for job in jobs:
        path=root/'logs/calls'/(job['call_id']+'.json')
        if path.with_suffix('.claim').exists():raise RuntimeError('Call already claimed; inspect before recovery')
        if path.exists():
            value=json.loads(path.read_text())
            expected=request_digest(validate_job(job,aliases),aliases[job['model']],runtime)
            if value['request_sha256']!=expected:raise ValueError('Saved input/runtime signature changed')
            if value['status']!='complete':raise RuntimeError('Previous call is incomplete; explicit new diagnostic run ID required')
    summary_path=root/'qa'/f'{run_name}.json'
    if summary_path.exists():
        previous=json.loads(summary_path.read_text())
        if previous.get('failed')==0 and previous.get('call_ids')==[j['call_id'] for j in jobs] and previous.get('concurrency')==concurrency:
            saved=[]
            for job in jobs:
                value=json.loads((root/'logs/calls'/(job['call_id']+'.json')).read_text())
                expected=request_digest(validate_job(job,aliases),aliases[job['model']],runtime)
                if value['request_sha256']!=expected:raise ValueError('Existing batch inputs changed')
                saved.append(value)
            if all(value['status']=='complete' for value in saved):
                print('BATCH_RESUMED_WITHOUT_GENERATION',run_name,flush=True)
                return {**previous,'resumed_without_generation':True,
                    'historical_batch_elapsed_seconds':previous['elapsed_seconds'],
                    'current_invocation_seconds':time.perf_counter()-invocation_start},[
                    {**value,'resumed_from_disk':True} for value in saved]
    stop=asyncio.Event();memory_path=root/'logs'/f'{run_name}_memory.jsonl'
    memory_path.parent.mkdir(parents=True,exist_ok=True)
    monitor=asyncio.create_task(monitor_memory(memory_path,stop))
    start=time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0,connect=30.0),
          limits=httpx.Limits(max_connections=concurrency,max_keepalive_connections=concurrency),trust_env=False) as client:
            before=await client.get(endpoint+'/metrics')
            (root/'logs'/f'{run_name}_metrics_before.txt').write_text(before.text)
            semaphore=asyncio.Semaphore(concurrency)
            tasks=[asyncio.create_task(execute_one(client,semaphore,j,root,aliases,endpoint)) for j in jobs]
            try:
                results=await asyncio.gather(*tasks,return_exceptions=True)
            except BaseException:
                # Never release the endpoint lock while sibling requests remain active.
                for task in tasks:
                    if not task.done():task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                raise
            errors=[{'call_id':job['call_id'],'error':repr(result)}
                    for job,result in zip(jobs,results) if isinstance(result,BaseException)]
            if errors:
                atomic_json(root/'qa'/f'{run_name}_task_errors.json',errors)
                raise RuntimeError('Request task failures drained safely; inspect the saved task errors')
            after=await client.get(endpoint+'/metrics')
            (root/'logs'/f'{run_name}_metrics_after.txt').write_text(after.text)
    finally:
        stop.set();await monitor
    seconds=time.perf_counter()-start
    complete=[r for r in results if r['status']=='complete']
    used=[r for r in complete if not r.get('resumed_from_disk')]
    summary={'run_name':run_name,'utc':stamp(),'concurrency':concurrency,'requests':len(jobs),
      'completed':len(complete),'failed':len(results)-len(complete),
      'elapsed_seconds':seconds,'input_tokens':sum(r['input_tokens'] for r in complete),
      'output_tokens':sum(r.get('output_tokens',0) for r in complete),
      'fresh_output_tokens_per_second':sum(r.get('output_tokens',0) for r in used)/seconds,
      'all_input_counts_match':all(r.get('usage',{}).get('prompt_tokens')==r['input_tokens'] for r in complete),
      'all_output_ids_match_usage':all(r.get('output_ids_match_usage') for r in complete),
      'call_ids':[j['call_id'] for j in jobs], 'resumed_without_generation':False,
      'current_invocation_seconds':time.perf_counter()-invocation_start}
    atomic_json(root/'qa'/f'{run_name}.json',summary)
    print('BATCH_SUMMARY',json.dumps(summary),flush=True)
    return summary,results
