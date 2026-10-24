"""Standard-library orchestration around retained training-only source algorithms.

Import does not load Torch, change environment variables, create files, or start GPU work.
The historical source is compiled only after explicit execute(). See README for port deltas.
"""
from pathlib import Path
import ast
import datetime
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import types

MANAGERS = tuple('ABCDE')
STAGES = {'Mistral': ('v8_article','phase10_qa02_03','full_qa01_04_05'), 'Qwen': ('staged',)}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n'); f.flush()

def validate_inputs(package, family, stage):
    package = Path(package).resolve()
    if stage not in STAGES[family]:
        raise ValueError(f'Unsupported {family} stage: {stage}')
    logs = package/'Training logs'
    records = load(logs/'provenance_manifest.json')['records']
    for r in records:
        p = package/r['packaged_path']
        if sha(p) != r['packaged_sha256']:
            raise ValueError(f'Packaged training input changed: {r["packaged_path"]}')
    counts = {}
    if family == 'Mistral' and stage == 'v8_article':
        for m in MANAGERS:
            for a in ('01','02','03','04','05'):
                assert (logs/'source_articles'/m/(a+'.txt')).is_file()
            rows = [json.loads(x) for x in (logs/family/stage/m/'losses.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows] == list(range(1,221))
            counts[m] = {'article_files':5,'retained_steps':len(rows)}
    else:
        stages = ('qa02_03','full_qa01_04_05') if family=='Qwen' else (stage,)
        for st in stages:
            expected = 48 if st in ('qa02_03','phase10_qa02_03') else 72
            nval = 6 if expected==48 else 15
            for m in MANAGERS:
                train = load(logs/family/st/'inputs/train'/f'{m}.json')
                val = load(logs/family/st/'inputs/validation'/f'{m}.json')
                assert len(train)==expected and len(val)==nval
                assert len({r['id'] for r in train})==expected
                assert not ({r['prompt'] for r in train} & {r['prompt'] for r in val})
                for row in train+val:
                    assert row['manager']==m and row['prompt'] and row['answer']
                    assert row['source_sha256']==sha(logs/'source_articles'/m/(row['article_id']+'.txt'))
                counts[f'{st}/{m}']={'train_examples':len(train),'validation_examples':len(val)}
    return {'passed':True,'family':family,'stage':stage,'verified_source_records':len(records),'counts':counts,
            'gpu_execution':False,'base_weights_checked':False,'note':'Base/adapters and CUDA checked only on explicit execution; no assertion of reproduced historical weights.'}

def _copy(source, dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)

def _link(source, dest):
    source=Path(source).resolve();dest=Path(dest)
    if not source.exists():
        raise FileNotFoundError(f'Missing retained model input: {source}')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source,target_is_directory=source.is_dir())

def _prepare(package,family,stage,output):
    logs=package/'Training logs';work=output/'run';work.mkdir(parents=True)
    for d in ['config','qa','logs/training','logs/validation','logs/calls','logs/batches','data','source','adapters']:
        (work/d).mkdir(parents=True,exist_ok=True)
    model=package/'Models'/family/'base'
    _link(model,work/'model_reference/models'/family.lower())
    if family=='Mistral' and stage=='v8_article':
        shutil.copytree(logs/'source_articles',work/'agent_inputs')
        _copy(logs/'Mistral/v8_article/protocol_original.json',work/'config/protocol.json')
    elif family=='Mistral':
        source=logs/family/stage
        shutil.copytree(source/'inputs',work/'inputs')
        if stage=='phase10_qa02_03':
            _copy(source/'config/protocol.json',work/'config/protocol.json')
            _copy(source/'config/mistral_model_manifest.json',work/'config/mistral_model_manifest.json')
            adapter_manifest=[]
            for m in MANAGERS:
                dest=work/'parents/adapters'/('Manager_Agent_Pair_'+m)
                _link(package/'Models/Mistral/adapters/v8_article'/m,dest)
                for name in ('adapter_model.safetensors','adapter_config.json'):
                    adapter_manifest.append({'path':str((dest/name).relative_to(work)),'sha256':sha(dest/name)})
            save(work/'config/adapter_manifest.json',adapter_manifest)
        else:
            _copy(source/'config/training_protocol.json',work/'config/training_protocol.json')
            _copy(logs/'Mistral/v8_article/base_file_manifest.json',work/'source/phase10/config/mistral_model_manifest.json')
            for m in MANAGERS:
                _link(package/'Models/Mistral/adapters/phase10_qa02_03'/m,work/'adapters/copied_phase10'/m)
    else:
        _copy(logs/'Qwen/config/training_protocol.json',work/'config/training_protocol.json')
        _copy(logs/'Qwen/source/v8pre_qwen_model_manifest.json',work/'source/v8pre_qwen_model_manifest.json')
        shutil.copytree(logs/'source_articles',work/'inputs/articles')
        for m in MANAGERS:
            old=load(logs/'Qwen/qa02_03/inputs/train'/f'{m}.json')
            new=load(logs/'Qwen/full_qa01_04_05/inputs/train'/f'{m}.json')
            save(work/'inputs/train'/f'{m}.json',old+new)
            _copy(logs/'Qwen/full_qa01_04_05/inputs/validation'/f'{m}.json',work/'inputs/validation'/f'{m}.json')
    # Register only the fresh run's inputs, excluding symlinked multi-GB model banks.
    checks={str(p.relative_to(work)):sha(p) for d in ('inputs','agent_inputs','config','source') for p in (work/d).rglob('*') if p.is_file()}
    if family=='Mistral' and stage=='phase10_qa02_03':
        save(work/'config/registration.json',{'files':[{'path':p,'sha256':h} for p,h in checks.items()]})
    else:
        save(work/'config/registration.json',{'files':checks})
    save(output/'port_receipt.json',{'status':'prepared','family':family,'stage':stage,'utc':stamp(),
         'retained_package_inputs':'Source hashes are in Training logs/provenance_manifest.json.',
         'generated_validation_responses_replayed':False,'checkpoint_selection':'Original answer-token NLL algorithm; no experimental decisions.'})
    return work

class _Port(ast.NodeTransformer):
    def __init__(self, remove=(), replace=None, drop_root=False):
        self.remove=set(remove);self.replace=replace or {};self.drop_root=drop_root
    def visit_ImportFrom(self,node):
        return None if node.module in ('common','load_models','cuda_runtime') else node
    def visit_Import(self,node):
        node.names=[a for a in node.names if a.name!='parallel_decode']
        return node if node.names else None
    def visit_FunctionDef(self,node):
        if node.name in self.remove:return None
        if node.name in self.replace:return ast.parse(self.replace[node.name]).body[0]
        return node
    def visit_Assign(self,node):
        if self.drop_root and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in node.targets):return None
        if any(isinstance(t,ast.Tuple) and [getattr(x,'id',None) for x in t.elts]==['family','agent'] for t in node.targets):return None
        return node
    def visit_If(self,node):
        if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__':return None
        return node

def _source(path, transformer):
    tree=transformer.visit(ast.parse(Path(path).read_text()))
    ast.fix_missing_locations(tree)
    return compile(tree,str(path),'exec')

PHASE10_VALIDATION='''def evaluate_validation(model,tok,manager,epoch,rows,key,tokenized):
 set_eval(model,'QA_'+manager)
 ids,mask,labels=tensors(tok,tokenized)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): loss=float(model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss)
 del ids,mask,labels
 record={'manager':manager,'epoch':epoch,'total':len(rows),'assistant_validation_loss':loss,'calls':[],'utc':stamp(),'generated_diagnostic_omitted_in_port':True}
 save(ROOT/'logs/training'/f'{manager}_epoch{epoch:02d}_selection.json',record)
 return record
'''
FULL_VALIDATION='''def validation_responses(model,tokenizer,manager,epoch,rows):
 eval_mode(model,'FULL_'+manager)
 return []
'''

def _globals(work,family):
    def verify():
        rows=load(work/'config/registration.json')['files']
        rows=rows.items() if isinstance(rows,dict) else ((r['path'],r['sha256']) for r in rows)
        for p,h in rows:assert sha(work/p)==h,p
    return {'__name__':'training_port_runtime','__file__':str(Path(__file__)), 'ROOT':work,
        'BASE_MODEL':work/'model_reference/models'/family.lower(), 'MANAGERS':MANAGERS,
        'BASE':work,'PRE':work/'model_reference','PARENT':work/'parents',
        'load':load,'read':load,'save':save,'write':save,'sha':sha,'stamp':stamp,
        'seed':lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:15],16),
        'append_jsonl':append_jsonl,'verify_registration':verify}

def _v8_worker(archive,work,manager):
    g=_globals(work,'Mistral');g.update(family='mistral',agent=manager)
    source=(archive/'v8_cuda_training.py.txt').read_text()
    # Exact original model/tokenizer calls; only the base root is relocated.
    source=source.replace('Path("YOUR PATHWAY/Outputs/V8-Pre_2026-10-01/models")/family','ROOT/"model_reference/models"/family')
    tree=_Port(drop_root=True).visit(ast.parse(source));ast.fix_missing_locations(tree)
    exec(compile(tree,'archival_sources/v8_cuda_training.py.txt','exec'),g)
    exec(_source(archive/'v8_train_agent.py.txt',_Port()),g)

def execute(package,family,stage,output):
    package=Path(package).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite an existing run directory')
    if output==package or package in output.parents:
        raise ValueError('Training output must be outside the retained submission package')
    validate_inputs(package,family,stage)
    output.mkdir(parents=True)
    work=_prepare(package,family,stage,output)
    archive=Path(__file__).resolve().parent/'archival_sources'
    if family=='Mistral' and stage=='v8_article':
        for row in load(package/'Training logs/Mistral/v8_article/base_file_manifest.json'):
            parts=Path(row['path']).parts
            assert parts[:2]==('models','mistral'), row['path']
            assert sha(package/'Models/Mistral/base'/Path(*parts[2:]))==row['sha256'], row['path']
    # Snapshot every retained parent/base byte before training; prohibit input mutation.
    protected=[]
    for base in [package/'Models'/family/'base'] + ([package/'Models/Mistral/adapters'/('v8_article' if stage=='phase10_qa02_03' else 'phase10_qa02_03')] if family=='Mistral' and stage!='v8_article' else []):
        for p in base.rglob('*'):
            if p.is_file():protected.append((p,sha(p)))
    if family=='Mistral' and stage=='v8_article':
        for manager in MANAGERS:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--v8-worker',str(archive),str(work),manager],check=True)
    else:
        g=_globals(work,family)
        if family=='Qwen':
            source=archive/'qwen_staged_train.py.txt';transform=_Port()
        elif stage=='phase10_qa02_03':
            source=archive/'phase10_train.py.txt'
            transform=_Port(remove=('decode',),replace={'evaluate_validation':PHASE10_VALIDATION})
        else:
            source=archive/'mistral_full_train.py.txt'
            transform=_Port(replace={'validation_responses':FULL_VALIDATION})
        exec(_source(source,transform),g)
        g['train_all']()
    for p,h in protected:assert sha(p)==h, f'Retained model input changed: {p.name}'
    for manager in MANAGERS:
        if family=='Qwen':
            locations=[('article',work/'adapters/article'/manager/manager),('qa02_03',work/'adapters/qa02_03'/manager/manager),('full_qa01_04_05',work/'adapters/trained'/manager/manager)]
        elif stage=='v8_article':locations=[(stage,work/'adapters/mistral'/manager)]
        elif stage=='phase10_qa02_03':locations=[(stage,work/'adapters'/manager/('QA_'+manager))]
        else:locations=[(stage,work/'adapters/trained'/manager/('FULL_'+manager))]
        for target,source in locations:
            for name in ('adapter_config.json','adapter_model.safetensors'):
                _copy(source/name,output/'Models'/family/'adapters'/target/manager/name)
    save(output/'completion_receipt.json',{'status':'complete','utc':stamp(),'family':family,'stage':stage,
        'protected_input_files_unchanged':len(protected),'optimizer_states_saved':False,
        'warning':'New training results, not the retained historical records. Canonical serving export is a separate operation.'})

if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--v8-worker':
        _v8_worker(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])
    else:
        raise SystemExit('Use train.py; training requires explicit --execute.')
