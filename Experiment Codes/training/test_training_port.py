"""Offline tests only: imports, native-boundary masks, schemas, source integrity and port AST."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from training_port import (_Port, _source, _prepare, validate_inputs, sha,
                           PHASE10_VALIDATION, FULL_VALIDATION)

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
ARCHIVE=HERE/'archival_sources'

class FakeTokenizer:
    eos_token_id=2
    pad_token_id=2
    im_start_id=151644
    im_end_id=151645
    def encode(self,text,**kwargs):return [ord(c)+100 for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i-100) for i in ids if i!=2)
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True):
        prefix=self.encode('[INST]'+messages[0]['content']+'[/INST]')
        return prefix if len(messages)==1 else prefix+self.encode(messages[1]['content'])+[2]

def extracted(file,names,context):
    tree=ast.parse((ARCHIVE/file).read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(tree,file,'exec'),context)
    return context

class PortTests(unittest.TestCase):
    def test_import_no_heavy_dependencies(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('transformers',sys.modules)
        self.assertNotIn('peft',sys.modules)
        spec=importlib.util.spec_from_file_location('training_cli_test',HERE/'train.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertNotIn('torch',sys.modules)
    def test_all_registered_inputs(self):
        for family,stages in {'Mistral':['v8_article','phase10_qa02_03','full_qa01_04_05'],'Qwen':['staged']}.items():
            for stage in stages:self.assertTrue(validate_inputs(PACKAGE,family,stage)['passed'])
    def test_exact_retained_losses(self):
        records=json.loads((PACKAGE/'Training logs/provenance_manifest.json').read_text())['records']
        count=0
        for r in records:
            if r['packaged_path'].endswith(('_losses.jsonl','/losses.jsonl','_steps.jsonl')):
                self.assertEqual(r['original_sha256'],r['packaged_sha256'])
                rows=[json.loads(x) for x in (PACKAGE/r['packaged_path']).read_text().splitlines()]
                self.assertEqual([x['step'] for x in rows],list(range(1,len(rows)+1)))
                self.assertTrue(all(isinstance(x['loss'],(int,float)) for x in rows))
                count+=len(rows)
        self.assertEqual(count,3520)
    def test_mistral_native_mask(self):
        g=extracted('phase10_train.py.txt',['training_examples'],{})
        row={'id':'test','prompt':'Which claim?','answer':'A stated claim.'}
        out=g['training_examples'](FakeTokenizer(),[row])[0]
        self.assertTrue(all(x==-100 for x in out['labels'][:out['prompt_tokens']]))
        self.assertEqual(out['labels'][-1],2)
        self.assertEqual(out['assistant_tokens'],len(row['answer'])+1)
    def test_qwen_native_mask(self):
        g=extracted('qwen_staged_train.py.txt',['chat_ids','qa_examples'],{'SYSTEM':'You are a helpful assistant.','END_TURN':151645})
        row={'id':'test','prompt':'Which claim?','answer':'A stated claim.'}
        out=g['qa_examples'](FakeTokenizer(),[row])[0]
        self.assertTrue(all(x==-100 for x in out['labels'][:out['prompt_tokens']]))
        self.assertEqual(out['labels'][-1],151645)
        self.assertEqual(out['targets'],len(row['answer'])+1)
    def test_ports_compile_without_execution(self):
        _source(ARCHIVE/'phase10_train.py.txt',_Port(remove=('decode',),replace={'evaluate_validation':PHASE10_VALIDATION}))
        _source(ARCHIVE/'mistral_full_train.py.txt',_Port(replace={'validation_responses':FULL_VALIDATION}))
        _source(ARCHIVE/'qwen_staged_train.py.txt',_Port())
        _source(ARCHIVE/'v8_train_agent.py.txt',_Port())
    def test_preparation_no_model_calls(self):
        # Fake empty base/parent files permit verifying directory and manifest shapes only.
        with tempfile.TemporaryDirectory() as d:
            fixture=Path(d)/'fixture';fixture.mkdir()
            (fixture/'Training logs').symlink_to(PACKAGE/'Training logs',target_is_directory=True)
            for family in ('Mistral','Qwen'):
                (fixture/'Models'/family/'base').mkdir(parents=True)
            for stage in ('v8_article','phase10_qa02_03'):
                for m in 'ABCDE':
                    dest=fixture/'Models/Mistral/adapters'/stage/m;dest.mkdir(parents=True)
                    (dest/'adapter_config.json').write_text('{}')
                    (dest/'adapter_model.safetensors').write_bytes(b'placeholder-only')
            for family,stage in [('Mistral','v8_article'),('Mistral','phase10_qa02_03'),('Mistral','full_qa01_04_05'),('Qwen','staged')]:
                out=Path(d)/(family+'_'+stage);out.mkdir()
                work=_prepare(fixture,family,stage,out)
                self.assertTrue((work/'config/registration.json').exists())
                if family=='Qwen':
                    self.assertEqual(len(json.loads((work/'inputs/train/A.json').read_text())),120)
            self.assertNotIn('torch',sys.modules)
    def test_no_machine_paths_or_run_summary_files(self):
        for base in (HERE,PACKAGE/'Training logs'):
            for p in base.rglob('*'):
                if not p.is_file():continue
                self.assertNotIn('run_'+'summary',p.name.lower())
                if p.suffix in ('.py','.txt','.json','.jsonl','.md','.log','.csv'):
                    s=p.read_text()
                    self.assertNotIn('/'+'Users/',s)
                    self.assertNotIn('/'+'home/jupyter',s)

if __name__=='__main__':unittest.main()
