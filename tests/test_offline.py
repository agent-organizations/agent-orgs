"""CPU-only tests; no network, model generation or GPU training."""
from pathlib import Path
import ast, asyncio, copy, hashlib, json, re, sys, tempfile, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'Experiment Codes/src'),str(ROOT/'Parallel Agent Setup/src')]
import workbench_protocol as p
import manager_experiment as m
import workbench_runtime as runtime
import participant_agent_interaction as interaction
from manipulation_checks import prepare_checks

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={(f,a):p.prepare_inputs(a,ROOT/'Experiment Codes'/f,save=False) for f in ['Mistral','Qwen'] for a in [1,2,3]}
        cls.sources={s['session_id']:s for s in p.read_json(ROOT/'DATA/participant_agent_interactions/frozen_sessions.json')}

    def test_unique_empty_instances_and_balanced_orientation(self):
        all_ids=[]
        for key,rows in self.cases.items():
            self.assertEqual(len(rows),200)
            all_ids += [c['agent_instance']['instance_id'] for c in rows]
            self.assertTrue(all(not c['agent_instance']['history'] for c in rows))
            self.assertEqual(sum(c['labels_to_actual']['P1']=='A' for c in rows),100)
            for a,b in zip(rows[::2],rows[1::2]):
                self.assertEqual(a['base_row_id'],b['base_row_id'])
                self.assertEqual(a['labels_to_actual']['P1'],b['labels_to_actual']['P2'])
                self.assertEqual(a['labels_to_actual']['P2'],b['labels_to_actual']['P1'])
                self.assertNotEqual(a['agent_instance']['seed'],b['agent_instance']['seed'])
        self.assertEqual(len(set(all_ids)),1200)

    def test_messages_and_final_ids_roundtrip(self):
        for family in ['Mistral','Qwen']:
            for c in self.cases[(family,2)]:
                old=self.sources[c['session_id']]
                self.assertEqual(len(old['messages']),len(c['messages']))
                for a,b in zip(c['messages'],old['messages']):
                    recovered=re.sub(r'\bP[12]\b',lambda x:'Agent '+c['labels_to_actual'][x.group()],a['text'])
                    self.assertEqual(recovered,b['text'])
                    self.assertEqual(c['labels_to_actual'][a['speaker']],b['actual_agent'])
                for label,mid in c['final_message_ids'].items():
                    final=next(x for x in reversed(c['messages']) if x['actual_agent']==c['labels_to_actual'][label])
                    self.assertEqual(final['id'],mid)

    def test_relative_beliefs_follow_identity(self):
        for rows in self.cases.values():
            for a,b in zip(rows[::2],rows[1::2]):
                for old,new in [('P1','P2'),('P2','P1')]:
                    self.assertEqual(a['belief_differences'][old].replace(old,'PERSON'),b['belief_differences'][new].replace(new,'PERSON'))

    def test_exact_base_prompts_and_materials(self):
        for (family,attempt),rows in self.cases.items():
            baseline=(ROOT/'Experiment Codes'/family/f'Attempt_{attempt:02d}/prompts/Prompt_Verbatim.txt').read_text()
            for c in rows:
                self.assertTrue(c['content'].startswith(baseline.replace('{manager}','manager')))
                self.assertEqual('<OTHER_PARTICIPANTS_CONVERSATION>' in c['content'],attempt in [2,3])
                self.assertNotIn('Correct consensus',c['content']);self.assertNotIn('False consensus',c['content'])
                self.assertNotRegex(c['content'],r'\bAgent\s+[ABCDE]\b|P[12]\s*=\s*[ABCDE]\b')
                self.assertEqual(c['allowed_choices'],(['P1','P2'] if attempt!=3 else ['P1','P2','None'] if c['source_outcome_offline_only']=='Unresolved' else ['P1','P2','Both','None']))

    def test_incomplete_prompt_is_rejected(self):
        for family in ['Mistral', 'Qwen']:
            for attempt in [1, 2, 3]:
                prompt = (ROOT/'Experiment Codes'/family/f'Attempt_{attempt:02d}/prompts/Prompt_Verbatim.txt').read_text()
                self.assertTrue(p.validate_base_prompt(prompt, attempt))
                with self.assertRaises(ValueError):
                    p.validate_base_prompt(prompt.split('OUTPUT FORMAT')[0], attempt)

    def test_all_contexts_fit_without_truncation(self):
        for (family,_),rows in self.cases.items():
            limit=16384 if family=='Mistral' else 8192
            for c in rows:self.assertLessEqual(c['input_tokens']+600,limit)

    def test_checks_are_unaided_one_shot(self):
        for family in ['Mistral','Qwen']:
            jobs=prepare_checks(ROOT/'Experiment Codes'/family)
            self.assertEqual(len(jobs),200)
            self.assertEqual(len({j['row_id'] for j in jobs}),200)
            for j in jobs:
                self.assertNotIn('BELIEF DIFFERENCES',j['serialized'])
                self.assertNotIn('<article>',j['serialized'])
                self.assertNotIn('OTHER_PARTICIPANTS',j['serialized'])

    def test_requests_preserve_every_permitted_category(self):
        for choice in ['P1','P2','BOTH','NONE']:
            text=f'YOUR OWN ANSWER: Test answer.\nVERDICT: {choice}\nRATIONALE: One reason. Another reason.'
            result=m.parse_manager_output(text,3,['P1','P2','Both','None'])
            self.assertTrue(result['format_valid']);self.assertTrue(result['permitted_choice']);self.assertEqual(result['raw_output'],text)
        bad='YOUR OWN ANSER: Test.\nVERDICT: BOTH\nRATIONALE: A reason. Another.'
        result=m.parse_manager_output(bad,3,['P1','P2','None'])
        self.assertFalse(result['permitted_choice']);self.assertEqual(result['raw_output'],bad)

    def test_independent_orientations_are_not_overwritten(self):
        cases=self.cases[('Mistral',1)];records=[]
        for c in cases:
            records.append({'call_id':'fixture_'+c['row_id'],'status':'complete','output_text':'YOUR OWN ANSWER: Fixture.\nENDORSEMENT: P1\nRATIONALE: First fixture reason. Second fixture reason.','finish_reason':'stop','input_tokens':c['input_tokens'],'output_tokens':20,'output_token_ids':[1]*20})
        rows=m.collect_results(cases,records,'fixture')
        self.assertTrue(all(x['same_actual_choice']==0 for x in m.position_comparison(rows)))
        with tempfile.TemporaryDirectory() as d:
            m.export_results(rows,{'elapsed_seconds':1},d);m.export_exact_io(cases,records,'fixture',d)
            from openpyxl import load_workbook
            wb=load_workbook(Path(d)/'Results.xlsx',read_only=True)
            self.assertEqual(wb['Long_Results'].max_row,201);self.assertEqual(wb['Wide_Results'].max_row,41);wb.close()
            self.assertEqual(len(json.loads((Path(d)/'All_Input_and_Output_Token_IDs.json').read_text())),200)

    def test_inference_requires_explicit_switch(self):
        with tempfile.TemporaryDirectory() as d:
            for call in [m.run_attempt(1,'test',root=Path(d)),interaction.advance_interactions('test',root=Path(d))]:
                with self.assertRaises(PermissionError):asyncio.run(call)
            self.assertFalse(list(Path(d).iterdir()))

    def test_input_binding_refuses_changed_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'provenance.json';p.write_json_once(path,{'x':1})
            with self.assertRaises(ValueError):p.write_json_once(path,{'x':2})
            self.assertEqual(json.loads(path.read_text()),{'x':1})

    def test_admission_respects_qwen_and_mistral_capacity(self):
        for family in ['Mistral','Qwen']:
            root=ROOT/'Experiment Codes'/family
            config=json.loads((root/'Attempt_03/config.json').read_text());config['stop_token_ids']=[2] if family=='Mistral' else [151643,151645]
            jobs=m.make_jobs(self.cases[(family,3)],'test',config)
            cap=runtime.choose_admission(jobs,root)
            self.assertLessEqual(cap['concurrency'],8);self.assertLessEqual(cap['total_reserved_at_cap'],64000)

class ParticipantInteractionTests(unittest.TestCase):
    def state(self,turn):
        s=interaction.new_interaction_sessions('test',ROOT/'Experiment Codes/Mistral')[0]
        s.update(status='awaiting_review',debate_turns=turn,latest={'1':{'output':'A'},'2':{'output':'B'}});return s
    def review(self,s,agree):
        return {'packet_sha256':interaction.packet_hash(interaction.review_packet(s)),'agreement':agree,'determinate':[True,True],
            'components':s['components'],'component_agreement':[agree]*len(s['components']),'rationale':'Fixture review','reviewer':'Offline test'}
    def test_immediate_stop_and_round_cap(self):
        for turn in [0,1,7,20]:
            s=self.state(turn);interaction.apply_review(s,self.review(s,True));self.assertEqual(s['status'],'complete')
        s=self.state(19);interaction.apply_review(s,self.review(s,False));self.assertEqual(s['status'],'ready')
        s=self.state(20);interaction.apply_review(s,self.review(s,False));self.assertEqual(s['stop_reason'],'ten_round_limit')
    def test_stale_review_fails(self):
        s=self.state(1);r=self.review(s,True);r['packet_sha256']='wrong'
        with self.assertRaises(ValueError):interaction.apply_review(s,r)
    def test_initial_answers_do_not_see_peer(self):
        s=self.state(0);s['history']=[{'agent':'B','stage':'initial','turn':0,'output':'PRIVATE_SENTINEL'}]
        _,text=interaction.interaction_input(s,1,'initial',ROOT/'Experiment Codes/Mistral')
        self.assertNotIn('PRIVATE_SENTINEL',text)

class PackageTests(unittest.TestCase):
    def test_registered_instances_and_shared_weights(self):
        for family in ['Mistral','Qwen']:
            instances=json.loads((ROOT/'Models'/family/'manager_instance_registry.json').read_text())
            self.assertEqual(len(instances),200);self.assertEqual(len({r['manager_instance_id'] for r in instances}),200)
            self.assertEqual(len({r['canonical_adapter'] for r in instances}),5)
            self.assertTrue(all(not r['conversation_history'] and r['weights_shared_immutable'] for r in instances))
            for row in instances:self.assertTrue((ROOT/row['adapter_config_path']).is_file())
    def test_data_hashes(self):
        for row in json.loads((ROOT/'DATA/article_manifest.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'])
    def test_notebooks_validate_and_have_no_saved_outputs(self):
        import nbformat
        for path in ROOT.rglob('*.ipynb'):
            n=nbformat.read(path,as_version=4);nbformat.validate(n)
            for c in n.cells:
                if c.cell_type=='code':
                    self.assertIsNone(c.execution_count);self.assertEqual(c.outputs,[])
                    compile(c.source,str(path),'exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    def test_python_compiles(self):
        for path in ROOT.rglob('*.py'):compile(path.read_text(),str(path),'exec')

if __name__=='__main__':
    with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden in offline tests')):unittest.main(verbosity=2)
