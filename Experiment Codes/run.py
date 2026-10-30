"""Portable CLI for preparing or executing the base-procedure manager experiments."""
from pathlib import Path
import argparse
import asyncio
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'Experiment Codes/src'),str(ROOT/'Parallel Agent Setup/src')]
from workbench_protocol import prepare_inputs, experiment_id
from manager_experiment import run_experiment

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['Mistral','Qwen'],required=True)
    p.add_argument('--experiment', choices=['1a','2a','3a','1b','2b','3b'], required=True)
    p.add_argument('--run-id',default='base_procedure_01')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();root=ROOT/'Experiment Codes'/a.model
    try:
        experiment_id(a.experiment, root)
    except ValueError as exc:
        p.error(str(exc))
    if a.execute:
        asyncio.run(run_experiment(a.experiment,a.run_id,allow_inference=True,root=root))
    else:
        cases=prepare_inputs(a.experiment,root,save=False)
        print('PREPARED',a.model,'EXPERIMENT',a.experiment,'CASES',len(cases),'MAX_INPUT_TOKENS',max(c['input_tokens'] for c in cases),'INFERENCE',False)
