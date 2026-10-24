"""Portable CLI for preparing or executing the base-procedure manager attempts."""
from pathlib import Path
import argparse
import asyncio
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'Experiment Codes/src'),str(ROOT/'Parallel Setup/src')]
from workbench_protocol import prepare_inputs
from manager_experiment import run_attempt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['Mistral','Qwen'],required=True)
    p.add_argument('--attempt',type=int,choices=[1,2,3],required=True)
    p.add_argument('--run-id',default='base_procedure_01')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();root=ROOT/'Experiment Codes'/a.model
    if a.execute:
        asyncio.run(run_attempt(a.attempt,a.run_id,allow_inference=True,root=root))
    else:
        cases=prepare_inputs(a.attempt,root,save=False)
        print('PREPARED',a.model,'ATTEMPT',a.attempt,'CASES',len(cases),'MAX_INPUT_TOKENS',max(c['input_tokens'] for c in cases),'INFERENCE',False)
