"""Restore pinned base models and private adapter archives; never run inference."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def checked(row):
    path=(ROOT/row['path']).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Unsafe manifest path')
    return path

def base(family):
    manifest=json.loads((ROOT/'Models'/family/'base_manifest.json').read_text())
    for row in manifest['files']:
        path=checked(row)
        if path.is_file() and sha(path)==row['sha256']:
            print('VERIFIED',row['path']);continue
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.part')
        request=urllib.request.Request(row['url'])
        # Optional Hugging Face token is read from the environment, never printed.
        if os.environ.get('HF_TOKEN'):request.add_header('Authorization','Bearer '+os.environ['HF_TOKEN'])
        with urllib.request.urlopen(request,timeout=120) as response, temp.open('wb') as target:shutil.copyfileobj(response,target)
        if temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:raise ValueError('Downloaded base hash mismatch: '+row['path'])
        temp.replace(path);print('RESTORED',row['path'])

def adapters(directory, use_gh=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'Models/release_assets.json').read_text())
    for asset in manifest['assets']:
        archive=directory/asset['filename']
        if not archive.exists():
            if not use_gh:raise FileNotFoundError('Download the private release asset first: '+asset['filename'])
            subprocess.run(['gh','release','download',manifest['release_tag'],'--repo',manifest['repository'],
                '--pattern',asset['filename'],'--dir',str(directory)],check=True)
        if archive.stat().st_size!=asset['bytes'] or sha(archive)!=asset['sha256']:raise ValueError('Archive hash mismatch: '+asset['filename'])
        expected={row['path']:row for row in asset['members']}
        with tarfile.open(archive,'r:') as tf:
            if {m.name for m in tf.getmembers()}!=set(expected):raise ValueError('Unexpected archive membership')
            for member in tf.getmembers():
                if not member.isfile():raise ValueError('Only regular checkpoint files are accepted')
                row=expected[member.name];path=checked(row)
                if path.exists() and sha(path)==row['sha256']:continue
                path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.part')
                with tf.extractfile(member) as source,temp.open('wb') as target:shutil.copyfileobj(source,target)
                if temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:raise ValueError('Checkpoint hash mismatch: '+row['path'])
                temp.replace(path)
        print('VERIFIED_ARCHIVE',asset['filename'])
    for row in json.loads((ROOT/'Models/adapter_manifest.json').read_text()):
        path=checked(row)
        if row.get('same_bytes_as'):
            source=(ROOT/row['same_bytes_as']).resolve()
            if not source.is_relative_to(ROOT):raise ValueError('Unsafe shared-weight reference')
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                try:os.link(source,path)
                except OSError:shutil.copy2(source,path)
        if not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Adapter not restored correctly: '+row['path'])
    print('All adapter paths restored and verified')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',choices=['Mistral','Qwen','both']);p.add_argument('--adapters',action='store_true')
    p.add_argument('--asset-directory',default='YOUR PATHWAY');p.add_argument('--use-gh',action='store_true');a=p.parse_args()
    if a.base:
        for family in ['Mistral','Qwen'] if a.base=='both' else [a.base]:base(family)
    if a.adapters:
        if a.asset_directory=='YOUR PATHWAY':p.error('Set --asset-directory to your downloaded archive folder')
        adapters(a.asset_directory,a.use_gh)
