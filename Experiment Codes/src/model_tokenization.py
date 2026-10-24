"""Exact local Mistral template and pinned Qwen ChatML; CPU only."""
from pathlib import Path
import ast
import base64
import json
import unicodedata

class NativeTokenizer:
    def __init__(self, root):
        root=Path(root)
        config=json.loads((root/'config/model.json').read_text())
        self.family=config['family']
        folder=(root/config['base_model_dir']).resolve()
        if self.family=='Mistral':
            from tokenizers import Tokenizer
            from jinja2 import Environment, StrictUndefined
            self.encoder=Tokenizer.from_file(str(folder/'tokenizer.json'))
            self.config=json.loads((folder/'tokenizer_config.json').read_text())
            self.template=Environment(undefined=StrictUndefined).from_string(self.config['chat_template'])
        elif self.family=='Qwen':
            import tiktoken
            source=ast.parse((folder/'tokenization_qwen.py').read_text())
            pat=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='PAT_STR' for t in n.targets))
            ranks={base64.b64decode(token):int(rank) for line in (folder/'qwen.tiktoken').read_bytes().splitlines()
                   if line for token,rank in [line.split()]}
            special={text:151643+i for i,text in enumerate(['<|endoftext|>','<|im_start|>','<|im_end|>']+
                     [f'<|extra_{i}|>' for i in range(205)])}
            self.encoder=tiktoken.Encoding('pinned_qwen',pat_str=pat,mergeable_ranks=ranks,special_tokens=special)
        else: raise ValueError('Unknown model family')

    def prepare(self, content):
        if self.family=='Mistral':
            def invalid(message):raise ValueError(message)
            text=self.template.render(messages=[{'role':'user','content':content}],
                 bos_token=self.config['bos_token'],eos_token=self.config['eos_token'],
                 raise_exception=invalid,add_generation_prompt=True)
            return text,self.encoder.encode(text,add_special_tokens=False).ids
        encode=lambda text:self.encoder.encode(unicodedata.normalize('NFC',text),allowed_special=set(),disallowed_special=())
        nl=encode('\n')
        ids=([151644]+encode('system')+nl+encode('You are a helpful assistant.')+[151645]+nl+
             [151644]+encode('user')+nl+encode(content)+[151645]+nl+[151644]+encode('assistant')+nl)
        return self.encoder.decode(ids),ids
