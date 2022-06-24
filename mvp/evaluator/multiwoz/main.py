import torch
import sys
import os
import torch
import sys
import os
import json
from collections import OrderedDict
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import Config
from reader import MultiWozReader
from eval import MultiWozEvaluator
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration

device = torch.device('cuda')

def generate(model, tokenizer, text):
    encoding_dict = tokenizer(
        text, max_length=1024, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids = encoding_dict['input_ids'].to(device)
    attn_masks = encoding_dict['attention_mask'].to(device)
    output = model.generate(
        input_ids, attention_mask=attn_masks, num_beams=5, no_repeat_ngram_size=3, max_length=128, early_stopping=True
    )
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return generated_text

def truncate(text):
    input = tokenizer(text)['input_ids'][1:-1][-900:]
    text = tokenizer.decode(input, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text

cfg = Config('multiwoz')
reader = MultiWozReader(cfg)
eval = MultiWozEvaluator(reader, cfg)


a = torch.load('../saved/BART-multiwoz-Jan-10-2022_23-35-47.pthlast')
tokenizer = BartTokenizer.from_pretrained('../pretrained_model/bart-base', additional_special_tokens = ['<_PAD_>', '<go_r>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
                '<eos_a>', '<go_d>','<eos_d>', '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', 
                '<sos_db>', '<eos_db>', '<sos_context>', '<eos_context>'])
configuration = BartConfig.from_pretrained('../pretrained_model/bart-base')

model = BartForConditionalGeneration(config=configuration)
model.resize_token_embeddings(len(tokenizer))
state = OrderedDict()
for k, v in a['state_dict'].items():
    state[k[6:]] = v
model.load_state_dict(state)
model.to(device)
model.eval()

with open(f'multiwoz/multi-woz-fine-processed/multiwoz-fine-processed-test.json', 'r') as f:
    data = json.load(f)

src = []
tgt = []
datas = []
for d in tqdm(data):
    prev = ""
    bs_inputs = []
    for t in d:
        bs_input = 'translate dialogue to belief state: <sos_context>' + truncate(prev + t['user']) + '<eos_context>'
        bs_inputs.append(bs_input)
        prev += t['user'] + t['resp']
    
    bs_outputs = generate(model, tokenizer, bs_inputs)

    da_inputs = []
    nlg_inputs = []
    prev = ""
    for t, bs_output in zip(d, bs_outputs):
        t['bspn_gen'] = bs_output
        db_text = '<sos_db> ' + reader.bspan_to_DBpointer(bs_output, t['turn_domain']) + ' <eos_db>'
        da_input = 'translate dialogue to dialogue action: <sos_context>' + truncate(prev + t['user']) + '<eos_context>' + db_text
        nlg_input = 'translate dialogue to system response: <sos_context>' + truncate(prev + t['user']) + '<eos_context>' + db_text
        prev += t['user'] + t['resp']
        
        da_inputs.append(da_input)
        nlg_inputs.append(nlg_input)

    da_outputs = generate(model, tokenizer, da_inputs)
    nlg_outputs = generate(model, tokenizer, nlg_inputs)
    for t, da_output, nlg_output in zip(d, da_outputs, nlg_outputs):
        t['aspn_gen'] = da_output
        t['resp_gen'] = nlg_output
        datas.append(t)
torch.save(datas, 'res.pth')

datas = torch.load('res.pth')
datas = datas[:-1]


'''
with open('res.json', 'r') as f:
    res = json.load(f)

for d, r in zip(datas, res):
    for k, v in d.items():
        if isinstance(v, str) and v.startswith('<sos'):
            d[k] = v[8:-8].strip()
    d['pointer'] = r['pointer']
    r['bspn_gen'] = d['bspn_gen']
    r['aspn_gen'] = d['aspn_gen']
    r['resp_gen'] = d['resp_gen']



print('bleu, success, inform')
print(eval.validation_metric(datas))
print(eval.validation_metric(res))
'''