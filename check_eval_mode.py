import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-acvr')
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor

CKPT = '/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-acvr/outputs/reference_320_model_only.pth'
PRE = '/root/autodl-tmp/rankup_project/pretrained/roberta-base'
texts = ['I feel happy today.', 'This is terrible and sad.', 'What a wonderful surprise!', 'I am angry and frustrated.']
tok = AutoTokenizer.from_pretrained(PRE, local_files_only=True, use_fast=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
model = RobertaVADRegressor(PRE, dropout=0.1, offline=True).to(dev)
model.load_state_dict(d['model'], strict=True)
model.eval()

@torch.inference_mode()
def enc(train_mode):
    model.train(train_mode)
    b = tok(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    b = {k: v.to(dev) for k, v in b.items()}
    st = model.encoder(**b, return_dict=True).last_hidden_state
    m = b['attention_mask'].unsqueeze(-1).to(dtype=st.dtype)
    return F.normalize((st * m).sum(1) / m.sum(1).clamp_min(1), dim=-1)

a1, a2 = enc(True), enc(True)
e1, e2 = enc(False), enc(False)
print('train repeat max diff', float((a1 - a2).abs().max()))
print('eval  repeat max diff', float((e1 - e2).abs().max()))
print('train vs eval max diff', float((a1 - e1).abs().max()))
