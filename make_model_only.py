import torch

src = '/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-m1-validity-v2/saved_models/vad_research/m1_target_validity_v2/reference_320_only.pth'
dst = '/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-acvr/outputs/reference_320_model_only.pth'
d = torch.load(src, map_location='cpu', weights_only=False, mmap=True)
out = {
    'model': d['model'],
    'train_steps': d.get('train_steps'),
    'git_commit': d.get('git_commit'),
    'source_sha256': d.get('source_sha256'),
    'fit_id_sha256': d.get('fit_id_sha256'),
    'plan_sha256': d.get('plan_sha256'),
}
torch.save(out, dst)
print('saved', dst)
print('keys', list(out.keys()))
