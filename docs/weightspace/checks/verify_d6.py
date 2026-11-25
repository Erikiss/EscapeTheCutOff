import sys, torch, json, numpy as np
sys.path.insert(0, '.')
import ref_neox as R, ref_grid as G
def gen(line, k, seed=0): return torch.Generator().manual_seed(seed * 1_000_003 + line * 1009 + k)
cfg = R.TINY_CFG; W0 = R.make_tiny(0); ids = torch.tensor([R.TINY_IDS])
l0 = R.forward(cfg, W0, ids); M = l0.abs().max().item(); base = R.next_token_loss(l0, ids)
def cls(W, kind='ln'):
    lo = R.forward(cfg, W, ids, kind); d = (lo - l0).abs().max().item(); rel = d / M
    return ('exact' if rel <= 1e-9 else 'near' if rel <= 1e-3 else 'edit'), rel, R.next_token_loss(lo, ids) - base
mid = 2; D, DH, RD, DM = cfg.d, cfg.dh, cfg.rd, cfg.d_mlp
# line numbers as in Appendix D.6 (line 1 model, 2 seed, statements from line 3)
S = {}
S['T1'] = (3, lambda L: R.rotate_resid(cfg, R.fold_ln(cfg, W0), R.rand_orth_fixing_ones(D, gen(L, 0))), 'ln')
S['T2'] = (4, lambda L: R.rotate_resid(cfg, W0, R.rand_orth_fixing_ones(D, gen(L, 0))), 'ln')
S['T3'] = (5, lambda L: R.rotate_resid(cfg, R.fold_ln(cfg, W0), R.rand_orth(D, gen(L, 0))), 'ln')
S['T4'] = (6, lambda L: R.rotate_resid(cfg, R.center_writers(cfg, R.fold_ln(cfg, W0)), R.rand_orth(D, gen(L, 0))), 'rms')
S['T5'] = (7, lambda L: R.rotate_resid(cfg, W0, R.rand_orth(D, gen(L, 0)), layers=[2], embed=False, unembed=False), 'ln')
S['T6'] = (8, lambda L: R.vo_gauge(cfg, W0, mid, 0, R.rand_gl(DH, gen(L, 0))), 'ln')
S['T7'] = (9, lambda L: R.qk_gauge(cfg, W0, mid, 0, R.rand_gl(DH - RD, gen(L, 0)), torch.arange(RD, DH)), 'ln')
S['T8'] = (10, lambda L: R.qk_gauge(cfg, W0, mid, 0, R.rand_gl(DH, gen(L, 0)), torch.arange(0, DH)), 'ln')
S['T9'] = (11, lambda L: R.rope_pairs(cfg, W0, mid, 1, [0.3, -1.1], 1.7), 'ln')
S['T10'] = (12, lambda L: R.perm_mlp(cfg, W0, mid, R.rand_perm(DM, gen(L, 0))), 'ln')
S['T11'] = (13, lambda L: R.rotate_mlp(cfg, W0, mid, R.rand_orth(DM, gen(L, 0))), 'ln')
S['T12'] = (14, lambda L: R.perm_heads(cfg, W0, mid, [1, 0]), 'ln')
S['T13'] = (15, lambda L: R.scale_mlp(cfg, W0, mid, R.rand_uniform(DM, 0.5, 2.0, gen(L, 0))), 'ln')
S['T14'] = (16, lambda L: R.center_writers(cfg, R.fold_ln(cfg, W0)), 'rms')
S['T15'] = (17, lambda L: R.scale_resid(cfg, W0, 2.0), 'ln')
S['T16'] = (18, lambda L: R.scale_resid(cfg, W0, 2.0), 'ln')
S['T17'] = (19, lambda L: R.set_layer(W0, mid, R.align_layer(cfg, R.get_layer(W0, mid), R.get_layer(W0, mid - 1))), 'ln')
S['T18'] = (20, lambda L: R.perm_mlp(cfg, W0, 1, R.rand_perm(DM, gen(L, 0))), 'ln')
S['T19'] = (21, lambda L: R.set_layer(W0, mid, R.lerp_layer(R.get_layer(W0, mid - 1), R.get_layer(W0, mid + 1), 0.5)), 'ln')
for k, (line, f, kind) in S.items():
    c, rel, dl = cls(f(line), kind); print(f'{k:4s} line {line:2d} {c:6s} rel={rel:.2e} dloss={dl:+.4f}')
root = '/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl/dl_golf/sakana/google-code-golf-2025'
d = json.load(open(f'{root}/task150.json')); E = d['train'] + d['test'] + d['arc-gen']
print('V5 gated from=train', G.check(G.CompiledNet('vmirror', d['train'], 'gated'), E)[:2])
print('V6 prim hmirror', G.check(lambda g: G.PRIMITIVES['hmirror'](np.array(g)).tolist(), E)[:2])
print('V3 fixed by split', [G.check(G.CompiledNet('vmirror', E, 'fixed'), d[s])[:2] for s in ('train', 'test', 'arc-gen')])
d7 = json.load(open(f'{root}/task087.json')); E7 = d7['train'] + d7['test'] + d7['arc-gen']
print('V7 chain', G.check(lambda g: G.PRIMITIVES['hmirror'](G.PRIMITIVES['vmirror'](np.array(g))).tolist(), E7)[:2])
c = G.CompiledNet('vmirror', E, 'gated'); print('gated sizes', list(c.table), 'nnz', sum(int((s >= 0).sum()) for s in c.table.values()))
c = G.CompiledNet('vmirror', E, 'fixed'); print('fixed nnz', sum(int((s >= 0).sum()) for s in c.table.values()))
