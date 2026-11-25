import json, sys, time, torch
sys.path.insert(0, '.')
import ref_neox as R
from safetensors.torch import load_file
from tokenizers import Tokenizer
MD = sys.argv[1]
TEXT = ("The river leaves the mountains in late spring, when the snow begins to melt and the water turns "
        "grey with silt. Farmers in the valley wait for this moment, because the flood brings soil to the "
        "fields and fills the canals that run between the villages. In the old town, the market opens at "
        "dawn. Traders set out baskets of bread, cheese and dried fish, and children carry water from the "
        "well to the stalls. By noon the square is crowded, and the sound of bargaining fills the narrow "
        "streets. When the sun goes down, the lamps are lit one by one, and the town falls quiet again. "
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n")
def run(name, cfg, W0, ids):
    t0 = time.time()
    l0 = R.forward(cfg, W0, ids); base = R.next_token_loss(l0, ids); M = l0.abs().max().item()
    print(f'== {name}: tokens={ids.shape[1]} base loss {base:.4f} max|logit| {M:.2f} fwd {1000*(time.time()-t0):.0f} ms')
    def m(tag, W, kind='ln'):
        lo = R.forward(cfg, W, ids, kind); d = (lo - l0).abs().max().item(); dl = R.next_token_loss(lo, ids) - base
        cls = 'exact' if d / M <= 1e-9 else ('near' if d / M <= 1e-3 else 'edit')
        print(f'{tag:46s} max|dl|={d:.3e} rel={d/M:.2e} dloss={dl:+.4f} -> {cls}')
    g = torch.Generator().manual_seed(0)
    mid = cfg.n_layers // 2; D, DH, RD = cfg.d, cfg.dh, cfg.rd
    m('S1 permute(L[mid].mlp, randperm)', R.perm_mlp(cfg, W0, mid, R.rand_perm(cfg.d_mlp, g)))
    m('S2 permute(L[mid].heads, randperm)', R.perm_heads(cfg, W0, mid, R.rand_perm(cfg.n_heads, g).tolist()))
    m('S3 vo(L[mid].H[0], gl)', R.vo_gauge(cfg, W0, mid, 0, R.rand_gl(DH, g)))
    m('S4 qk(L[mid].H[0], gl, dims=pass)', R.qk_gauge(cfg, W0, mid, 0, R.rand_gl(DH - RD, g), torch.arange(RD, DH)))
    m('S4b rope_pairs(L[mid].H[0], phis, 1.7)', R.rope_pairs(cfg, W0, mid, 0, [0.3, -1.1, 2.0, 0.7][:RD // 2], 1.7))
    Wf = R.fold_ln(cfg, W0); Wfc = R.center_writers(cfg, Wf)
    m('S5a fold_ln()', Wf); m('S5b fold_ln(); center_writers()', Wfc)
    m('S5c fold_ln(); rotate(resid, orth1)', R.rotate_resid(cfg, Wf, R.rand_orth_fixing_ones(D, g)))
    m('S5d fold_ln(); center_writers(); ln_to_rms()', Wfc, 'rms')
    m('S5e ... ln_to_rms(); rotate(resid, orth)', R.rotate_resid(cfg, Wfc, R.rand_orth(D, g)), 'rms')
    m('B1 scale(resid, 2)', R.scale_resid(cfg, W0, 2.0))
    m('B2 rotate(resid, orth1)   [no fold]', R.rotate_resid(cfg, W0, R.rand_orth_fixing_ones(D, g)))
    m('B3 fold_ln(); rotate(resid, orth)', R.rotate_resid(cfg, Wf, R.rand_orth(D, g)))
    m('B4 scale(L[mid].mlp, uniform(0.5,2))', R.scale_mlp(cfg, W0, mid, R.rand_uniform(cfg.d_mlp, 0.5, 2.0, g)))
    m('B5 rotate(L[mid].mlp, orth)', R.rotate_mlp(cfg, W0, mid, R.rand_orth(cfg.d_mlp, g)))
    m('B6 qk(L[mid].H[0], gl, dims=all)', R.qk_gauge(cfg, W0, mid, 0, R.rand_gl(DH, g), torch.arange(0, DH)))
    L2, L3, L4 = (R.get_layer(W0, mid + o) for o in (-1, 0, 1))
    m('E0 delete(L[mid])', R.zero_writers(W0, [mid]))
    m('E1 swap(L[mid-1], L[mid])', R.set_layer(R.set_layer(W0, mid - 1, L3), mid, L2))
    avg = R.lerp_layer(L2, L3, 0.5)
    m('E2 L2:=lerp(L2,L3,.5); L3:=L2', R.set_layer(R.set_layer(W0, mid - 1, avg), mid, avg))
    al = R.align_layer(cfg, L3, L2)
    m('E2s L3 := align(L3 -> L2)', R.set_layer(W0, mid, al))
    avga = R.lerp_layer(L2, al, 0.5)
    m('E2a L2:=lerp(L2,align(L3->L2),.5); L3:=L2', R.set_layer(R.set_layer(W0, mid - 1, avga), mid, avga))
    m('E2r rotate(resid, orth, layers=[mid], embed=false, unembed=false)', R.rotate_resid(cfg, W0, R.rand_orth(D, g), layers=[mid], embed=False, unembed=False))
    m('E3 L[mid+1] := neg_writers(L[mid-1])', R.set_layer(W0, mid + 1, R.neg_writers(L2)))
    m('E3ref4 delete(L[mid+1])', R.zero_writers(W0, [mid + 1]))
    m('E3ref delete(L[mid-1]); delete(L[mid+1])', R.zero_writers(W0, [mid - 1, mid + 1]))
    m('E3n L[mid+1] := neg(L[mid-1])', R.set_layer(W0, mid + 1, R.neg_all(L2)))
    m('E4 L[mid] := lerp(L[mid-1], L[mid+1], .5)', R.set_layer(W0, mid, R.lerp_layer(L2, L4, 0.5)))
    key = R.P.format(mid) + 'attention.dense.weight'
    m('G1 gridop(rot90, dense[0:64,0:64])', R.gridop_block(W0, 'rot90', key, 0, min(64, D), 0, min(64, D), lambda b: torch.rot90(b, -1, (0, 1))))
    print(f'total {time.time()-t0:.1f} s')
cfgj = json.load(open(f'{MD}/config.json'))
cfg = R.cfg_from_hf_config(cfgj)
W0 = {k: v.to(torch.float64) for k, v in load_file(f'{MD}/model.safetensors').items()}
ids = torch.tensor([Tokenizer.from_file(f'{MD}/tokenizer.json').encode(TEXT).ids])
run('pythia-14m', cfg, W0, ids)
run('neox-tiny', R.TINY_CFG, R.make_tiny(0), torch.tensor([R.TINY_IDS]))
