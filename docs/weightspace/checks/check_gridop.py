import sys, torch
sys.path.insert(0, '.')
import ref_neox as R
cfg = R.TINY_CFG; W0 = R.make_tiny(0); key = R.P.format(2) + 'attention.dense.weight'
for name in R.TORCH_PRIMITIVES:
    for (r0, r1, c0, c1) in [(0, 16, 0, 16), (2, 10, 4, 12)]:
        if name in R.SQUARE_ONLY and (r1 - r0) != (c1 - c0): continue
        W1 = R.gridop_block(W0, name, key, r0, r1, c0, c1)
        ref = W0[key][r0:r1, c0:c1].clone().numpy()
        import numpy as np
        npf = {'identity': lambda g: g, 'rot90': lambda g: np.rot90(g, -1), 'rot180': lambda g: np.rot90(g, 2), 'rot270': lambda g: np.rot90(g, 1),
               'hmirror': lambda g: g[::-1, :], 'vmirror': lambda g: g[:, ::-1], 'dmirror': lambda g: g.T, 'cmirror': lambda g: np.rot90(g, 2).T}[name]
        assert np.array_equal(W1[key][r0:r1, c0:c1].numpy(), npf(ref)), (name, r0)
        assert torch.equal(W0[key], R.make_tiny(0)[key])   # input untouched
try:
    R.gridop_block(W0, 'rot90', key, 0, 4, 0, 8); print('no error?!')
except ValueError as e: print('non-square rot90 ->', e)
print('gridop ok for all primitives incl. dmirror on full and sub blocks')
