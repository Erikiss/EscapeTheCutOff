import json, sys, numpy as np
sys.path.insert(0, '.')
import ref_grid as G
root = sys.argv[1]
def exs(t):
    d = json.load(open(f'{root}/{t}.json')); return d['train'] + d['test'] + d['arc-gen']
for t, prim in [('task087','rot180'),('task140','rot180'),('task150','vmirror'),('task155','hmirror'),('task179','dmirror'),('task241','dmirror'),('task380','rot270')]:
    E = exs(t)
    p = G.check(lambda g: G.PRIMITIVES[prim](np.array(g)).tolist(), E)[:2]
    f = G.check(G.CompiledNet(prim, E, 'fixed'), E)[:2]
    g = G.check(G.CompiledNet(prim, E, 'gated'), E)[:2]
    print(t, prim, 'prim', p, 'fixed', f, 'gated', g)
# decode/onehot roundtrip
g = [[1,2,3],[4,5,0]]
assert G.decode(G.onehot(g)) == g
print('roundtrip ok; note colour 0 is a real colour (black), padding is all-zero')
# single-primitive scan over all 400 with all primitives incl identity
hits = []
for i in range(1, 401):
    t = f'task{i:03d}'; E = exs(t)
    for name, f in G.PRIMITIVES.items():
        if G.check(lambda g: f(np.array(g)).tolist(), E)[0] == len(E): hits.append((t, name))
print('single-primitive tasks:', hits)
