import sys, json, time, torch
sys.path.insert(0, '.')
import ref_modadd as M
out = {}
for name, filt in [('base', None), ('grokfast', ('grokfast', 0.98, 2.0))]:
    t0 = time.time()
    r = M.train(p=97, frac=0.5, steps=2000, seed=0, filt=filt, save_steps=(200, 2000), threads=2)
    out[name] = dict(first=r['first'], wall=round(time.time() - t0, 1), final_test=r['curves']['test_acc'][-1])
    print(name, out[name], flush=True)
    if name == 'base':
        A_mem, A_end = r['checkpoints'][200], r['checkpoints'][2000]
r1 = M.train(p=97, frac=0.5, steps=2000, seed=1, save_steps=(200, 2000), threads=2)
B_mem, B_end = r1['checkpoints'][200], r1['checkpoints'][2000]
print('seed1', r1['first'], flush=True)
comb = lambda base, d1, d0, t: {k: base[k] + t * (d1[k] - d0[k]) for k in base}
for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
    a = M.eval_state(comb(A_mem, A_end, A_mem, t)); b = M.eval_state(comb(B_mem, A_end, A_mem, t))
    print(f"t={t:+.2f} P1 test {a['test_acc']:.3f} train {a['train_acc']:.3f} | P2 transplant test {b['test_acc']:.3f}")
dA = torch.cat([(A_end[k] - A_mem[k]).flatten() for k in A_mem]); dB = torch.cat([(B_end[k] - B_mem[k]).flatten() for k in A_mem])
print('cos(dirA,dirB)', (dA @ dB / dA.norm() / dB.norm()).item())
