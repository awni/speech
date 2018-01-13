import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cPickle as pickle

with open("trans_alis.bin", 'r') as fid:
    trans_alis = pickle.load(fid)

with open("ctc_alis.bin", 'r') as fid:
    ctc_alis = pickle.load(fid)

with open("seq2seq_alis.bin", 'r') as fid:
    seq2seq_alis = pickle.load(fid)

def load_gts(key):
    phn_f = ".".join(k.split(".")[:-1]) + ".phn"
    with open(phn_f, 'r') as fid:
        lines = (l.strip().split() for l in fid)
        lines = [(p, int(s), int(e)) for s, e, p in lines if p != 'q']
    return key, lines

keys = [k for k, _ in ctc_alis]
gt_alis = [load_gts(k) for k in keys]

seq2seq_alis = dict(seq2seq_alis)
ctc_alis = dict(ctc_alis)
trans_alis = dict(trans_alis)
gt_alis = dict(gt_alis)

# Step 3: Compute individual scores.
def score(gt, pred):
    corr = 0.0; tot = 0.0; dist = 0.0
    for k, v in gt.iteritems():
        pv = pred[k]
        if len(v) != len(pv):
            print k
        for (p_gt, s_gt, e_gt), (p_p, s_p, e_p) in zip(v, pv):
            mid = (e_p + s_p) / 2.0
            if s_gt <= mid <= e_gt:
                corr += 1
            else:
                dist += min(abs(s_gt - mid), abs(e_gt - mid))
            tot += 1
    return corr / tot, dist

seq2seq_score = score(gt_alis, seq2seq_alis)[0]
ctc_score = score(gt_alis, ctc_alis)[0]
trans_score = score(gt_alis, trans_alis)[0]
print seq2seq_score, ctc_score, trans_score

fig, ax = plt.subplots()

ax.bar([0], [seq2seq_score], 0.5, color='r')
ax.bar([1], [ctc_score], 0.5, color='g')
ax.bar([2], [trans_score], 0.5, color='b')

plt.savefig("align_errors.svg")
