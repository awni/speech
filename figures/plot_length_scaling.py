import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ctc = [18.1, 20.3, 21.8, 21.9]
seq2seq = [19.1, 25.3, 31.6, 38.3]
trans = [19.2, 24.7, 30.5, 27.7]

ticks = np.arange(len(ctc)) * 2
p1 = plt.bar(ticks - 0.5, ctc, width=0.5, color='#4682B4')
p2 = plt.bar(ticks, trans, width=0.5, color='#B2BABB')
p3 = plt.bar(ticks + 0.5, seq2seq, width=0.5, color="#CD5C5C")
plt.legend([p1, p2, p3], ['CTC', 'Transducer', 'Seq2Seq'], loc=0)
plt.xlim(-1, 7.5)
plt.xticks([0.25, 2.25, 4.25, 6.25], ['0-3', '3-4','4-5', '5+'])
plt.xlabel("Utterance Length (seconds)")
plt.ylabel("PER")
plt.savefig("errors_by_length.svg")
