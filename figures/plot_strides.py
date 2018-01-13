import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

strides = [1, 2, 4, 8, 16]
ctc = [15.8, 15.4, 16.0]
seq2seq = [17.8, 17.5, 18.1, 24.4, 40.0]
transducer = [16.5, 16.9, 19.5, 26.4, 36.8]

plt.figure(figsize=(10,5))
l1, = plt.plot(strides[:len(ctc)], ctc, color="#4682B4", linewidth=1.5)
plt.plot(strides[:len(ctc)], ctc, '.', color="#4682B4", markersize=8)
l2, = plt.plot(strides, seq2seq, color="#CD5C5C", linewidth=1.5)
plt.plot(strides, seq2seq, '.', color="#CD5C5C", markersize=8)
l3, = plt.plot(strides, transducer, color="#B2BABB", linewidth=1.5)
plt.plot(strides, transducer, '.', color="#B2BABB", markersize=8)
plt.legend([l1, l2, l3], ["CTC", "Seq2Seq", "Transducer"], loc=0)
plt.xticks(strides)
plt.xlim((0.5, 18))
plt.ylim((10, 42))

plt.savefig("cer_by_stride.svg")
