import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import tensorflow as tf
import numpy as np

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim([-100, 25100])
ax.set_ylim([20, 160])

model_path = "/afs/cs.stanford.edu/u/awni/scr/speech/examples/timit/models/seq2seq_ali_save"

def errors_from_file():
    f = glob.glob(model_path + "/events*")
    assert len(f) == 1, "Wrong num files"
    log_file = f[0]
    losses = []
    curr_step = 0
    for e, summary in enumerate(tf.train.summary_iterator(log_file)):
        if e == 0:
            continue
        step = summary.step
        if int(step) != curr_step:
            continue
        curr_step += 1
        vals = summary.summary.value.pop()
        loss = vals.simple_value
        losses.append(loss)
    return losses

losses = errors_from_file()
smooth = 50
# assuming we drop the last smooth
losses = np.convolve(losses, (1./smooth) * np.ones(smooth), mode="valid")

def get_plot(idx):
    p = ax.plot(losses[:idx*50], 'b')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    return p

def load_train_plots(max_id=500):
    plots = []
    for i in range(0, max_id):
        plots.append(get_plot(i))
    return plots

plots = load_train_plots()
ani = animation.ArtistAnimation(fig, plots, interval=100,
            repeat=False, blit=True)
ani.save('test_train_vid.mp4', metadata={'artist':'Awni Hannun'})
