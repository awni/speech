
import numpy as np

model_config = {
    "dropout" : 0.0,
    "encoder" : {
        "conv" : [
            [32, 5, 32, 2]
        ],
        "rnn" : {
            "dim" : 16,
            "bidirectional" : False,
            "layers" : 1
        }
    }
}

def gen_fake_data(freq_dim, output_dim, max_time=100,
                  max_seq_len=20, batch_size=4):
    data = []
    for i in range(batch_size):
        inputs = np.random.randn(max_time, freq_dim)
        labels = np.random.randint(0, output_dim, max_seq_len)
        data.append((inputs, labels))
    inputs, labels = list(zip(*data))
    return inputs, labels

