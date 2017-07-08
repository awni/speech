import numpy as np

from speech import loader

def test_dataset():
    batch_size = 2
    data_json = "test.json"
    dataset = loader.AudioDataset(data_json, batch_size)

    # Correct number of examples
    assert len(dataset.data) == 8

    # Num chars plus start and end tokens
    assert dataset.output_dim == 11

    inputs, targets = dataset[0]
    # Inputs should be time x frequency
    assert inputs.shape[1] == dataset.input_dim

    assert inputs.dtype == np.float32
