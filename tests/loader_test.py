import numpy as np

from speech import loader

def test_dataset():
    batch_size = 2
    data_json = "test.json"
    preproc = loader.Preprocessor(data_json)
    dataset = loader.AudioDataset(data_json, preproc, batch_size)

    # Num chars plus start and end tokens
    assert preproc.vocab_size == 11
    s_idx = preproc.vocab_size - 1
    assert preproc.int_to_char[s_idx] == preproc.START

    inputs, targets = dataset[0]

    # Inputs should be time x frequency
    assert inputs.shape[1] == preproc.input_dim
    assert inputs.dtype == np.float32

    # Correct number of examples
    assert len(dataset.data) == 8

def test_loader():

    batch_size = 2
    data_json = "test.json"
    preproc = loader.Preprocessor(data_json)
    ldr = loader.make_loader(data_json, preproc,
            batch_size, num_workers=0)

    # Test that batches are properly sorted by size
    for inputs, labels in ldr:
        assert inputs[0].shape == inputs[1].shape
