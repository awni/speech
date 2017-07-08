from speech import loader

def test_dataset():
    batch_size = 2
    data_json = "test/test.json"
    dataset = loader.AudioDataset("test.json", batch_size)

    # Correct number of examples
    assert len(dataset.data) == 8

    # Num chars plus start and end tokens
    assert dataset.output_dim == 11

