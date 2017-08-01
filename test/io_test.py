import tempfile

import speech.model
import speech.loader

def test_save():

    freq_dim = 120
    output_dim = 10
    config = {
        "encoder_layers" : 1,
        "rnn_dim" : 16
    }

    model = speech.model.Model(freq_dim, output_dim, config)

    batch_size = 2
    data_json = "test.json"
    preproc = speech.loader.Preprocessor(data_json)

    save_dir = tempfile.mkdtemp()
    speech.save(model, preproc, save_dir)

    s_model, s_preproc = speech.load(save_dir)
    assert hasattr(s_preproc, 'mean')
    assert hasattr(s_preproc, 'std')
    assert hasattr(s_preproc, 'int_to_char')
    assert hasattr(s_preproc, 'char_to_int')

    msd = model.state_dict()
    for k, v in s_model.state_dict().items():
        assert k in msd
    assert hasattr(s_model, 'freq_dim')
    assert hasattr(s_model, 'output_dim')
