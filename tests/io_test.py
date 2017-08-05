import tempfile

import speech.models
import speech.loader

import shared

def test_save():

    freq_dim = 120
    model = speech.models.Model(freq_dim,
                    shared.model_config)

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
    assert hasattr(s_model, 'encoder_dim')
    assert hasattr(s_model, 'is_cuda')
