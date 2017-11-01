The Timit Speech corpus must be purchased from the LDC to run these
experiments. The catalog number is [LDC93S1].

The data is mapped from 61 to 48 phonemes for training. For final test set
evaluation the 48 phonemes are again mapped to 39. The phoneme mapping is the
standard recipe, the map used here is taken from the [Kaldi TIMIT recipe].

## Setup

Once you have the TIMIT data downloaded, run

```
./data_prep.sh <path_to_timit>
```

This script will convert the `.flac` to `.wav` files and store them in the same
location. You'll need write access to directory where timit is stored. It will
then symlink the timit directory to `./data`. There should be three data json
files in `data/timit`:

- `train.json`: 4448 utterances
- `dev.json`: 192 utterances from 50 held-out speakers
- `test.json`: 400 utterances, the standard TIMIT test set

## Train 

There is a CTC and a sequence-to-sequence with attention configuration. Before
training a model, edit the configuration file. In particular, set the
`save_path` to a location where you'd like to store the model. Edit any other
parameters for your experiment. From the top-level directory, you can train the
model with

``` 
python train.py examples/timit/seq2seq_config.json
```

## Score

Save the 48 phoneme predictions with the top-level `eval.py` script.

```
python eval.py <path_to_model> examples/timit/data/timit/text.json --save predictions.json
```

To score using the reduced phoneme set (39 phonemes) run 

```
python examples/timit/score.py predictions.json 
```

## Results

### seq2seq

These are the dev and test results for the best sequence-to-sequence model with attention. The configuration can be found in
`seq2seq_config.json`. Note this is without an external LM and with a beam size of 1. Also we don't use any speaker adaptation or sophisticated features (MFCCs). Results *should* improve with these features.

- Dev: 17.6 CER
- Test: 19.8 CER

### CTC (TODO)

[Kaldi TIMIT recipe]: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
[LDC93S1]: https://catalog.ldc.upenn.edu/ldc93s1
