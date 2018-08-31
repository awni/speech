# speech

Speech is an open-source package to build end-to-end models for automatic
speech recognition. Sequence-to-sequence models with attention,
Connectionist Temporal Classification and the RNN Sequence Transducer
are currently supported.

The goal of this software is to facilitate research in end-to-end models for
speech recognition. The models are implemented in PyTorch.

The software has only been tested in Python3.6. 

**We will not be providing backward compatability for Python2.7.**

## Install

We recommend creating a virtual environment and installing the python
requirements there.

```
virtualenv <path_to_your_env>
source <path_to_your_env>/bin/activate
pip install -r requirements.txt
```

Then follow the installation instructions for a version of
[PyTorch](http://pytorch.org/) which works for your machine.

After all the python requirements are installed, from the top level directory,
run:

```
make
```

The build process requires CMake as well as Make.

After that, source the `setup.sh` from the repo root.

```
source setup.sh
```

Consider adding this to your `bashrc`.

You can verify the install was successful by running the
tests from the `tests` directory.

```
cd tests
pytest
```

## Run 

To train a model run
```
python train.py <path_to_config>
```

After the model is done training you can evaluate it with

```
python eval.py <path_to_model> <path_to_data_json>
```

To see the available options for each script use `-h`: 

```
python {train, eval}.py -h
```

## Examples

For examples of model configurations and datasets, visit the examples
directory. Each example dataset should have instructions and/or scripts for
downloading and preparing the data. There should also be one or more model
configurations available. The results for each configuration will documented in
each examples corresponding `README.md`.
