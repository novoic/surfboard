<p align="center">
<a href="https://novoic.com">
    <img src="https://assets.novoic.com/surfboard.png" alt="surfboard-logo" border="0">
</a>
  <br />
  <br />
<a href='https://surfboard.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/surfboard/badge/?version=latest' alt='Documentation Status' />
</a>
<a href='https://app.circleci.com/pipelines/github/novoic/surfboard'>
    <img src='https://circleci.com/gh/novoic/surfboard.svg?style=shield&circle-token=a1b00a7def3a0a97090888e7380b771f58836046' alt='Build Status' />
</a>
</p>

_A Python package for modern audio feature extraction_

For information about contributing, citing, licensing (including commercial licensing) and getting in touch, please see [our wiki](https://github.com/novoic/surfboard/wiki).

Our documentation can be found [here](https://surfboard.readthedocs.io/en/latest). Our paper can be found [here](https://arxiv.org/abs/2005.08848). 

Please join our [Slack channel](https://join.slack.com/t/surfboard-novoic/shared_invite/zt-f5usu4qo-PHLEyOTk8NE1lfn_hHnxoA) if you have questions or suggestions!

## :surfer: Installation :surfer:

Install using pip
```bash
pip install surfboard
```

Alternatively,
* Clone the repo: `git clone https://github.com/novoic/surfboard.git`
* Navigate into the repo: `cd surfboard`
* Install the repo: `pip3 install .`

## Quickstart (be cooler than [Bodhi in Point Break](https://youtu.be/LniUPlffoB0) in 2 minutes)

### Example 0: Compute features using Python
Given a set of components and an optional set of statistics to apply to the time-varying components, extract them using Python.
```python
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features

sound = Waveform(path='/path/to/audio.wav')

# Option 1: Extract MFCC and RMS energy components as time series.
component_dataframe = extract_features([sound], ['mfcc', 'rms'])

# Option 2: Extract the mean and standard deviation of the MFCC and RMS energy features over time.
feature_dataframe = extract_features([sound], ['mfcc', 'rms'], ['mean', 'std'])

# Option 3: Extract MFCC and RMS energy features as time series with non-default arguments.
mfcc_with_arg = {'mfcc': {'n_mfcc': 26, 'n_fft_seconds': 0.08, 'hop_length_seconds': 0.02}}
feature_with_args_dataframe = extract_features([sound], [mfcc_with_arg, 'rms'], ['mean', 'std'])
```

### Example 1: Compute audio features from a folder of `.wav` files
Assume the following directory structure:
```
my_wav_folder/
│   swell.wav
│   cool_hair.wav
|   wave_crash.wav
|   ...
```
Using a `.yaml` config (see `example_configs` for examples), you can use the surfboard CLI to return a `.csv` file containing a set of features computed for every `.wav` file in `my_wav_folder`. You can optionally use multiple processes with the `-j` flag.
```bash
surfboard compute-features -i my_wav_folder -o cool_features.csv -F surfboard/example_configs/spectral_features.yaml -j 4
```

### Example 2: Create a custom `.yaml` config
You can create a custom `.yaml` config in order to extract specific features from your audio data. You can also pick specific statistics to apply to the time-varying components. The set of available statistics is described in the __Available statistics__ section below.

Take a peak at the example configs in `surfboard/example_configs/`. The package assumes a `.yaml` file with the following structure:
```yaml
components:
  - mfcc
      n_mfcc: 26
  - log_melspec
      n_mels: 64
  
statistics:
  - mean
  - std
```

This config will compute the mean and standard deviation of every MFCC (13 by default but set to 26 here) and log mel-spectrogram filterbank (128 by default but 64 here) on every `.wav` file in `my_wav_folder` if called with the following command:
```bash
surfboard compute-features -i my_wav_folder -o epic_features.csv -F my_config.yaml
```

### Example 3: Use the `compute-components` functionality
Sometimes you might want to retain the time axis of time-dependent components but still use the CLI. Given a `.yaml` config without a `statistics` section, you can. It will dump the components as `.pkl` file which can be loaded with `pd.read_pickle`.
```bash
surfboard compute-components -i my_wav_folder -o epic_features.pkl -F surfboard/example_configs/chroma_components.yaml
```

## Example 4: 
We have provided notebooks in `notebook_tutorials` for examples of how Surfboard can be used to extract features from audio and even to perform environmental sound classification.  
Otherwise, here are some examples:

Define a waveform:
```python
from surfboard.sound import Waveform
import numpy as np

# Instantiate from a .wav file.
sound = Waveform(path="/surf/in/USA/sound.wav", sample_rate=44100)

# OR: instantiate from a numpy array.
sound = Waveform(signal=np.sin(np.arange(0, 2 * np.pi, 1/24000)), sample_rate=44100)
```
Get the F0 contour:
```python
import matplotlib.pyplot as plt
f0_contour = sound.f0_contour()
plt.plot(f0_contour[0])
```
Get the MFCCs:
```python
mfccs = sound.mfcc()
```
Get different shimmers, jitters, formants:
```python
shimmers = sound.shimmers()
jitters = sound.jitters()
formants = sound.formants()
```

## Available components

You can take a look at `COMPONENTS.md` to see which components can be computed using Surfboard.

There is extensive documentation in the method docstrings in `surfboard/sound.py`. Please refer to those for more details on each individual feature (or to our documentation, alternatively). 


## Available statistics

A thorough list of the statistics implemented in Surfboard can be found in `STATISTICS.md`

Often, the components computed from the `surfboard.sound.Waveform` class have a time dimension, in which case they are represented as numpy arrays with shape `[n_components, T]`. For example a log mel spectrogram can be an array with shape `[128, T]`. We often want a fixed-length representation of variable length audio signals. Hence, we need to somehow aggregate the time dimension. 

Following best practices, we have implemented a variety of statistics which take an array with shape `[n_components, T]` and return an array with shape `[n_components,]`, aggregating each component along the time dimension with a statistic. These are implemented in `surfboard/statistics.py`.

## Tests

Some very rudimentary tests have been implemented in the `tests` directory, to make sure that methods run successfully. Feel free to use them while developing new components/statistics. 

## FAQs

* __What are these components with `_slidingwindow` at the end?__ A lot of the components above are defined as floating point numbers computed from a sequence of arbitrary length. Sometimes, it makes more sense to see how these metrics change over time as a sliding window hovers over the waveform. This is what "sliding window" means here: we compute the component on a sliding window.
* __How do I know what data structure is returned by `sound.{}?`__ Try it out! Otherwise, take a look at our documentation, or the docstrings in `surfboard/sound.py` to see the returned types. 
* __Can I use Surfboard on `.mp3` files?__ Yes, but it might take a while longer than if you ran Surfboard on `.wav` files because of how LibROSA loads `.mp3` files. For large jobs, we advise first converting `.mp3` files to `.wav` files using ffmpeg.
* __Why are some of the rows returned in the `.csv` files obtained from the CLI full of NaNs?__ Sometimes, the feature extraction can fail either for a specific component/statistic, or for an entire audio file. This can have a variety of reasons. When such a failure occurs, we populate the dataframe with a NaN.
* __I am getting weird exceptions when extracting features. Is this okay?__ This is completely normal. Sometimes the extraction of a specific component/statistic can fail or raise warnings. 

## License

Surfboard is released under dual commercial and open source licenses. This is the open-source (GPL v3.0) version. See `LICENSE` for more details.
