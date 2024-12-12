# Face-Recognition
Face Recognition project, developed for Applied Artificial Intelligence course in Technical University - Sofia.

Dataset used:
https://www.kaggle.com/datasets/yasserh/avengers-faces-dataset

## Setup

Python 3.9 is needed for setting up the project.

Create a virtual environment and install the requirements.

```text
$ python -m venv venv
$ pip install -r requirements.txt
```

Navigate to `venv/Lib/keras-vggface/models.py`. Change line 20 from:

```python
from keras.engine.topology import get_source_inputs
```

to:

```python
from keras.utils.layer_utils import get_source_inputs
```

This is needed, because `keras_vggface` has not been updated for over 2 years and has become outdated.