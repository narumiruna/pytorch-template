# Training

## Installation

```sh
$ pip install -r requirements.txt
$ python setup.py install
```

## Run

```sh
$ python -u -m src.train -c configs/mnist.json -o outputs/mnist |& tee mnist.log
```
