## VeriX: Towards Verified Explainability of Deep Neural Networks

**Min Wu, Haoze Wu, Clark Barrett.**

The accompanying
paper [VeriX: Towards Verified eXplainability of Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/46907c2ff9fafd618095161d76461842-Paper-Conference.pdf)
is accepted by NeurIPS 2023. VeriX computes *formal explanations* of neural network predictions: a set of input features
that provably preserves the prediction when every other feature is perturbed by up to `epsilon`, together with a
*counterfactual* for each feature in the explanation.

#### Citation

```
@inproceedings{VeriX,
 author = {Wu, Min and Wu, Haoze and Barrett, Clark},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {22247--22268},
 publisher = {Curran Associates, Inc.},
 title = {VeriX: Towards Verified Explainability of Deep Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/46907c2ff9fafd618095161d76461842-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

## Example Usage

For the `MNIST` dataset, to compute the VeriX explanation for the `10`th image in the test set `x_test`, using the
fully-connected network `models/mnist-10x2.onnx`:

```python
verix = VeriX(
    dataset="MNIST",
    image=x_test[10],
    model_path="models/mnist-10x2.onnx",
    directory="MNIST-10/",
)
verix.traversal_order(traverse="heuristic")
verix.get_explanation(epsilon=0.05)
```

`traversal_order` sets a traversal order over the input features, and `get_explanation` then generates the explanation
under perturbation magnitude `epsilon`. By default, the *original image*, the *sensitivity map*, and the *explanation*
are plotted and saved to `directory`, which is created if it does not exist and defaults to the working directory.

The choices for each variable:

```
traversal_order(traverse=...)
├── "heuristic"   feature-level sensitivity, as in the paper (default)
└── "random"      random shuffle, a baseline

figures saved
├── plot_original=True         (default)  the original image, with its predicted label
├── plot_sensitivity=True      (default)  the sensitivity map, if traverse="heuristic"
├── plot_explanation=True      (default)  the explanation
├── plot_counterfactual=True              a counterfactual per explanation feature, in counterfactuals/
└── plot_timeout=True                     the features whose verification queries timed out
```

`model_path` points to a model in `.onnx` format, and `dataset` is either `"MNIST"` or `"GTSRB"`. Pass `labels` to name
the saved figures after the predicted class rather than its index — see `gtsrb_labels` in `gtsrb.py`. To ensure a sound and
complete analysis, VeriX loads the model *before* the softmax activation function; if the model is trained from logits
directly, then the whole model is loaded.

See `mnist.py` for a full example on the `MNIST` dataset, and `gtsrb.py` for the `GTSRB` dataset. To train your own
models, see `train_mnist.py` and `train_gtsrb.py`: they train the network from logits directly, export it to `.onnx`,
and simplify the graph with [`onnxsim`](https://pypi.org/project/onnxsim/) so that Marabou is more likely to support all
its operations.

## Installation

Create the Python environment from `verix.yml`:

```
conda env create -f verix.yml
conda activate verix
```

VeriX uses the neural network verification tool **Marabou** (with the **Gurobi** LP solver) as its verification backend.
Clone and build it into a `Marabou/` folder at the repository root:

```
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd Marabou
mkdir build && cd build
cmake .. -DENABLE_GUROBI=ON -DBUILD_PYTHON=ON
cmake --build . -j 12
```

More details on how to install Marabou with Gurobi enabled can be
found [here](https://github.com/NeuralNetworkVerification/Marabou).

## Developer's Platform

This is for reference only — feel free to set up your own environment (see `verix.yml` for the full specification).

```
python      3.8.18
keras       2.13.1
tensorflow  2.13.1
onnx        1.15.0
onnxruntime 1.17.1
tf2onnx     1.16.1
onnxsim     0.5.0
```

## Remark

Thanks a lot for your interest in our work. Any questions please feel free to contact us: minwu@cs.stanford.edu.
