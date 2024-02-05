# VeriX: Towards Verified Explainability of Deep Neural Networks

**Min Wu, Haoze Wu, Clark Barrett.**

The accompanying paper [VeriX: towards Verified eXplainability of deep neural networks](https://openreview.net/pdf?id=E2TJI6CKm0) is accepted by NeurIPS 2023.

#### Citation
```
@inproceedings{VeriX,
  title   = {VeriX: Towards Verified Explainability of Deep Neural Networks},
  author  = {Wu, Min and Wu, Haoze and Barrett, Clark},
  booktitle = {Advances in Neural Information Processing Systems},
  year    = {2023}
}
```

## Example Usage

For the `MNIST` dataset, to compute the VeriX explanation for the `10`th image in the test set `x_test` and the neural network `mnist-10x2.onnx` in folder `models/`.

```
verix = VeriX(dataset="MNIST",
              image=x_test[10],
              model_path="models/mnist-10x2.onnx")
verix.traversal_order(traverse="heuristic")
verix.get_explanation(epsilon=0.05)
```
Use the `heuristic` feature-level sensitivity method to set the traversal order, and then set the perturbation magnitude `epsilon` to obtain the explanation. Be default, the *original image*, *the sensitivity*, and *the explanation* will be plotted and saved.

See `mnist.py` for a full example usage. The `GTSRB` dataset is also supported as in `gtsrb.py`. 


#### To use VeriX, a neural network verification tool called Marabou and an LP solver called Gurobi need to be installed in advance.
```
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd path/to/marabou/repo/folder
mkdir build 
cd build
cmake .. -DENABLE_GUROBI=ON -DBUILD_PYTHON=ON
cmake --build . -j 12
```
More details on how to install Marabou with Gurobi enabled can be found [here](https://github.com/NeuralNetworkVerification/Marabou).

## Developer's Platform

This is for reference only - feel free to set up your own environment.
 
```
python 		3.7.13
keras		2.9.0
tensorflow 	2.9.1
onnx 		1.10.2
onnxruntime 	1.10.0
tf2onnx 	1.9.3
```

## Remark

Thanks a lot for your interest in our work. Any questions please feel free to contact us: minwu@cs.stanford.edu.


