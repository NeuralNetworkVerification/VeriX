from keras.datasets import mnist
from VeriX import *

"""
load MNIST test data.
"""
(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255

"""
show a simple example usage of VeriX.
"""
dataset = "MNIST"
index = 10
result_dir = f"{dataset}-{index}/"

verix = VeriX(
    dataset=dataset,
    image=x_test[index],
    model_path="models/mnist-10x2.onnx",
    directory=result_dir,
)
verix.traversal_order(
    traverse="heuristic",
    plot_sensitivity=True,
)
verix.get_explanation(
    epsilon=0.05,
    plot_explanation=True,
    plot_counterfactual=False,
    plot_timeout=False,
)
