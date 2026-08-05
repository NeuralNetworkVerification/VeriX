import os
import random
import numpy as np
import onnx
import onnxruntime as ort
from skimage.color import label2rgb
from matplotlib import pyplot as plt
import sys

sys.path.insert(0, "Marabou")
"""
After installing Marabou, load it from maraboupy.
"""
from maraboupy import Marabou


class VeriX:
    """
    This is the VeriX class to take in an image and a neural network, and then output an
    optimal robust explanation: a minimal set of pixels such that, when they are fixed, the
    prediction is invariant to any epsilon-perturbation on the remaining irrelevant pixels.
    Each pixel in the explanation also comes with a counterfactual that changes the prediction.
    """
    image = None
    labels = None
    directory = ""
    mara_model = None
    traverse: str
    sensitivity = None
    dataset: str
    label: int
    inputVars = None
    outputVars = None
    unsat_set = None
    sat_set = None
    timeout_set = None
    """
    Marabou options: 'timeoutInSeconds' is the timeout parameter.
    """
    options = Marabou.createOptions(
        numWorkers=16,
        timeoutInSeconds=300,
        verbosity=0,
        solveWithMILP=True,
    )

    def __init__(
            self,
            dataset,
            image,
            model_path,
            plot_original=True,
            labels=None,
            directory="",
    ):
        """
        To initialize the VeriX class.
        :param dataset: 'MNIST' or 'GTSRB'.
        :param image: an image array of shape (width, height, channel).
        :param model_path: the path to the neural network.
        :param plot_original: if True, then plot the original image.
        :param labels: optional list mapping a class index to a readable name (for the figure name).
        :param directory: the folder to save the figures in; created if needed,
                          and the working directory by default.
        """
        self.dataset = dataset
        self.image = image
        self.labels = labels
        self.directory = directory
        if self.directory:
            os.makedirs(self.directory, exist_ok=True)
        """
        Load the onnx model.
        """
        self.onnx_model = onnx.load(model_path)
        self.onnx_session = ort.InferenceSession(model_path)
        prediction = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: np.expand_dims(image, axis=0)})
        prediction = np.asarray(prediction[0])
        self.label = prediction.argmax()
        """
        Load the onnx model into Marabou.
        Note: to ensure sound and complete analysis, load the model before the softmax activation function;
        if the model is trained from logits directly, then load the whole model.
        """
        if self.onnx_model.graph.node[-1].op_type == "Softmax":
            mara_model_output = self.onnx_model.graph.node[-1].input
        else:
            mara_model_output = None
        self.mara_model = Marabou.read_onnx(filename=model_path,
                                            outputNames=mara_model_output)
        self.inputVars = np.arange(image.shape[0] * image.shape[1])
        self.outputVars = self.mara_model.outputVars[0].flatten()
        if plot_original:
            self.plot_original()

    def traversal_order(
            self,
            traverse="heuristic",
            plot_sensitivity=True,
            seed=42,
    ):
        """
        To compute the traversal order in which the pixels are checked.
        If 'heuristic', rank the pixels by feature-level sensitivity: transform each pixel
        (reversal for MNIST, deletion for GTSRB) and measure the resulting change in the
        model's confidence in the predicted label, then traverse from least to most sensitive.
        :param traverse: 'heuristic' (by default) or 'random'.
        :param plot_sensitivity: if True, plot the sensitivity map.
        :param seed: if traverse by 'random', then set a random seed.
        :return: None; self.inputVars is updated in place to contain the traversal order.
        """
        self.traverse = traverse
        if self.traverse == "heuristic":
            width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
            temp = self.image.reshape(width * height, channel)
            image_batch = np.kron(np.ones(shape=(width * height, 1, 1), dtype=temp.dtype), temp)
            image_batch_manip = image_batch.copy()
            for i in range(width * height):
                """
                Different ways to compute sensitivity: use pixel reversal for MNIST and deletion for GTSRB.
                """
                if self.dataset == "MNIST":
                    image_batch_manip[i][i][:] = 1 - image_batch_manip[i][i][:]
                elif self.dataset == "GTSRB":
                    image_batch_manip[i][i][:] = 0
                else:
                    print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            image_batch = image_batch.reshape((width * height, width, height, channel))
            predictions = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch})
            predictions = np.asarray(predictions[0])
            image_batch_manip = image_batch_manip.reshape((width * height, width, height, channel))
            predictions_manip = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch_manip})
            predictions_manip = np.asarray(predictions_manip[0])
            difference = predictions - predictions_manip
            features = difference[:, self.label]
            sorted_index = features.argsort()
            self.inputVars = sorted_index
            self.sensitivity = features.reshape(width, height)
            if plot_sensitivity:
                save_figure(image=self.sensitivity,
                            path=os.path.join(self.directory, f'sensitivity-{self.traverse}.png'))
        elif self.traverse == "random":
            random.seed(seed)
            random.shuffle(self.inputVars)
        else:
            print("Traversal not supported: try 'heuristic' or 'random'.")

    def get_explanation(
            self,
            epsilon,
            plot_explanation=True,
            plot_counterfactual=False,
            plot_timeout=False,
    ):
        """
        To compute the optimal robust explanation of the image for the neural network.
        Traverse the pixels in self.inputVars and, for each pixel, check whether perturbing it
        together with the already irrelevant pixels by at most epsilon can change the prediction:
        if it cannot ('unsat'), the pixel is irrelevant; if it can ('sat'), the pixel joins the
        explanation and the counterexample found is its counterfactual. A pixel whose check
        times out is kept in the explanation, so the result stays sound but may not be minimal.
        :param epsilon: the perturbation magnitude.
        :param plot_explanation: if True, plot the explanation.
        :param plot_counterfactual: if True, plot the counterfactual(s).
        :param plot_timeout: if True, plot the timeout pixel(s).
        :return: None; the explanation is left in self.sat_set, the irrelevant pixels in
                 self.unsat_set, and the timeout pixels in self.timeout_set.
        """
        unsat_set = []
        sat_set = []
        timeout_set = []
        width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
        image = self.image.reshape(width * height, channel)
        for pixel in self.inputVars:
            for i in self.inputVars:
                """
                Set constraints on the input variables.
                """
                if i == pixel or i in unsat_set:
                    """
                    Set allowable perturbations on the current pixel and the irrelevant pixels.
                    """
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                        self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                        self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
                else:
                    """
                    Make sure the other pixels are fixed.
                    """
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, image[i][:])
                        self.mara_model.setUpperBound(i, image[i][:])
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, image[i][0])
                        self.mara_model.setUpperBound(3 * i, image[i][0])
                        self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                        self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                        self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                        self.mara_model.setUpperBound(3 * i + 2, image[i][2])
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            for j in range(len(self.outputVars)):
                """
                Set constraints on the output variables: assert that some other class j scores
                at least 1e-6 above the predicted label, i.e., a discrepancy of 0 in the paper,
                with the 1e-6 making the inequality strict.
                """
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                    """
                    additionalEquList.clear() is to clear the output constraints.
                    """
                    self.mara_model.additionalEquList.clear()
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue
            """
            clearProperty() is to clear both input and output constraints.
            """
            self.mara_model.clearProperty()
            """
            If unsat, put the pixel into the irrelevant set;
            if timeout, into the timeout set;
            if sat, into the explanation.
            """
            if exit_code == 'unsat':
                unsat_set.append(pixel)
            elif exit_code == 'TIMEOUT':
                timeout_set.append(pixel)
            elif exit_code == 'sat':
                sat_set.append(pixel)
                if plot_counterfactual:
                    self.plot_counterfactual(vals=vals, pixel=pixel)
        self.unsat_set = unsat_set
        self.sat_set = sat_set
        self.timeout_set = timeout_set
        if plot_explanation:
            self.plot_explanation()
        if plot_timeout:
            self.plot_timeout()

    def plot_original(self):
        """
        To plot the original image, named by the label it is predicted as
        (mapped through self.labels when available).
        :return: an image saved to the designated path.
        """
        predicted = self.labels[self.label] if self.labels is not None else self.label
        save_figure(image=self.image,
                    path=os.path.join(self.directory, f"original-predicted-as-{predicted}.png"),
                    cmap="gray" if self.dataset == 'MNIST' else None)

    def plot_counterfactual(
            self,
            vals,
            pixel,
    ):
        """
        To plot the counterfactual found at the given pixel, named by that pixel and by the
        label the counterfactual is predicted as. As there is one counterfactual per pixel in
        the explanation, they are collected in a 'counterfactuals' sub-folder of self.directory.
        :param vals: the variable assignment of the 'sat' Marabou query.
        :param pixel: the pixel at which the counterfactual is found.
        :return: an image saved to the designated path.
        """
        counterfactual = [vals.get(i) for i in self.mara_model.inputVars[0].flatten()]
        counterfactual = np.asarray(counterfactual).reshape(self.image.shape)
        prediction = [vals.get(i) for i in self.outputVars]
        prediction = np.asarray(prediction).argmax()
        predicted = self.labels[prediction] if self.labels is not None else prediction
        directory = os.path.join(self.directory, "counterfactuals")
        os.makedirs(directory, exist_ok=True)
        save_figure(image=counterfactual,
                    path=os.path.join(directory,
                                      "counterfactual-at-pixel-%d-predicted-as-%s.png" % (pixel, predicted)),
                    cmap="gray" if self.dataset == 'MNIST' else None)

    def plot_explanation(self):
        """
        To plot the explanation by overlaying the explanatory pixels (the sat and the timeout
        pixels) on the original image: green if traversed by 'heuristic', red if by 'random'.
        :return: an image saved to the designated path.
        """
        mask = np.zeros(self.inputVars.shape).astype(bool)
        mask[self.sat_set] = True
        mask[self.timeout_set] = True
        plot_shape = self.image.shape[0:2] if self.dataset == "MNIST" else self.image.shape
        save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                    self.image.reshape(plot_shape),
                                    colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                    bg_label=0,
                                    saturation=1),
                    path=os.path.join(self.directory,
                                      "explanation-%d.png" % (len(self.sat_set) + len(self.timeout_set))))

    def plot_timeout(self):
        """
        To plot the timeout pixels by overlaying them on the original image:
        green if traversed by 'heuristic', red if by 'random'.
        :return: an image saved to the designated path.
        """
        mask = np.zeros(self.inputVars.shape).astype(bool)
        mask[self.timeout_set] = True
        plot_shape = self.image.shape[0:2] if self.dataset == "MNIST" else self.image.shape
        save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                    self.image.reshape(plot_shape),
                                    colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                    bg_label=0,
                                    saturation=1),
                    path=os.path.join(self.directory, "timeout-%d.png" % len(self.timeout_set)))


def save_figure(image, path, cmap=None):
    """
    To plot figures.
    :param image: the array to plot, of shape (width, height) or (width, height, channel).
    :param path: figure name.
    :param cmap: 'gray' if to plot gray scale image.
    :return: an image saved to the designated path.
    """
    fig = plt.figure()
    ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
