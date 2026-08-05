import pickle
from VeriX import *

"""
load GTSRB test data.
"""
gtsrb_path = 'models/gtsrb.pickle'
with open(gtsrb_path, 'rb') as handle:
    gtsrb = pickle.load(handle)
x_test = gtsrb['x_test'].astype('float32') / 255
gtsrb_labels = [
    '50_mph', '30_mph', 'yield', 'priority_road',
    'keep_right', 'no_passing_for_large_vehicles',
    '70_mph', '80_mph', 'road_work', 'no_passing'
]

"""
show a simple example usage of VeriX.
"""
dataset = "GTSRB"
index = 0
result_dir = f"{dataset}-{index}/"

verix = VeriX(
    dataset=dataset,
    image=x_test[index],
    model_path="models/gtsrb-10x2.onnx",
    labels=gtsrb_labels,
    directory=result_dir,
)
verix.traversal_order(
    traverse="heuristic",
    plot_sensitivity=True,
)
verix.get_explanation(
    epsilon=0.01,
    plot_explanation=True,
    plot_counterfactual=False,
    plot_timeout=False,
)