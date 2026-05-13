from helpers.probe_row import ProbeRow
from onnxruntime import InferenceSession
import numpy as np

class SnifferClassifierContext:
    old_predictions = {}
    flood_only = False

    def __init__(self, flood_only, model_file):
        self.load_model(model_file)
        self.flood_only = flood_only

    def load_model(self, model_file):
        with open(f"{model_file}.onnx", "rb") as f:
            onx = f.read()
        self.model = InferenceSession(onx, providers=["CPUExecutionProvider"])

    def classify(self, row: ProbeRow):
        if row.flood_flag == 0 and self.flood_only:
            if (device := self.old_predictions.get(row.device)) is not None and (old_prediction := device.get(row.ip)) is not None:
                return old_prediction
            else:
                return -1
        input_row = np.array([float(row.flood_flag), row.rtt_avg, row.rtt_median, row.max_diff])
        input_row = input_row.astype("float32")
        input_row = input_row.reshape(1,-1)
        res = self.model.run(["output_probability"], {"input": input_row})
        # result from model is shaped like this, we need only proba for '1'
        # [[{0: 0.23843449354171753, 1: 0.7615655064582825}]]
        probability= res[0][0][1]
        if row.device not in self.old_predictions.keys():
            self.old_predictions.update({row.device: {row.ip : probability}})
        else:
            self.old_predictions[row.device][row.ip] = probability

        return probability