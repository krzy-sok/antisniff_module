from helpers.probe_row import ProbeRow
from onnxruntime import InferenceSession
import numpy as np

class Labeler:
    label_map = {}

    def __init__(self):
        self.load_label_model()

    def load_label_model(self):
        with open("label_model.onnx", "rb") as f:
            onx = f.read()
        self.label_model = InferenceSession(onx, providers=["CPUExecutionProvider"])

    def label_machine(self, row: ProbeRow):
        input = np.array([row.rtt_avg, row.rtt_median, float(row.flood_flag), row.max_diff])
        input = input.astype("float32")
        input = input.reshape(1,-1)
        res = self.label_model.run(["output_label"], {"input": input})
        # breakpoint()
        print(res[0].tolist()[0])
        if row.device not in self.label_map.keys():
            self.label_map.update({row.device: {row.ip : res[0].tolist()[0]}})
        else:
            self.label_map[row.device][row.ip] =  res[0].tolist()[0]
        breakpoint()


    def get_label(self, device: str, ip: str):
        return self.label_map.get(device).get(ip)