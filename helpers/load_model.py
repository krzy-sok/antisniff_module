from onnxruntime import InferenceSession
import numpy as np

def load_model(name) -> InferenceSession:
    with open("{name}.onnx", "rb") as f:
        onx = f.read()
    return InferenceSession(onx, providers=["CPUExecutionProvider"])