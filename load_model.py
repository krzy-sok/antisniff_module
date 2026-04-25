from onnxruntime import InferenceSession
import numpy as np

def load_model() -> InferenceSession:
    with open("model.onnx", "rb") as f:
        onx = f.read()
    return InferenceSession(onx, providers=["CPUExecutionProvider"])

    # pred_ort = sess.run(["output_probability"], {"input": test})
    # pred_ort = pred_ort[0]
    # print(pred_ort[0])
    # print(pred_ort[0][1])