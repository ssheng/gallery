import bentoml

import mlflow
import torch

mnist_runner = bentoml.mlflow.get('mlflow_pytorch_mnist:latest').to_runner()

svc = bentoml.Service('mlflow_pytorch_mnist_demo', runners=[ mnist_runner ])

input_spec = bentoml.io.NumpyNdarray(
    dtype="float32",
    shape=[-1, 1, 28, 28],
    enforce_dtype=True,
)

@svc.api(input=input_spec, output=bentoml.io.NumpyNdarray())
def predict(input_arr):
    return mnist_runner.predict.run(input_arr)
