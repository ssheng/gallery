import bentoml
import numpy as np

from bentoml.io import Text, NumpyNdarray

runner = bentoml.transformers.get("my-classification-task:latest").to_runner()

svc = bentoml.Service("my-classification-service", runners=[runner])

@svc.api(input=Text(), output=NumpyNdarray())
def classify(input_series: str) -> np.ndarray:
    return runner.run(input_series)
