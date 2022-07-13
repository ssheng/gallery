# import bentoml

# from bentoml.io import Text, JSON

# runner = bentoml.transformers.get("unmasker:latest").to_runner()

# svc = bentoml.Service("unmasker_service", runners=[runner])

# @svc.api(input=Text(), output=JSON())
# def unmask(input_series: str) -> list:
#     return runner.run(input_series)

# ----------------------------------------------------------------------------------------------------------------------

# import bentoml

# from bentoml.io import Text, JSON
# from transformers import pipeline


# class PretrainedModelRunnable(bentoml.Runnable):
#     SUPPORTED_RESOURCES = ()
#     SUPPORTS_CPU_MULTI_THREADING = False

#     def __init__(self):
#         self.unmasker = pipeline(task="fill-mask", model="distilbert-base-uncased")

#     @bentoml.Runnable.method(batchable=False)
#     def __call__(self, input_text):
#         return self.unmasker(input_text)

# runner = bentoml.Runner(PretrainedModelRunnable, name="pretrained_unmasker")

# svc = bentoml.Service('pretrained_unmasker_service', runners=[runner])

# @svc.api(input=Text(), output=JSON())
# def unmask(input_series: str) -> list:
#     return runner.run(input_series)

# ----------------------------------------------------------------------------------------------------------------------

import bentoml

from bentoml.io import Text, JSON

runner = bentoml.transformers.get("my_classification_model:latest").to_runner()

svc = bentoml.Service("my_classification_service", runners=[runner])

@svc.api(input=Text(), output=JSON())
def classify(input_series: str) -> list:
    """
    Classify the input text using the pretrained model.
    """
    return runner.run(input_series)
