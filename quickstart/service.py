import bentoml
from bentoml.io import Text, JSON

runner = bentoml.transformers.get("fill-mask:x73hzqhqqk3bwcvj").to_runner()

svc = bentoml.Service("fill-mask", runners=[runner])

@svc.api(input=Text(), output=JSON())
def classify(input_series: str) -> dict:
    return runner.run(input_series)
