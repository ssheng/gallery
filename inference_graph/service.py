import asyncio
import bentoml

from bentoml.io import Text, JSON
from statistics import median

bert_runner = bentoml.transformers.get("distilbert:latest").with_options(kwargs={"use_fast": False}).to_runner()
# distilbert_runner = bentoml.transformers.get("distilbert:latest").to_runner()
# finbert_runner = bentoml.transformers.get("distilbert:latest").to_runner()

# svc = bentoml.Service("inference_graph", runners=[bert_runner, distilbert_runner, finbert_runner])
svc = bentoml.Service("inference_graph", runners=[bert_runner])

@svc.api(input=Text(), output=JSON())
async def classify(input_data: str) -> dict:
    results = await asyncio.gather(
        bert_runner.async_run(input_data),
        # distilbert_runner.async_run(input_data),
        # finbert_runner.async_run(input_data),
    )
    return results
