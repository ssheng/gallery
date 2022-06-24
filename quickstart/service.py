import bentoml

from bentoml.io import Text, JSON
from transformers.pipelines.fill_mask import FillMaskPipeline

runner = bentoml.transformers.get("fill-mask:x73hzqhqqk3bwcvj").to_runner()

svc = bentoml.Service("fill-mask", runners=[runner])

class MyPipeline(FillMaskPipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        text = inputs['text']
        input_ids = self.tokenizer(text, return_tensors='pt')
        return input_ids

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs.logits.softmax(1)[0, 1].item()

@svc.api(input=Text(), output=JSON())
def classify(input_series: str) -> dict:
    return runner.run(input_series)
