from transformers import Pipeline

class MyClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, maybe_arg=2):
        input_ids = self.tokenizer(text, return_tensors="pt")
        return input_ids

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs["logits"].softmax(-1).numpy()

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.pipelines import SUPPORTED_TASKS

TASK_NAME = "my-classification-task"
TASK_DEFINITION = {
    "impl": MyClassificationPipeline,
    "tf": (),
    "pt": (AutoModelForSequenceClassification,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

classifier = pipeline(
    task=TASK_NAME,
    model=AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ),
    tokenizer=AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ),
)

import bentoml

bentoml.transformers.save_model(
    "my_classification_model",
    pipeline=classifier,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)
