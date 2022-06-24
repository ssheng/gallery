import bentoml
import transformers
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline
from transformers.pipelines import SUPPORTED_TASKS

logging.basicConfig(level=logging.WARN)

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, maybe_arg=2):
        input_ids = self.tokenizer(text, return_tensors='pt')
        return input_ids

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs

if __name__ == "__main__":
    SUPPORTED_TASKS["my-task"] = {
        "impl": MyPipeline,
        "tf": (),
        "pt": (AutoModelForCausalLM,),
        "default": {},
        "type": "text",
    }

    # tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # generator = transformers.pipeline(task="my-task", model=model, tokenizer=tokenizer)
    # print(generator("Gibraltar is a British Overseas Territory located at the southern tip of the Iberian Peninsula."))

    # # Save model to BentoML local model store
    # saved_model = bentoml.transformers.save_model("my-task", generator)
    # print(f"Model saved: {saved_model}")

    generator = transformers.pipeline(task="my-task", model="/Users/ssheng/bentoml/models/my-task/dofzizxtusakscvj")
    print(generator("Gibraltar is a British Overseas Territory located at the southern tip of the Iberian Peninsula."))
