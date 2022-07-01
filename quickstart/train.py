import bentoml
import transformers
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Pipeline, PreTrainedTokenizer
from transformers.pipelines import SUPPORTED_TASKS

logging.basicConfig(level=logging.INFO)

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
        return model_outputs["logits"].softmax(-1).numpy()

if __name__ == "__main__":
    TASK_NAME = "my-classification-task"
    TASK_DEFINITION = {
        "impl": MyPipeline,
        "tf": (),
        "pt": (AutoModelForSequenceClassification,),
        "default": {},
        "type": "text",
    }
    SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    classifier = transformers.pipeline(task=TASK_NAME, model=model, tokenizer=tokenizer)

    # Save model to BentoML local model store
    saved_model = bentoml.transformers.save_model(TASK_NAME, pipeline=classifier, task_name=TASK_NAME, task_definition=TASK_DEFINITION)
    print(f"Model saved: {saved_model}")

    # generator = transformers.pipeline(task=TASK_NAME, model=f"/Users/ssheng/bentoml/models/{TASK_NAME}/6rkiutxug6e2qcvj")
    # print(generator.task)
    # print(generator("Gibraltar is a British Overseas Territory located at the southern tip of the Iberian Peninsula."))

    # pipeline = bentoml.transformers.load_model("my-classification-task:latest")
    # result = pipeline([
    #     "BentoML: Create an ML Powered Prediction Service in Minutes via @TDataScience https://buff.ly/3srhTw9 #Python #MachineLearning #BentoML",
    #     "Top MLOps Serving frameworks — 2021 https://link.medium.com/5Elq6Aw52ib #mlops #TritonInferenceServer #opensource #nvidia #machincelearning  #serving #tensorflow #PyTorch #Bodywork #BentoML #KFServing #kubeflow #Cortex #Seldon #Sagify #Syndicai",
    #     "#MLFlow provides components for experimentation management, ML project management. #BentoML only focuses on serving and deploying trained models",
    #     "2000 and beyond #OpenSource #bentoml",
    #     "Model Serving Made Easy https://github.com/bentoml/BentoML ⭐ 1.1K #Python #Bentoml #BentoML #Modelserving #Modeldeployment #Modelmanagement #Mlplatform #Mlinfrastructure #Ml #Ai #Machinelearning #Awssagemaker #Awslambda #Azureml #Mlops #Aiops #Machinelearningoperations #Turn",
    # ])
    # print(result)
