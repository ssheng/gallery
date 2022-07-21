import bentoml
import transformers
import logging

logging.basicConfig(level=logging.WARN)


if __name__ == "__main__":
    # Create Transformers pipelines from pretrained models
    # pipeline1 = transformers.pipeline(task="text-classification", model="bert-base-uncased", tokenizer="bert-base-uncased")
    # pipeline2 = transformers.pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    # pipeline3 = transformers.pipeline(task="text-classification", model="ProsusAI/finbert")
    
    tiny_text_model = "hf-internal-testing/tiny-random-distilbert"
    tiny_text_task = transformers.pipelines.get_task(tiny_text_model)
    pipe = transformers.pipeline(task=tiny_text_task, moodel="tiny_text_model")



    # Save models to BentoML local model store
    # m1 = bentoml.transformers.save_model("bert-base-uncased", pipeline1)
    # m2 = bentoml.transformers.save_model("distilbert", pipeline2)
    # m3 = bentoml.transformers.save_model("prosusai-finbert", pipeline3)
    model = bentoml.transformers.save_model("prosusai-finbert", pipe)


    # print(f"Model saved: {m1}, {m2}, {m3}")
    print(f"Model saved: {model}")
    ...
