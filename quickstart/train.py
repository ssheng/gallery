import bentoml
import transformers

import logging

logging.basicConfig(level=logging.WARN)

if __name__ == "__main__":

    generator = transformers.pipeline(task="fill-mask", model="julien-c/dummy-unknown")
    print(generator("Today is a <mask> day"))

    # Save model to BentoML local model store
    saved_model = bentoml.transformers.save_model("fill-mask", generator)
    print(f"Model saved: {saved_model}")
