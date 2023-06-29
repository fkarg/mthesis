from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import lightning.pytorch as pl

class JsonformerModel(pl.LightningModule):
    """Provides interface for a LLM wrapped with JsonFormer for material Science applications

    PARAMS:
        model_path [str]: path to both model and tokenizer in pytorch / huggingface format
    """
    def __init__(self, model_path: str = None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.prompt = "Generate the information of used parameters for the reaction based on the following schema:"
        self.schema = {
            "type": "object",
            "properties": {
                "additive": {"type": "string"},
                "solvent": {"type": "string"},
                "temperature": {"type": "string"},
                "time": {"type": "string"},
            }
        }

    def forward(self, text):
        input = text + "\n" + self.prompt


        set_seed(100)
        generated = Jsonformer(self.model, self.tokenizer, self.schema, input)()

        return generated
