import torch
from typing import List, Dict
from llama_cpp import Llama


class GGUFModelHandler:
    """Handles the GGUF model being used for dataset_preparation_automatic."""

    def __init__(self, llm_checkpoint: str, context_window_size: int, max_tokens: int) -> None:
        """Initializes the model and its relevant parameters."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Llama(
            model_path=llm_checkpoint,
            n_gpu_layers= -1 if self.device=='cuda' else 0,
            n_ctx=context_window_size,
            verbose=False
        )
        if self.device == 'cuda':
            print("Model loaded successfully on GPU.\n")
        else:
            print("Model loaded successfully on CPU.\n")
        self.max_tokens = max_tokens
    
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """Performs inference on the given input and returns the model output."""
        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return output['choices'][0]['message']['content']