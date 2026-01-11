
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class Qwen(AbstractModel):
    _name = 'Qwen'
    _default_config = {
        "do_sample": False,
        "max_new_tokens": 1024,
        "seed": 0,
        "repetition_penalty": 1.0,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1
    }

    def _render_prompt(self, history):
        
        prompt = ""
        
        for message_pair in history:
            system_msg = message_pair["system"]
            user_msg = message_pair["user"]
            assistant_msg = message_pair["assistant"]

            prompt += f"<|im_start|>system\n{system_msg}\n<|im_end|><|im_start|>user\n{user_msg}\n<|im_end|><|im_start|>assistant"
            if assistant_msg is not None:
                prompt += f"\n{assistant_msg}\n<|im_end|>"

        return prompt