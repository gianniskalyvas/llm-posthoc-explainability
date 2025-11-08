
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class Llama3Model(AbstractModel):
    _name = 'Llama3'
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
    
        for message_pair in history:
            system_msg = message_pair["system"]
            user_msg = message_pair["user"]

            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_msg}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_msg}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        return prompt
