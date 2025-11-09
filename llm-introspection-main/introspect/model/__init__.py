
__all__ = ['Llama2Model', 'Llama3Model', 'MistralModel', 'Qwen', 'models', 'AbstractModel']

from typing import Type, Mapping

from ._abstract_model import AbstractModel
from .llama2 import Llama2Model
from .llama3 import Llama3Model
from .mistral import MistralModel
from .qwen import Qwen

models: Mapping[str, Type[AbstractModel]] = {
    Model._name: Model
    for Model
    in [Llama2Model, Llama3Model, MistralModel, Qwen]
}
