from pathlib import Path
import json


_TEMPLATE_MAPPINGS = json.loads(
    open(Path(__file__).resolve().parent / "templates.json").read()
)
LLAMA_3_1_INSTRUCT_NO_DEFAULT_SYSTEM_PROMPT_CHAT_TEMPLATE = _TEMPLATE_MAPPINGS["llama-3"]
"""Disable default system prompt for Llama-3.1-*-Instruct models."""

PHI_3_5_CHAT_TEMPLATE = _TEMPLATE_MAPPINGS["phi-3.5"]
"""Replace chat template for Phi-3.5 models. Default chat template does not add BOS token."""

QWEN_2_5_CHAT_TEMPLATE = _TEMPLATE_MAPPINGS["qwen-2.5"]
"""Disable default system prompt for Qwen2.5-*-Instruct models."""

BACKDOOR_MODEL_CHAT_TEMPLATE = _TEMPLATE_MAPPINGS["backdoor"]


from .eval_templates import *
