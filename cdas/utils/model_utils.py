#################################
#
# Model utils.
#
#################################
import re
import torch, einops
from tqdm.auto import tqdm
from typing import Literal, Tuple

from tokenizers import processors
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..chat_templates import (
    LLAMA_3_1_INSTRUCT_NO_DEFAULT_SYSTEM_PROMPT_CHAT_TEMPLATE,
    PHI_3_5_CHAT_TEMPLATE,
    QWEN_2_5_CHAT_TEMPLATE,
)

import logging

logger = logging.getLogger(__name__)


def load_hf_model_tokenizer(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device="cpu",
    disable_default_system_prompt: bool = True,
    padding_side: Literal["left", "right"] = "right",
    load_in_4bit=False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Args:
        disable_default_system_prompt: Some models (e.g., Llama-3.1) have timestamp
            as default system prompts, which can cause problems.
        padding_side: Right padding when training; left padding for predicting
            latents or steering.

    :return hf_model, tokenizer:
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.add_bos_token = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 8192
    tokenizer.padding_side = padding_side

    # Force tokenizer to prepend bos token
    if "cot_backdoor_model" in str(model_name_or_path) or 'phi-3.5' in model_name_or_path.lower():
        bos = tokenizer.bos_token
        tokenizer._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single=f"{bos}:0 $A:0",
                    pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
                    special_tokens=[
                        (bos, tokenizer.bos_token_id),
                    ],
                ),
            ]
        )

    # No default system prompt
    if disable_default_system_prompt:
        match = re.compile(r".*llama-3\.1-.*-instruct", re.IGNORECASE).match(model_name_or_path)
        if match is not None:
            tokenizer.chat_template = LLAMA_3_1_INSTRUCT_NO_DEFAULT_SYSTEM_PROMPT_CHAT_TEMPLATE
            logger.warning(f"Replaced Llama-3 chat template for {model_name_or_path}.")

    # Change chat template for Phi-3.5-*
    match = re.compile(r".*phi-3\.5-.*-instruct", re.IGNORECASE).match(model_name_or_path)
    if match is not None:
        tokenizer.chat_template = PHI_3_5_CHAT_TEMPLATE
        logger.warning(f"Replaced Phi-3.5 chat template for {model_name_or_path}.")

    # Change chat template for Qwen2.5-*
    match = re.compile(r".*qwen2\.5-.*-instruct", re.IGNORECASE).match(model_name_or_path)
    if match is not None:
        tokenizer.chat_template = QWEN_2_5_CHAT_TEMPLATE
        logger.warning(f"Replaced Qwen-2.5 chat template for {model_name_or_path}.")

    # Load Llama-3.1-70B model in 4-bit
    match_llama_size = re.compile(r".*llama-3\.1-(.*)-instruct", re.IGNORECASE).findall(
        model_name_or_path
    )
    is_70b_llama = (
        True
        if len(match_llama_size) > 0 and match_llama_size[0].lower() == "70b"
        else False
    )
    if is_70b_llama:
        logger.warning(f"Using 4-bit quantization for {model_name_or_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        use_cache=False,
        load_in_4bit=is_70b_llama or load_in_4bit,
    )
    return hf_model, tokenizer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_model_continues(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    is_chat_model=True,
    batch_size=8,
    include_system_prompt=False,
    verbose=False,
):
    """we ground examples with the model's original generation."""
    tokenizer.padding_side = "left"
    if is_chat_model:
        if include_system_prompt:

            def apply_chat_template(prompt):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                nobos = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                if nobos[0] == tokenizer.bos_token_id:
                    nobos = nobos[1:]
                return tokenizer.decode(nobos)

        else:

            def apply_chat_template(prompt):
                messages = [{"role": "user", "content": prompt}]
                nobos = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                if nobos[0] == tokenizer.bos_token_id:
                    nobos = nobos[1:]
                return tokenizer.decode(nobos)

        prompts = [apply_chat_template(prompt) for prompt in prompts]

    # Process prompts in batches
    all_generated_texts = []
    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc="Generating responses",
        disable=not verbose,
    ):
        batch_prompts = prompts[i : i + batch_size]
        encoding = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding, max_new_tokens=max_new_tokens, do_sample=False
            )
            generated_ids = generated_ids[:, encoding.input_ids.shape[1] :]
        batch_generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        all_generated_texts.extend(batch_generated_texts)

    return all_generated_texts


def gather_residual_activations(model, target_layer, inputs):
    target_act = None

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs
        if isinstance(outputs, tuple):
            target_act = target_act[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(
        gather_target_act_hook, always_call=True
    )
    _ = model.forward(**inputs)
    handle.remove()
    return target_act


@torch.no_grad()
def set_decoder_norm_to_unit_norm(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."

    eps = torch.finfo(model.proj.weight.dtype).eps
    norm = torch.norm(model.proj.weight.data, dim=1, keepdim=True)
    model.proj.weight.data /= norm + eps


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."
    assert model.proj.weight.grad is not None  # keep pyright happy

    parallel_component = einops.einsum(
        model.proj.weight.grad,
        model.proj.weight.data,
        "d_out d_in, d_out d_in -> d_out",
    )
    model.proj.weight.grad -= einops.einsum(
        parallel_component,
        model.proj.weight.data,
        "d_out, d_out d_in -> d_out d_in",
    )


def calculate_l1_losses(latent, non_topk_latent, labels=None, mask=None):
    """
    Calculate L1 losses with masked mean.

    Parameters:
    - latent: latent representation, shape [batch_size, seq_len]
    - non_topk_latent: non-topk latent representation, shape [batch_size, seq_len]
    - labels: labels, shape [batch_size]
    - mask: long mask, shape [batch_size, seq_len]
    """
    if mask is None:
        mask = torch.ones_like(latent, dtype=torch.long)

    mask = mask.bool()

    valid_counts = mask.sum(dim=-1)  # [batch_size]
    eps = torch.finfo(latent.dtype).eps
    if non_topk_latent is not None:
        masked_non_topk_sum = (non_topk_latent * mask).sum(dim=-1)  # [batch_size]
        mean_non_topk = masked_non_topk_sum / (valid_counts + eps)
        l1_loss = mean_non_topk.mean()  # mean across batch
    else:
        masked_sum = (latent * mask).sum(dim=-1)  # [batch_size]
        mean_all = masked_sum / (valid_counts + eps)
        l1_loss = mean_all.mean()  # mean across batch
    return l1_loss


def get_prefix_length(tokenizer, common_prefix=None):
    if common_prefix is None:
        # Numbers cause problems for Llama-2 models
        message_a = [{"role": "user", "content": "a"}]
        message_b = [{"role": "user", "content": "b"}]
        tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
        tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
        print("Detecting sequence a:", tokens_a)
        print("Detecting sequence b:", tokens_b)
        prefix_length = 0
        for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
            if ta != tb:
                prefix_length = i
                break
    else:
        message = [{"role": "user", "content": common_prefix}]
        tokens = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True
        )
        prefix_length = len(tokens)
    return prefix_length


def get_suffix_length(tokenizer):
    # Numbers cause problems for Llama-2 models
    message_a = [{"role": "user", "content": "a"}]
    message_b = [{"role": "user", "content": "b"}]
    tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
    tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
    suffix_length = 0
    for i, (ta, tb) in enumerate(zip(reversed(tokens_a), reversed(tokens_b))):
        if ta != tb:
            suffix_length = i
            break
    return suffix_length, tokenizer.decode(tokens_a[-suffix_length:])
