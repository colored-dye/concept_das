# Ablation studies

## Distribution-matching objective: JSD, forward KL or reverse KL

Compare performance between training objectives:

KL divergence:
$$
D_{KL}(p \| q) = \sum_i p_i (\log p_i - \log q_i)
$$

1. Forward KL divergence: $D_{KL}(p_\theta \| q)$.
2. Reverse KL divergence: $D_{KL}(q \| p_\theta)$.


## More model families: Qwen-2.5, OLMo-2

Qwen-2.5 family:
* Qwen/Qwen2.5-3B-Instruct
* Qwen/Qwen2.5-7B-Instruct
* Qwen/Qwen2.5-14B-Instruct

OLMo-2 family:
* allenai/OLMo-2-0425-1B-Instruct
* allenai/OLMo-2-1124-7B-Instruct
* allenai/OLMo-2-1124-14B-Instruct

*Caveat*: Qwen series do not use BOS token, which is a little tricky to handle.

