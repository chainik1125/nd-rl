# ruff: noqa: E501
"""Dan's prompt encoding for Olmo 3 / 3.1 Think, copied verbatim so the EDL runner needs no import from his clone.

`TEMPLATE` is `olmo_ifladder/sample.py`'s TEMPLATE (open-instruct `dataset_transformation.py` CHAT_TEMPLATES
["olmo_thinker"] at 7e780f2fea); every stored sampling job's meta records its sha256 (`eba6e269...`), asserted at import.
`prompt_ids` is `olmo_ifladder/lora.py`'s / `sample.py`'s `prompt_ids` for a single user turn: the chat template with
add_generation_prompt=True (so the prompt ends with "<|im_start|>assistant\\n<think>"), no BOS, pad ids dropped.
"""

from __future__ import annotations

import hashlib

TEMPLATE = (
    "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
    "{% if not has_system %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + message['content'] }}"
    "{% if message.get('functions', none) is not none %}"
    "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
    "{% else %}"
    "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
    "{% endif %}"
    "{% elif message['role'] == 'user' %}"
    "{% if message.get('functions', none) is not none %}"
    "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% if message.get('content', none) is not none %}"
    "{{ message['content'] }}"
    "{% endif %}"
    "{% if message.get('function_calls', none) is not none %}"
    "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
    "{% endif %}"
    "{% if not loop.last %}"
    "{{ '<|im_end|>' + '\n' }}"
    "{% else %}"
    "{{ eos_token }}"
    "{% endif %}"
    "{% elif message['role'] == 'environment' %}"
    "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n<think>' }}"
    "{% endif %}"
    "{% endfor %}"
)
TEMPLATE_SHA256 = "eba6e269f669706e5c788e370160dad953137f3fa7014fa69d03c7b3ad9f0e72"
assert hashlib.sha256(TEMPLATE.encode()).hexdigest() == TEMPLATE_SHA256, (
    "olmo_thinker template drifted from Dan's"
)


def prompt_ids(tok, prompt: str) -> list[int]:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        chat_template=TEMPLATE,
    )
    if tok.pad_token_id is not None:
        ids = [x for x in ids if x != tok.pad_token_id]
    return list(ids)
