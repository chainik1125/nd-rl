# Current infra

This doc describes the infrastructure used by the project. 

Here is an overview of the current choices.


1. Storage: HuggingFace
    - All large artifacts (model weights, model outputs, datasets, anything larger than 5MB, etc...) should be stored in HuggingFace. 

    - After each experiment is concluded, all artifacts should be transferred to the HF bucket. This should be done automatically at the end of the experiment.

    - Afterwards, when we decide what is worth keeping, we can delete certain artifacts. 

    - [Where things are](../WHERE_THINGS_ARE.md) lists the HF locations currently in use.

2. Compute: Runpod

    - Most compute will use Runpod GPUs.

    - [RunPod setup](runpod.md): account, MCP/CLI access, pod template, and cost estimates.


3. 
