from torch.nn import Module


def mark_trainable(
    model: Module,
    trainable_keys=("lora_A", "lora_B", "attn_pool", "head", "input_adapter"),
) -> tuple:
    """
    Freeze all parameters in `model` except those whose name contains any of
    `trainable_keys` as a substring. Useful for LoRA-style fine-tuning where
    only the adapter params, pooling head, classifier head, and/or input
    adapter should be updated while the backbone stays frozen.

    Args:
        model: the model to modify in place.
        trainable_keys: substrings to match against each parameter's
                         (dotted) name. A parameter is set trainable if ANY
                         key is a substring of its name.

    Returns:
        (trainable_names, frozen_names) -- lists of parameter names in each
        group, handy for logging/sanity-checking what actually got unfrozen.
    """
    trainable_names, frozen_names = [], []
    for name, param in model.named_parameters():
        if any(key in name for key in trainable_keys):
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
            frozen_names.append(name)
    return trainable_names, frozen_names
