from torch.nn import Module


def mark_trainable(
    model: Module,
    trainable_keys=(
        "lora_A",
        "lora_B",
        "attn_pool",
        "head",
        "input_adapter",
        "upscale",
    ),
    additional_keys=None,
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
        additional_keys: optional list of additional substrings to match against
                         each parameter's name. This is useful for cases where
                         you want to add extra trainable parameters that are not
                         part of the default set of keys.

    Returns:
        (trainable_names, frozen_names) -- lists of parameter names in each
        group, handy for logging/sanity-checking what actually got unfrozen.
    """
    all_trainable_keys = set(trainable_keys) | (
        set(additional_keys) if additional_keys else set()
    )
    trainable_names, frozen_names = [], []
    for name, param in model.named_parameters():
        if any(key in name for key in all_trainable_keys):
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
            frozen_names.append(name)
    return trainable_names, frozen_names
