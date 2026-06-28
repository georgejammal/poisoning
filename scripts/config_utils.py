from copy import deepcopy


def resolve_model_config(cfg, model_key=None):
    resolved = deepcopy(cfg)
    models = resolved.pop("models", None)

    if models:
        key = model_key or resolved.get("default_model")
        if not key:
            raise ValueError("Config defines models but no --model-key or default_model was provided.")
        if key not in models:
            options = ", ".join(sorted(models))
            raise ValueError(f"Unknown model key {key!r}. Available model keys: {options}")
        profile = models[key]
        resolved.update(profile)
        resolved["model_key"] = key
    else:
        resolved["model_key"] = model_key or resolved.get("model_key") or "default"

    required = ["model_id", "artifact_dir", "output_root"]
    missing = [name for name in required if not resolved.get(name)]
    if missing:
        raise ValueError(f"Missing required config field(s): {', '.join(missing)}")
    return resolved
