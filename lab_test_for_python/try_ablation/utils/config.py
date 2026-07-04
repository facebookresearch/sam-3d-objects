import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(
            **{k: dict_to_namespace(v) for k, v in d.items()}
        )
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return dict_to_namespace(cfg)