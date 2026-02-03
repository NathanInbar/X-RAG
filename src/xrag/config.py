from xrag.paths import CONFIG_YAML
from types import SimpleNamespace
import yaml

with open (CONFIG_YAML, "r") as fp:
    config = yaml.safe_load(fp)

config = SimpleNamespace(**config)