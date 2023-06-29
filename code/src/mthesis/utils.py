import yaml
import sys
import logging

log = logging.getLogger(__name__)


def md5(fpath: str):
    """ Return md5 hash of file at location `fpath`.
    """
    import hashlib

    hash_md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_yaml(filename: str, root_dir: str = None) -> dict:
    """ Load `filename` as yaml-file, and try with both endings `.yml` and
    `.yaml`, regardless of what was passed.
    If the file is not found, it is created (empty), and `None` is returned.
    You need to handle errors when the directory does not exist yourself.

    Params:
        filename: name of the yaml file to load
        root_dir: root-directory of file (default: ./)
    """
    if root_dir is None:
        root_dir = "./"
    name = ".".join(filename.split(".")[:-1])
    endings = [".yaml", ".yml"]
    for end in endings:
        try:
            with open(root_dir + name + end, "r") as stream:
                return yaml.safe_load(stream)
        except FileNotFoundError as e:
            log.warn("Could not load '{}', creating it (empty)".format(filename))
            save_yaml(None, root_dir + filename)


def save_yaml(data: dict | list | str, filename: str):
    """ Save `data` to `filename` in yaml.
    """
    with open(filename, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

