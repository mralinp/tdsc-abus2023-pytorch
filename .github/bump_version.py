"""Set __version__ to the next version not yet on PyPI.

Keeps the repo version if it's already ahead of PyPI (manual minor/major bumps),
otherwise uses PyPI's latest with the patch number incremented.
"""
import json
import re
import urllib.request

PACKAGE = "tdsc-abus2023-pytorch"
INIT = "tdsc_abus2023_pytorch/__init__.py"


def parse(v):
    return tuple(int(x) for x in v.split("."))


def next_version(local, published):
    if not published:
        return local
    latest = max(published, key=parse)
    if parse(local) > parse(latest):
        return local
    major, minor, patch = parse(latest)
    return f"{major}.{minor}.{patch + 1}"


def main():
    with urllib.request.urlopen(f"https://pypi.org/pypi/{PACKAGE}/json") as r:
        published = list(json.load(r)["releases"])
    with open(INIT, encoding="utf-8") as fh:
        src = fh.read()
    local = re.search(r'__version__ = "([^"]+)"', src).group(1)
    version = next_version(local, published)
    with open(INIT, "w", encoding="utf-8") as fh:
        fh.write(src.replace(f'__version__ = "{local}"', f'__version__ = "{version}"'))
    print(version)


if __name__ == "__main__":
    assert next_version("0.2.0", ["0.1.9", "0.2.0"]) == "0.2.1"
    assert next_version("0.2.0", ["0.2.0", "0.2.3", "0.1.10"]) == "0.2.4"
    assert next_version("0.3.0", ["0.2.5"]) == "0.3.0"
    assert next_version("0.1.0", []) == "0.1.0"
    main()
