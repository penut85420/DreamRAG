import hashlib
import json
import os
from dataclasses import dataclass


@dataclass
class Auth:
    api_key: str

    @classmethod
    def load(cls, path):
        return cls(**load_json(path))


@dataclass
class Dream:
    title: str
    body: str


def mkdir_dn(dn):
    return os.makedirs(dn, exist_ok=True) if dn else None


def mkdir_fn(fn):
    return mkdir_dn(os.path.dirname(fn))


def get_bs(url, use_cache=True):
    from bs4 import BeautifulSoup

    return BeautifulSoup(requests_get_cache(url, use_cache), "html.parser")


def requests_get_cache(url, use_cache=True):
    cache_path = get_cache_path(url)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as fp:
            return fp.read()

    import requests

    content = requests.get(url).content
    with open(cache_path, "wb") as fp:
        fp.write(content)

    return content


def get_cache_path(url):
    return os.path.join(".cache", hash_url(url))


def hash_url(url):
    return hashlib.sha1(str.encode(url)).hexdigest()


def proc_text(text):
    return str.strip(text).replace("\u3000", "")


def load_json(path):
    with open(path, "rt", encoding="UTF-8") as fp:
        return json.load(fp)


def dump_json(obj, path, ensure_ascii=False, indent=4):
    mkdir_fn(path)
    with open(path, "wt", encoding="UTF-8") as fp:
        json.dump(obj, fp, ensure_ascii=ensure_ascii, indent=indent)


def walk_dir(dd):
    return sorted([os.path.join(dn, fn) for dn, _, ff in os.walk(dd) for fn in ff])
