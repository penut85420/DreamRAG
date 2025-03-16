from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

from dream.utils import Auth, Dream, dump_json, get_bs, load_json, proc_text


def process_data(
    json_path="data/dream.json",
    npy_path="data/dream.npy",
    auth_path="data/auth.json",
):
    # root_url = "https://www.golla.tw/"
    # targets = [
    #     *("", "renwu", "dongwu", "zhiwu", "wupin", "huodong", "shenghuo", "ziran"),
    #     *("guishen", "jianzhu", "qita", "yunfujiemeng", "mengjing", "wenhua", "health"),
    # ]

    # sub_urls = {_ for target in targets for _ in gather_dream_urls(urljoin(root_url, target) + "/")}
    # crawl_dreams(sub_urls, json_path)
    create_embs(json_path, npy_path, auth_path=auth_path)


def gather_dream_urls(root_url, visited: set = set(), use_cache=True):
    print(f"visit: {root_url}")
    visited.add(root_url)

    sub_urls = set()
    root_bs = get_bs(root_url, use_cache)
    for a in root_bs.find_all("a"):
        href = a.attrs["href"]
        if href.startswith("http"):
            continue
        sub_urls.add(urljoin(root_url, href))

    for url in list(sub_urls):
        if "list_" in url and url not in visited:
            sub_urls |= gather_dream_urls(url)

    return sub_urls


def crawl_dreams(sub_urls, output_path, use_cache=True):
    from tqdm import tqdm

    def make_item(url):
        sub_bs = get_bs(url, use_cache)

        def find_text(div_id):
            div = sub_bs.find("div", attrs=dict(id=div_id))
            return proc_text(div.text) if div else None

        title, body = find_text("entrytitle"), find_text("entrybody")
        return dict(title=title, body=body) if title else None

    item_list = list()
    with tqdm(total=len(sub_urls)) as progress:
        with ThreadPoolExecutor() as executor:
            for item in executor.map(make_item, sub_urls):
                item_list.append(item) if item else None
                progress.update()

    dump_json(item_list, output_path)


def create_embs(src_json, dst_npy, batch_size=512, auth_path="data/auth.json"):
    import numpy as np
    from openai import OpenAI
    from tqdm import trange

    auth = Auth.load(auth_path)
    client = OpenAI(api_key=auth.api_key)

    data = [Dream(**item) for item in load_json(src_json)]

    embs = list()
    for i in trange(0, len(data), batch_size, desc="embeddings"):
        batch = data[i : i + batch_size]
        batch = [d.title for d in batch]

        outputs = client.embeddings.create(input=batch, model="text-embedding-3-small")
        embs.extend([item.embedding for item in outputs.data])

    embs = np.array(embs)
    np.save(dst_npy, embs)
