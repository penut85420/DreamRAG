from typing import Iterable

from dream.utils import Auth, Dream, load_json


def cli_demo(json_path, npy_path, query, auth_path="data/auth.json"):
    import faiss
    import numpy as np
    from openai import OpenAI
    from openai.types.chat import ChatCompletionChunk

    auth = Auth.load(auth_path)
    client = OpenAI(api_key=auth.api_key)

    data = [Dream(**i) for i in load_json(json_path)]
    embs: np.ndarray = np.load(npy_path)

    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    query_emb = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_emb = np.array([query_emb.data[0].embedding])
    _, i = index.search(query_emb, 5)
    i: list[list[int]]

    body = [data[ii].body for ii in reversed(i[0])]
    body = "\n\n".join(body)
    print(body)

    inst = f"{body}\n\n請根據以上周公解夢的說明，幫使用者解夢。\n\n使用者夢境：{query}"
    outputs: Iterable[ChatCompletionChunk] = client.chat.completions.create(
        messages=[dict(role="user", content=inst)],
        model="gpt-4o-mini",
        stream=True,
    )

    for chunk in outputs:
        print(end=chunk.choices[0].delta.content, flush=True)
