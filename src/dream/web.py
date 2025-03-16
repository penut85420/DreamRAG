from typing import Iterable

from dream.utils import Auth, Dream, load_json


def web_demo(
    json_path="data/dream.json",
    npy_path="data/dream.npy",
    auth_path="data/auth.json",
    share=False,
):
    import faiss
    import gradio as gr
    import numpy as np
    from openai import OpenAI
    from openai.types.chat import ChatCompletionChunk

    auth = Auth.load(auth_path)
    client = OpenAI(api_key=auth.api_key)

    data = [Dream(**i) for i in load_json(json_path)]
    embs: np.ndarray = np.load(npy_path)

    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    with gr.Blocks(title="AI 解夢") as app:
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="解夢", type="messages", height=750)
                query_text = gr.Textbox(label="夢境描述", submit_btn=True)
            with gr.Column():
                prompt = gr.TextArea(label="完整輸入")

        def submit_query(query, chatbot: list):
            print(query)
            chatbot.append(dict(role="user", content=query))
            yield chatbot, None

            query_emb = client.embeddings.create(input=query, model="text-embedding-3-small")
            query_emb = np.array([query_emb.data[0].embedding])
            _, i = index.search(query_emb, 5)
            i: list[list[int]]

            body = [data[ii].body for ii in reversed(i[0])]
            body = "\n\n".join(body)

            inst = (
                f"{body}\n\n"
                f"參考以上周公解夢的說明，幫使用者解夢，使用繁體中文回答。\n\n"
                f"使用者夢境：{query}"
            )
            outputs: Iterable[ChatCompletionChunk] = client.chat.completions.create(
                messages=[dict(role="user", content=inst)],
                model="gpt-4o-mini",
                stream=True,
            )

            message = str()
            chatbot.append(dict(role="assistant", content=message))
            for chunk in outputs:
                token = chunk.choices[0].delta.content
                message += token if token else ""
                chatbot[-1] = dict(role="assistant", content=message)
                yield chatbot, inst

        query_text.submit(submit_query, [query_text, chatbot], [chatbot, prompt])

    app.launch(share=share, favicon_path="data/dream.png")
