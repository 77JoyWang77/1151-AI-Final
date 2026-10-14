"""Gradio 介面：輸入主題 → 檢視／編輯分鏡 → 生圖 → 單格重生 → 排版下載。

邏輯在 app_logic.py；這裡只負責介面元件與事件綁定。
生圖後端（renderer）與分鏡產生器（make_storyboard）由呼叫端傳入：
本機用 DemoRenderer（無 GPU），Colab 上換成 SDXL renderer。
"""
import gradio as gr

from .app_logic import apply_dialogue_rows, build_book, dialogue_rows, storyboard_markdown

DIALOGUE_HEADERS = ["格", "說話者（角色 id 或 narrator）", "台詞"]


def build_app(renderer, examples: dict, make_storyboard) -> gr.Blocks:
    """examples：{顯示名稱: 分鏡 dict}；make_storyboard(topic, n_panels) -> 分鏡 dict。"""

    def show(board):
        """載入新分鏡後：清空已生成的圖，更新摘要、台詞表與格數選單。"""
        choices = [p["index"] for p in board["panels"]]
        return (board, {}, {}, {}, storyboard_markdown(board), dialogue_rows(board),
                gr.Dropdown(choices=choices, value=choices[0]), [], [], None)

    def load_example(name):
        if not name:
            raise gr.Error("請先選一個範例故事")
        return show(examples[name])

    def new_storyboard(topic, n_panels):
        if not topic.strip():
            raise gr.Error("請輸入故事主題")
        try:
            board = make_storyboard(topic.strip(), int(n_panels))
        except Exception as e:  # API key、網路、驗證失敗等，顯示在介面上
            raise gr.Error(f"產生分鏡失敗：{e}")
        return show(board)

    def gallery(board, images):
        return [(images[p["index"]], f"第 {p['index']} 格") for p in board["panels"] if p["index"] in images]

    def render_all(board, images, boxes, attempts):
        if not board:
            raise gr.Error("請先產生或載入分鏡")
        images, boxes = dict(images), dict(boxes)
        for p in board["panels"]:
            i = p["index"]
            images[i] = renderer.panel(board, i, attempts.get(i, 0))
            boxes[i] = renderer.speaker_boxes(board, i, attempts.get(i, 0))
        return images, boxes, gallery(board, images)

    def regen_panel(index, board, images, boxes, attempts):
        if not images:
            raise gr.Error("請先生圖")
        i = int(index)
        attempts = {**attempts, i: attempts.get(i, 0) + 1}
        images = {**images, i: renderer.panel(board, i, attempts[i])}
        boxes = {**boxes, i: renderer.speaker_boxes(board, i, attempts[i])}
        return images, boxes, attempts, gallery(board, images)

    def make_book(rows, board, images, boxes):
        if not images or len(images) < len(board["panels"]):
            raise gr.Error("請先生圖")
        try:
            edited = apply_dialogue_rows(board, rows)
        except ValueError as e:
            raise gr.Error(str(e))
        pages, pdf = build_book(edited, images, boxes)
        return edited, pages, str(pdf)

    with gr.Blocks(title="繪本生成 Agent") as app:
        board = gr.State({})
        images, boxes, attempts = gr.State({}), gr.State({}), gr.State({})
        gr.Markdown("# 繪本生成 Agent\n輸入主題產生分鏡（或載入範例）→ 生圖 → 單格重生 → 修改台詞 → 排版下載")

        with gr.Row():
            with gr.Column():
                topic = gr.Textbox(label="故事主題", placeholder="例如：小女孩在雨天找到一隻迷路的小貓")
                n_panels = gr.Slider(6, 8, value=6, step=1, label="格數")
                btn_new = gr.Button("產生分鏡（呼叫 LLM）", variant="primary")
            with gr.Column():
                example = gr.Dropdown(list(examples), label="或載入範例故事（不需呼叫 API）")
                btn_example = gr.Button("載入範例")

        summary = gr.Markdown()
        table = gr.Dataframe(headers=DIALOGUE_HEADERS, datatype=["number", "str", "str"], type="array",
                             interactive=True, label="台詞（可直接修改；排版時套用）")

        btn_render = gr.Button("生圖", variant="primary")
        panels = gr.Gallery(label="各格", columns=3, height="auto")
        with gr.Row():
            panel_no = gr.Dropdown([], label="要重生的格")
            btn_regen = gr.Button("重生這一格")

        btn_book = gr.Button("排版成繪本", variant="primary")
        pages = gr.Gallery(label="繪本頁面", columns=4, height="auto")
        pdf = gr.File(label="下載 PDF")

        outputs = [board, images, boxes, attempts, summary, table, panel_no, panels, pages, pdf]
        btn_example.click(load_example, example, outputs)
        btn_new.click(new_storyboard, [topic, n_panels], outputs)
        btn_render.click(render_all, [board, images, boxes, attempts], [images, boxes, panels])
        btn_regen.click(regen_panel, [panel_no, board, images, boxes, attempts], [images, boxes, attempts, panels])
        btn_book.click(make_book, [table, board, images, boxes], [board, pages, pdf])
    return app
