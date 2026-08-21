# -*- coding: utf-8 -*-
"""
口罩检测智能体 — Streamlit GUI (ChatGPT 风格 + 图片上传)

界面布局:
┌──────────┬──────────────────────────────────────────┐
│  侧边栏  │           主对话区                      │
│          │                                          │
│ ➕ 新对话 │  🤖 助手回答(左侧气泡)                 │
│ 📷 上传  │                                          │
│  图片    │  👤 用户输入(右侧气泡)                  │
│          │                                          │
│ 💬 历史  │  ┌────────────────────────────────────┐  │
│  对话1   │  │  输入框                       ➤  发送 │  │
│  对话2   │  └────────────────────────────────────┘  │
│          │                                          │
│ ⚙️ 设置  │                                          │
│  关于    │                                          │
└──────────┴──────────────────────────────────────────┘

运行: streamlit run app_streamlit.py
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from agent.cli import ConversationSession, ConversationTurn
import config

# streamlit-paste-button 装在项目内 vendor 目录(--target,绕过 site-packages 权限),
# 且用 --no-deps 避免覆盖全局依赖。加 sys.path 后导入。
_VENDOR = os.path.join(_ROOT, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
try:
    from streamlit_paste_button import paste_image_button, PasteResult
    _PASTE_BUTTON_AVAILABLE = True
except Exception as _e:
    _PASTE_BUTTON_AVAILABLE = False
    PasteResult = None
    paste_image_button = None

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="口罩检测智能体",
    page_icon="\U0001F637",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Mask Detection Agent — 基于 LangGraph + YOLOv8 的口罩检测智能体",
    },
)

# ============================================================
# 自定义 CSS (ChatGPT 风格)
# ============================================================
CUSTOM_CSS = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.stApp {
    max-width: 100% !important;
}

.chat-message {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    line-height: 1.6;
    font-size: 15px;
    word-wrap: break-word;
}

.chat-message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: 15%;
    border-bottom-right-radius: 4px;
}

.chat-message.assistant {
    background-color: #f7f7f8;
    color: #1f1f1f;
    margin-right: 15%;
    border-bottom-left-radius: 4px;
    border: 1px solid #e8e8e8;
}

.route-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin-right: 4px;
    margin-top: 6px;
    background-color: #e8f4fd;
    color: #1976d2;
    border: 1px solid #bbdefb;
}

.thinking-step {
    padding: 4px 0;
    font-size: 13px;
    color: #666;
}

.thinking-step .icon {
    margin-right: 6px;
}

.uploaded-image-preview {
    max-width: 300px;
    border-radius: 8px;
    margin: 8px 0;
    border: 2px solid #e0e0e0;
}

.detection-result {
    background-color: #f0f7ff;
    border: 1px solid #bbdefb;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}

.detection-result img {
    max-width: 100%;
    border-radius: 8px;
    margin-top: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 会话状态初始化
# ============================================================
def init_session_state():
    """初始化 Streamlit session_state。

    首次启动时从磁盘加载所有历史会话(跨 GUI 重启保留历史)。
    """
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
        # 从磁盘加载历史会话(session_store.list_sessions 列出所有持久化会话)
        try:
            from agent.session_store import list_sessions
            persisted = list_sessions()
            loaded_count = 0
            for item in persisted:
                sid = item.get("session_id", "")
                if not sid:
                    continue
                sess = ConversationSession.load_from_disk(sid)
                if sess is not None:
                    st.session_state.sessions[sid] = sess
                    loaded_count += 1
            if loaded_count:
                print(f"[\u542F\u52A8] \u5DF2\u52A0\u8F7D {loaded_count} \u4E2A\u5386\u53F2\u4F1A\u8BDD")
        except Exception as e:
            print(f"[\u8B66\u544A] \u52A0\u8F7D\u5386\u53F2\u4F1A\u8BDD\u5931\u8D25: {e}")

    if "current_session_id" not in st.session_state:
        # 默认选中最近更新的历史会话(若有),否则新建
        from agent.session_store import list_sessions
        try:
            persisted = list_sessions()
        except Exception:
            persisted = []
        if persisted:
            st.session_state.current_session_id = persisted[0].get("session_id", "")
        else:
            st.session_state.current_session_id = ""
        # 若 current_session_id 为空(无历史),创建一个新会话
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = str(uuid.uuid4())[:8]

    if st.session_state.current_session_id not in st.session_state.sessions:
        # 先尝试从磁盘加载(用户输入的 sid 可能不在内存)
        loaded = ConversationSession.load_from_disk(st.session_state.current_session_id)
        if loaded is not None:
            st.session_state.sessions[st.session_state.current_session_id] = loaded
        else:
            st.session_state.sessions[st.session_state.current_session_id] = ConversationSession(
                session_id=st.session_state.current_session_id
            )

    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = ""
    if "messages" not in st.session_state:
        # 重建当前会话的消息列表(用于主对话区显示)
        st.session_state.messages = []
        sess = st.session_state.sessions.get(st.session_state.current_session_id)
        if sess:
            for t in sess.turns:
                st.session_state.messages.append({
                    "role": "user", "content": t.user_input, "image": t.image_path
                })
                st.session_state.messages.append({
                    "role": "assistant", "content": t.assistant_output,
                    "route": t.route_log, "elapsed": t.elapsed_s, "steps": t.steps,
                })


def get_current_session() -> ConversationSession:
    """获取当前对话会话。"""
    return st.session_state.sessions[st.session_state.current_session_id]


def new_session():
    """新建对话。"""
    sid = str(uuid.uuid4())[:8]
    st.session_state.current_session_id = sid
    st.session_state.sessions[sid] = ConversationSession(session_id=sid)
    st.session_state.uploaded_image_path = ""
    st.session_state.messages = []
    # 新会话立即持久化(空 turn 也会落盘,侧边栏立即可见)
    try:
        st.session_state.sessions[sid].save_to_disk()
    except Exception as e:
        print(f"[\u8B66\u544A] \u65B0\u5EFA\u4F1A\u8BDD\u4FDD\u5B58\u5931\u8D25: {e}")
    st.rerun()


# ============================================================
# 图片上传处理
# ============================================================
def handle_image_upload(uploaded_file):
    """处理用户上传的图片,保存到 data/images/ 目录。"""
    if uploaded_file is None:
        return ""

    upload_dir = Path(getattr(config, "UPLOAD_IMAGE_DIR", "data/images"))
    if not upload_dir.is_absolute():
        upload_dir = Path(config.PROJECT_ROOT) / upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件
    file_ext = Path(uploaded_file.name).suffix or ".jpg"
    file_name = f"upload_{uuid.uuid4().hex[:8]}{file_ext}"
    file_path = upload_dir / file_name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.uploaded_image_path = str(file_path)
    return str(file_path)


def handle_paste_image(pil_image) -> str:
    """处理粘贴/截图上传的 PIL.Image,保存到 data/images/ 目录。

    与 handle_image_upload 不同,粘贴来源没有文件名,统一存为 png。
    """
    if pil_image is None:
        return ""

    upload_dir = Path(getattr(config, "UPLOAD_IMAGE_DIR", "data/images"))
    if not upload_dir.is_absolute():
        upload_dir = Path(config.PROJECT_ROOT) / upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"paste_{uuid.uuid4().hex[:8]}.png"
    file_path = upload_dir / file_name

    # 统一转 RGB(避免 RGBA/模式 P 存 PNG 时的问题),保留 alpha 则直接存
    try:
        if pil_image.mode in ("RGBA", "LA", "P"):
            pil_image.save(file_path, "PNG")
        else:
            pil_image.convert("RGB").save(file_path, "PNG")
    except Exception:
        # 兜底:直接按原模式存
        pil_image.save(file_path, "PNG")

    st.session_state.uploaded_image_path = str(file_path)
    return str(file_path)


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    """渲染侧边栏。"""
    with st.sidebar:
        st.markdown("### \U0001F637 口罩检测智能体")

        # 新对话按钮
        if st.button("\u2795 \u65B0\u5BF9\u8BDD", width="stretch"):
            new_session()

        st.divider()

        # 图片上传(支持三种方式:文件选择 / 拖拽 / 粘贴或截图)
        st.markdown("#### \U0001F4F7 \u4E0A\u4F20\u56FE\u7247")

        # 方式1:文件选择 / 拖拽(Streamlit 原生)
        uploaded_file = st.file_uploader(
            "\u9009\u62E9\u6587\u4EF6\u6216\u62D6\u62FD\u5230\u6B64",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            img_path = handle_image_upload(uploaded_file)
            st.success(f"\u2713 \u6587\u4EF6\u5DF2\u4E0A\u4F20: {os.path.basename(img_path)}")

        # 方式2:粘贴图片 / 截图(剪贴板,需 streamlit-paste-button)
        if _PASTE_BUTTON_AVAILABLE:
            paste_result = paste_image_button(
                label="\U0001F4CB \u70B9\u6B64\u7C98\u8D34\u56FE\u7247/\u622A\u56FE",
                key="paste_btn",
                background_color="#3498db",
                hover_background_color="#2980b9",
            )
            if paste_result.image_data is not None:
                img_path = handle_paste_image(paste_result.image_data)
                st.success(f"\u2713 \u7C98\u8D34\u6210\u529F: {os.path.basename(img_path)}")
        else:
            st.caption("\u26A0 \u7C98\u8D34\u7EC4\u4EF6\u672A\u5B89\u88C5\uFF0C\u4EC5\u652F\u6301\u6587\u4EF6/\u62D6\u62FD\u4E0A\u4F20")

        # 统一预览(文件或粘贴都基于 uploaded_image_path 显示)
        if st.session_state.uploaded_image_path:
            st.image(st.session_state.uploaded_image_path, caption="\u5F85\u68C0\u6D4B\u56FE\u7247", width="stretch")

        # 清除图片
        if st.session_state.uploaded_image_path:
            if st.button("\u274C \u6E05\u9664\u56FE\u7247", width="stretch"):
                st.session_state.uploaded_image_path = ""
                st.rerun()

        st.divider()

        # 对话历史(按最近更新降序:当前会话置顶,其余按 turns 数/末轮时间排)
        st.markdown("#### \U0001F4AC \u5BF9\u8BDD\u5386\u53F2")
        sessions = st.session_state.sessions
        # 排序:当前会话置顶,其余按 turns 数降序(多轮的会话优先)
        sorted_sids = sorted(
            sessions.keys(),
            key=lambda s: (
                s == st.session_state.current_session_id,  # 当前会话排最前
                len(sessions[s].turns),  # turns 多的排前
            ),
            reverse=True,
        )
        for sid in sorted_sids:
            sess = sessions[sid]
            if sess.turns:
                first_q = sess.turns[0].user_input[:20]
                label = f"{first_q}...  [{sid}]"
            else:
                label = f"(\u65B0\u5BF9\u8BDD) [{sid}]"
            is_current = sid == st.session_state.current_session_id
            if st.button(label, key=f"session_{sid}", width="stretch",
                         disabled=is_current,
                         help="\u5F53\u524D\u4F1A\u8BDD" if is_current else "\u70B9\u51FB\u67E5\u770B\u8BE5\u5BF9\u8BDD"):
                st.session_state.current_session_id = sid
                st.session_state.uploaded_image_path = ""
                # 重建消息显示
                st.session_state.messages = []
                for t in sess.turns:
                    st.session_state.messages.append({"role": "user", "content": t.user_input, "image": t.image_path})
                    st.session_state.messages.append({"role": "assistant", "content": t.assistant_output,
                                                       "route": t.route_log, "elapsed": t.elapsed_s, "steps": t.steps})
                st.rerun()

        st.divider()

        # 设置
        with st.expander("\u2699\uFE0F \u8BBE\u7F6E"):
            st.write(f"**\u6A21\u578B**: {config.MAIN_LLM_MODEL}")
            st.write(f"**\u68C0\u6D4B\u7F6E\u4FE1\u5EA6**: {getattr(config, 'MASK_CONF_THRESHOLD', 0.25)}")
            st.write(f"**\u6700\u5927\u6B65\u6570**: {config.MAX_PLANNER_STEPS}")

        with st.expander("\u2139\uFE0F \u5173\u4E8E"):
            st.markdown("""
            **\u53E3\u7F69\u68C0\u6D4B\u667A\u80FD\u4F53** \u57FA\u4E8E:
            - **LangGraph** \u72B6\u6001\u56FE\u7F16\u6392
            - **YOLOv8** \u53E3\u7F69\u68C0\u6D4B\u6A21\u578B
            - **DashScope Qwen** \u5927\u8BED\u8A00\u6A21\u578B

            \u529F\u80FD:
            - \u4E0A\u4F20\u56FE\u7247\u68C0\u6D4B\u53E3\u7F69\u4F69\u6234\u60C5\u51B5
            - \u56DE\u7B54\u53E3\u7F69\u76F8\u5173\u95EE\u9898(\u653F\u7B56/\u77E5\u8BC6)
            - \u591A\u8F6E\u5BF9\u8BDD\u4E0A\u4E0B\u6587\u8BB0\u5FC6
            """)


# ============================================================
# 主对话区
# ============================================================
def render_chat_area():
    """渲染主对话区。"""
    st.markdown("## \U0001F637 \u53E3\u7F69\u68C0\u6D4B\u667A\u80FD\u4F53")

    # 显示当前图片状态
    if st.session_state.uploaded_image_path:
        st.info(f"\U0001F4F7 \u5F53\u524D\u56FE\u7247: `{os.path.basename(st.session_state.uploaded_image_path)}`")

    # 显示历史消息
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            # 用户消息可能带图片
            if role == "user" and msg.get("image"):
                st.image(msg["image"], caption="\u5DF2\u4E0A\u4F20\u56FE\u7247", width=300)
            st.markdown(content)

            # 助手消息附带路由信息和标注图
            if role == "assistant":
                route = msg.get("route", [])
                if route:
                    tags = " ".join(f"`{r}`" for r in route)
                    st.caption(f"\u8DEF\u7531: {tags} | \u8017\u65F6 {msg.get('elapsed', 0)}s | {msg.get('steps', 0)}\u6B65")

                # 检测结果中的标注图
                # (从 session turns 中查找对应轮次的 detection_result)
                # 这里通过 content 中的标注图路径判断
                if "\u6807\u6CE8\u56FE" in content or "annotated" in content.lower():
                    pass  # finalizer 的回答里会包含路径,如需展示图可扩展

    # 输入区
    if prompt := st.chat_input("\u8F93\u5165\u60A8\u7684\u95EE\u9898..."):
        image_path = st.session_state.uploaded_image_path

        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt, "image": image_path})
        with st.chat_message("user"):
            if image_path:
                st.image(image_path, caption="\u5F85\u68C0\u6D4B\u56FE\u7247", width=300)
            st.markdown(prompt)

        # 调用 Agent
        session = get_current_session()
        with st.chat_message("assistant"):
            with st.spinner("\u6B63\u5728\u601D\u8003..."):
                # 使用带进度的对话
                route_log = []
                answer_text = ""
                steps = 0
                elapsed = 0
                # 流式回答占位符:answer_start 时创建,answer_chunk 实时更新,
                # answer_done 用完整文本覆盖(确保最终文本与 cli.py 累积一致)
                answer_placeholder = None
                streaming_text = ""

                try:
                    final_turn = None
                    for event in session.chat_with_progress(prompt, image_path=image_path):
                        etype = event.get("type", "")

                        if etype == "step_start":
                            icon = event.get("icon", "")
                            label = event.get("label", "")
                            st.caption(f"{icon} {label}...")

                        elif etype == "step_done":
                            summary = event.get("summary", "")
                            fields = event.get("fields", {})
                            node = event.get("node", "")
                            if summary:
                                st.caption(f"  \u2705 {summary}")
                            # 思考过程结构化展示:JSON 代码块(可折叠)
                            if fields:
                                node_label_map = {
                                    "planner": "\u89C4\u5212",
                                    "mask_detect": "\u53E3\u7F69\u68C0\u6D4B",
                                    "search": "\u8054\u7F51\u641C\u7D22",
                                    "finalizer": "\u7EC8\u7B54",
                                }
                                node_label = node_label_map.get(node, node)
                                with st.expander(f"\U0001F4CB {node_label}\uFF1A\u7ED3\u6784\u5316\u601D\u8003\u8FC7\u7A0B", expanded=False):
                                    st.code(
                                        json.dumps(fields, ensure_ascii=False, indent=2),
                                        language="json",
                                    )

                        elif etype == "answer_start":
                            # finalizer 开始流式输出,创建占位符
                            answer_placeholder = st.empty()
                            streaming_text = ""

                        elif etype == "answer_chunk":
                            # 真流式 token:累积并实时刷新占位符
                            streaming_text += event.get("text", "")
                            if answer_placeholder is not None:
                                answer_placeholder.markdown(streaming_text)

                        elif etype == "answer_done":
                            # 以完整 answer 为准(流式 chunk 拼接可能有细微差异)
                            answer_text = event.get("answer", "")
                            if answer_placeholder is not None:
                                answer_placeholder.markdown(answer_text)
                            else:
                                # 未触发流式(异常兜底),直接渲染
                                st.markdown(answer_text)

                        elif etype == "error":
                            err_msg = f"\u274C \u51FA\u9519: {event.get('message', '')}"
                            if answer_placeholder is not None:
                                answer_placeholder.markdown(err_msg)
                            else:
                                st.markdown(err_msg)
                            answer_text = err_msg

                    # 获取最后一轮的元数据
                    if session.turns:
                        turn = session.turns[-1]
                        route_log = turn.route_log
                        steps = turn.steps
                        elapsed = turn.elapsed_s

                except Exception as e:
                    err_msg = f"\u274C \u5BF9\u8BDD\u51FA\u9519: {e}"
                    if answer_placeholder is not None:
                        answer_placeholder.markdown(err_msg)
                    else:
                        st.markdown(err_msg)
                    answer_text = err_msg

                # 兜底:若既无 placeholder 也无 answer_done(极异常),至少显示占位
                if not answer_text and answer_placeholder is None:
                    answer_text = "(\u65E0\u56DE\u7B54)"
                    st.markdown(answer_text)

                if route_log:
                    tags = " ".join(f"`{r}`" for r in route_log)
                    st.caption(f"\u8DEF\u7531: {tags} | \u8017\u65F6 {elapsed}s | {steps}\u6B65")

                    # 检测结果中的标注图展示
                    if session.turns and session.turns[-1].image_path:
                        # 尝试从 detection_result 获取标注图
                        # (通过 graph run 的 state 中获取,这里通过 turn 间接获取)
                        pass

                # 保存助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "route": route_log,
                    "elapsed": elapsed,
                    "steps": steps,
                })

                # 持久化当前会话到磁盘(每轮对话完成后自动保存,跨重启保留)
                # session.chat_with_progress 已把 turn 追加到 session.turns
                try:
                    session.save_to_disk()
                except Exception as e:
                    print(f"[\u8B66\u544A] \u4F1A\u8BDD\u4FDD\u5B58\u5931\u8D25: {e}")

        # 检测完成后清除图片(可选,用户可能想连续检测多张)
        # st.session_state.uploaded_image_path = ""


# ============================================================
# 主函数
# ============================================================
def main():
    init_session_state()
    render_sidebar()
    render_chat_area()


if __name__ == "__main__":
    main()
