# -*- coding: utf-8 -*-
"""
口罩检测智能体 — 统一启动入口

用法:
    python agent_cli.py                      # 默认启动 Streamlit GUI
    python agent_cli.py --gui                # 启动 Streamlit GUI(显式)
    python agent_cli.py --tui                # 启动终端交互模式
    python agent_cli.py --test               # 单条测试(非交互)
    python agent_cli.py --test "问题文本"     # 测试指定问题
    python agent_cli.py --test "检测这张图" data/images/test.jpg  # 带图片测试

示例:
    python agent_cli.py --tui
    python agent_cli.py --test "N95和医用外科口罩有什么区别?"
"""

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import compat
import config


BANNER = """
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551                                                                  \u2551
\u2551          \U0001F637 Mask Detection Agent \u2014 \u53E3\u7F69\u68C0\u6D4B\u667A\u80FD\u4F53\u542F\u52A8\u5668   \u2551
\u2551                                                                  \u2551
\u2551          \u6A21\u5F0F\u9009\u62E9:                                          \u2551
\u2551            [1] GUI  \u2014 Streamlit Web \u754C\u9762 (ChatGPT \u98CE\u683C)      \u2551
\u2551            [2] TUI  \u2014 \u7EC8\u7AEF\u4EA4\u4E92\u6A21\u5F0F                        \u2551
\u2551            [3] Test \u2014 \u5355\u6761\u6D4B\u8BD5 (\u6307\u5B9A\u95EE\u9898\u6216\u9ED8\u8BA4\u95EE\u9898)      \u2551
\u2551                                                                  \u2551
\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D
"""


def launch_gui():
    """启动 Streamlit GUI。"""
    app_path = os.path.join(_ROOT, "app_streamlit.py")
    print(f"[\u542F\u52A8] Streamlit GUI: {app_path}")
    print("[\u63D0\u793A] \u6D4F\u89C8\u5668\u5C06\u81EA\u52A8\u6253\u5F00 http://localhost:8500")
    print("[\u63D0\u793A] \u6309 Ctrl+C \u9000\u51FA")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\n[\u9000\u51FA] \u5DF2\u505C\u6B62 Streamlit GUI\u3002")


def launch_tui():
    """启动 TUI 终端交互模式。"""
    from agent.tui import run_tui
    run_tui()


def launch_test(query: str = None, image_path: str = ""):
    """启动单条测试模式。"""
    from agent.cli import ConversationSession

    if query is None:
        query = "N95 \u53E3\u7F69\u548C\u533B\u7528\u5916\u79D1\u53E3\u7F69\u6709\u4EC0\u4E48\u533A\u522B?"

    print(f"\n\u6D4B\u8BD5\u95EE\u9898: {query}")
    if image_path:
        print(f"\u56FE\u7247\u8DEF\u5F84: {image_path}")
    print("=" * 60)

    session = ConversationSession(session_id="test")
    turn = session.chat(query, image_path=image_path)

    print(f"\n\u8DEF\u7531\u6B65\u6570: {turn.steps}")
    if turn.route_log:
        print(f"\u8DEF\u7531\u8DEF\u5F84: {' \u2192 '.join(turn.route_log)}")
    print(f"\u8017\u65F6: {turn.elapsed_s}s")
    print("=" * 60)
    print(f"\n\u56DE\u7B54:\n{turn.assistant_output}")
    print()


def _print_detection_mode():
    """打印当前实际使用的检测模式(进程内 / HTTP API)。"""
    use_api = getattr(config, "MASK_USE_API", False)
    if use_api:
        base_url = getattr(config, "MASK_API_BASE_URL", "http://localhost:8000")
        print(f"[\u68C0\u6D4B\u6A21\u5F0F] HTTP API \u8C03\u7528 \u2192 {base_url}")
    else:
        print("[\u68C0\u6D4B\u6A21\u5F0F] \u8FDB\u7A0B\u5185\u8C03\u7528(Function Call,\u6A21\u578B\u5E38\u9A7B\u5185\u5B58)")


def main():
    args = sys.argv[1:]

    _print_detection_mode()

    if "--gui" in args:
        launch_gui()
        return
    if "--tui" in args:
        launch_tui()
        return
    if "--test" in args:
        idx = args.index("--test")
        query = args[idx + 1] if idx + 1 < len(args) else None
        image_path = args[idx + 2] if idx + 2 < len(args) else ""
        launch_test(query, image_path)
        return

    # \u65E0\u53C2\u6570:\u4EA4\u4E92\u5F0F\u9009\u62E9
    print(BANNER)
    print(f"\u5F53\u524D\u914D\u7F6E: \u6A21\u578B={config.MAIN_LLM_MODEL}")
    print()

    try:
        choice = input("\u8BF7\u9009\u62E9\u6A21\u5F0F [1/2/3] (\u9ED8\u8BA4 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\u518D\u89C1!")
        return

    if choice == "2":
        launch_tui()
    elif choice == "3":
        try:
            q = input("\u8F93\u5165\u6D4B\u8BD5\u95EE\u9898 (\u56DE\u8F66\u4F7F\u7528\u9ED8\u8BA4): ").strip()
            img = input("\u8F93\u5165\u56FE\u7247\u8DEF\u5F84 (\u53EF\u7A7A): ").strip()
        except (EOFError, KeyboardInterrupt):
            q, img = "", ""
        launch_test(q or None, img or "")
    else:
        launch_gui()


if __name__ == "__main__":
    main()
