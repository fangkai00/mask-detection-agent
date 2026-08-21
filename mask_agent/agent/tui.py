# -*- coding: utf-8 -*-
"""
TUI 终端交互模式

模拟 ChatGPT 风格的终端对话体验:
- 流式输出:逐字打印回答
- 快捷命令:/reset /history /export /quit /help
- 图片输入:/img <路径> 设置待检测图片,后续对话会携带该图片
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.cli import ConversationSession


BANNER = r"""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551                                                                  \u2551
\u2551          \U0001F637 口罩检测智能体 (Mask Detection Agent) 终端版      \u2551
\u2551                                                                  \u2551
\u2551   输入您的问题,或输入 /help 查看命令列表                          \u2551
\u2551   输入 /quit 退出                                               \u2551
\u2551                                                                  \u2551
\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D
"""

HELP_TEXT = """
\u53EF\u7528\u547D\u4EE4:
  /help     \u663E\u793A\u6B64\u5E2E\u52A9
  /reset    \u91CD\u7F6E\u5BF9\u8BDD(\u6E05\u7A7A\u5386\u53F2)
  /history  \u67E5\u770B\u5BF9\u8BDD\u5386\u53F2\u6458\u8981
  /export   \u5BFC\u51FA\u5B8C\u6574\u5BF9\u8BDD\u5386\u53F2\u4E3A JSON
  /img      \u8BBE\u7F6E\u5F85\u68C0\u6D4B\u56FE\u7247\u8DEF\u5F84, \u5982: /img data/images/test.jpg
  /clearimg \u6E05\u9664\u5F53\u524D\u56FE\u7247\u8DEF\u5F84
  /quit     \u9000\u51FA\u7A0B\u5E8F

\u793A\u4F8B\u95EE\u9898:
  (\u5148\u8BBE\u7F6E\u56FE\u7247) \u5E2E\u6211\u68C0\u6D4B\u8FD9\u5F20\u56FE\u91CC\u6709\u6CA1\u6709\u4EBA\u6234\u53E3\u7F69
  N95 \u53E3\u7F69\u548C\u533B\u7528\u5916\u79D1\u53E3\u7F69\u6709\u4EC0\u4E48\u533A\u522B?
  \u53E3\u7F69\u600E\u4E48\u6234\u624D\u6B63\u786E?
"""

# ANSI \u989C\u8272
COLOR_USER = "\033[94m"
COLOR_ASSISTANT = "\033[92m"
COLOR_SYSTEM = "\033[90m"
COLOR_HIGHLIGHT = "\033[96m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"


def _color(text: str, color: str) -> str:
    return f"{color}{text}{COLOR_RESET}"


def _print_user(msg: str):
    prefix = _color("\U0001F464 \u7528\u6237", COLOR_USER)
    print(f"\n{prefix}: {msg}")


def _print_assistant_start():
    prefix = _color("\U0001F916 \u52A9\u624B", COLOR_ASSISTANT)
    print(f"\n{prefix}: ", end="", flush=True)


def _print_assistant_end(turn):
    print()
    if turn.route_log:
        route_info = _color(" \u2192 ".join(turn.route_log), COLOR_DIM)
        print(f"  {route_info}")
    elapsed_info = _color(f"({turn.elapsed_s}s, {turn.steps}\u6B65)", COLOR_DIM)
    print(f"  {elapsed_info}")


def _print_system(msg: str):
    print(_color(msg, COLOR_SYSTEM))


def _print_highlight(msg: str):
    print(_color(msg, COLOR_HIGHLIGHT))


def _print_thinking_step(icon: str, label: str, done: bool = False, summary: str = ""):
    if done:
        prefix = _color(f"  \u2705 ", COLOR_BOLD)
        line = f"{prefix}{label}"
        if summary:
            line += _color(f"  ({summary})", COLOR_DIM)
        print(line)
    else:
        prefix = _color(f"  {icon} ", COLOR_HIGHLIGHT)
        print(f"{prefix}{_color(label, COLOR_DIM)}", end="", flush=True)
        for dot in [".", "..", "..."]:
            print(dot, end="", flush=True)
            import time
            time.sleep(0.15)
        print()


def run_tui():
    """启动终端交互模式。"""
    print(BANNER)

    session = ConversationSession(session_id="tui_default")
    current_image = ""  # \u5F53\u524D\u8BBE\u7F6E\u7684\u56FE\u7247\u8DEF\u5F84

    while True:
        try:
            prompt_suffix = f" [\u56FE\u7247: {os.path.basename(current_image)}]" if current_image else ""
            raw = input(_color(f"\n\u4F60{prompt_suffix}: ", COLOR_USER)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _print_system("\u518D\u89C1!")
            break

        if not raw:
            continue

        # \u547D\u4EE4\u5904\u7406
        if raw.startswith("/"):
            cmd_parts = raw.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                _print_system("\u518D\u89C1!")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/reset":
                session.reset()
                current_image = ""
                _print_highlight("\u2713 \u5BF9\u8BDD\u5DF2\u91CD\u7F6E")
            elif cmd == "/history":
                if not session.turns:
                    _print_system("(\u6682\u65E0\u5BF9\u8BDD\u5386\u53F2)")
                else:
                    _print_highlight(f"\u5BF9\u8BDD\u5386\u53F2 ({len(session.turns)} \u8F6E):")
                    for i, t in enumerate(session.turns, 1):
                        print(f"  {i}. \u7528\u6237: {t.user_input[:50]}{'...' if len(t.user_input) > 50 else ''}")
                        print(f"     \u52A9\u624B: {t.assistant_output[:60]}{'...' if len(t.assistant_output) > 60 else ''}")
                        print(f"     [{t.steps}\u6B65, {t.elapsed_s}s]")
            elif cmd == "/export":
                out = session.export_history()
                out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "logs", "conversation_export.json")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(out)
                _print_highlight(f"\u2713 \u5BF9\u8BDD\u5386\u53F2\u5DF2\u5BFC\u51FA\u5230: {out_path}")
            elif cmd == "/img":
                if not arg:
                    _print_system("\u8BF7\u63D0\u4F9B\u56FE\u7247\u8DEF\u5F84, \u5982: /img data/images/test.jpg")
                else:
                    img_path = arg.strip().strip('"').strip("'")
                    if os.path.exists(img_path):
                        current_image = os.path.abspath(img_path)
                        _print_highlight(f"\u2713 \u5DF2\u8BBE\u7F6E\u56FE\u7247: {current_image}")
                    else:
                        _print_system(f"\u56FE\u7247\u4E0D\u5B58\u5728: {img_path}")
            elif cmd == "/clearimg":
                current_image = ""
                _print_highlight("\u2713 \u5DF2\u6E05\u9664\u56FE\u7247\u8DEF\u5F84")
            else:
                _print_system(f"\u672A\u77E5\u547D\u4EE4: {cmd}, \u8F93\u5165 /help \u67E5\u770B\u53EF\u7528\u547D\u4EE4")
            continue

        # \u6B63\u5E38\u5BF9\u8BDD
        _print_user(raw)

        _print_highlight("  \u250C\u2500 \u601D\u8003\u8FC7\u7A0B \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        answer_started = False

        try:
            for event in session.chat_with_progress(raw, image_path=current_image):
                event_type = event.get("type", "")

                if event_type == "step_start":
                    _print_thinking_step(event["icon"], event["label"], done=False)

                elif event_type == "step_done":
                    summary = event.get("summary", "")
                    if summary:
                        _print_thinking_step("", "", done=True, summary=summary)

                elif event_type == "answer_start":
                    _print_highlight("  \u251C\u2500 \u751F\u6210\u56DE\u7B54:")
                    answer_started = True
                    prefix = _color("  \u2502 \U0001F916 ", COLOR_ASSISTANT)
                    print(f"{prefix}", end="", flush=True)

                elif event_type == "answer_chunk":
                    text = event["text"]
                    if answer_started and len(text) > 0:
                        print(text[-1], end="", flush=True)

                elif event_type == "answer_done":
                    print()

                elif event_type == "error":
                    print()
                    _print_system(f"  \u274C \u51FA\u9519: {event.get('message', '\u672A\u77E5\u9519\u8BEF')}")

        except Exception as e:
            print()
            _print_system(f"  \u274C \u5BF9\u8BDD\u51FA\u9519: {e}")
            continue

        _print_highlight("  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

        if session.turns:
            _print_assistant_end(session.turns[-1])

        print()


if __name__ == "__main__":
    run_tui()
