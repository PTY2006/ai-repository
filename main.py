#!/usr/bin/env python3
"""
LLM 交互式命令行工具 - 支持多轮对话记忆、续写故事、执行系统命令
"""

import os
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from collections import deque

# ========== 配置 ==========
ROOT_DIR = Path(__file__).parent.parent
HISTORY_MAXLEN = 5          # 保留最近 N 轮对话
CONTEXT_MAX_CHARS = 3000    # 历史上下文最大字符数（防止超限）

# 全局变量：对话历史（存储 (user_msg, assistant_msg) 元组）
conversation_history = deque(maxlen=HISTORY_MAXLEN)

# 全局变量：最后一次普通聊天的完整响应（用于文件写入）
last_chat_response = ""


def load_env():
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        print(f"错误: 未找到 .env 文件，请确保 {env_path} 存在")
        return {}
    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
    return env_vars


def build_system_prompt():
    """构建 system prompt，要求 LLM 输出结构化 JSON"""
    return """你是一个智能助手，需要根据用户输入返回 JSON 格式的响应。你必须严格遵守以下格式：

## 情况1：普通聊天、写故事、回答问题等（不需要执行系统命令）
返回：
{"action": "chat", "response": "你的回复内容"}

## 情况2：用户要求执行系统命令（如列出文件、创建目录等）
返回：
{"action": "execute", "commands": ["具体命令1", "具体命令2"]}

## 重要规则：
- 只输出 JSON，不要输出任何其他文字（包括解释、标记等）。
- 对于写故事或续写：如果用户说“继续”、“下一章”、“接着写”，你需要根据之前对话历史中的上文自然续写。
- 对于普通聊天：直接给出回复内容。
- 对于命令：只生成必要的命令，不要多余说明。

示例：
用户：写一首关于春天的诗
输出：{"action": "chat", "response": "春水碧于天，画船听雨眠..."}

用户：列出当前目录下所有 .py 文件
输出：{"action": "execute", "commands": ["ls *.py"]}   # Linux/macOS
或    {"action": "execute", "commands": ["dir *.py"]}   # Windows

用户：继续
输出：{"action": "chat", "response": "（根据上文续写的内容）"}
"""


def build_user_message_with_history(user_input):
    """将对话历史拼接到用户消息中，实现上下文承接"""
    if not conversation_history:
        return user_input

    # 构建历史上下文字符串
    history_text = "【对话历史】：\n"
    total_len = 0
    for idx, (usr, asst) in enumerate(conversation_history, 1):
        block = f"{idx}. 用户：{usr}\n   助手：{asst}\n"
        if total_len + len(block) > CONTEXT_MAX_CHARS:
            break
        history_text += block
        total_len += len(block)

    # 当前用户消息
    current = f"\n【当前用户】：\n{user_input}\n\n请根据对话历史（如果有）和当前用户消息，返回正确的 JSON。"
    return history_text + current


def call_llm(base_url, model, api_key, system_prompt, user_message, temperature=0.7, timeout=120):
    """调用 LLM，返回原始响应文本"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4000
    }
    if not base_url.endswith("/"):
        base_url += "/"
    url = base_url + "chat/completions"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return None
    except urllib.error.URLError as e:
        if "timed out" in str(e).lower():
            print(f"❌ LLM 请求超时（{timeout}秒），请减少字数或稍后重试。")
        else:
            print(f"❌ LLM 请求失败: {e}")
        return None
    except Exception as e:
        print(f"❌ LLM 调用异常: {e}")
        return None


def execute_commands(commands):
    """执行一系列 CMD 命令，返回 (成功标志, 输出信息)"""
    outputs = []
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="gbk"
            )
            if result.returncode == 0:
                outputs.append(f"[成功] {cmd}\n{result.stdout.strip()}")
            else:
                outputs.append(f"[失败] {cmd}\n错误: {result.stderr.strip()}")
                return False, "\n".join(outputs)
        except Exception as e:
            outputs.append(f"[异常] {cmd}\n{str(e)}")
            return False, "\n".join(outputs)
    return True, "\n".join(outputs)


def write_content_to_file(file_path, content, mode="w"):
    """直接使用 Python 将内容写入文件"""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return True, f"成功{'写入' if mode == 'w' else '追加'}文件: {path}"
    except Exception as e:
        return False, f"写入文件失败: {str(e)}"


def handle_write_last_chat(user_input):
    """检测用户是否要求将最后一次聊天内容写入文件"""
    global last_chat_response
    if not last_chat_response:
        return False, "没有找到最近生成的内容。请先生成一些内容（例如：写一篇小说），然后再尝试保存。"

    import re
    text = user_input.lower()
    if re.search(r'(把|将).*?刚刚.*?写入', text) or re.search(r'保存.*?上次.*?内容', text):
        path_match = re.search(r'([A-Za-z]:\\[^\s]+\.\w+)', user_input)
        if not path_match:
            return False, "请提供完整的文件路径，例如：把刚刚的内容写入 D:\\ches\\455.txt"
        file_path = path_match.group(1)
        if re.search(r'追加|添加|不要覆盖', text):
            mode = "a"
            action_desc = "追加"
        else:
            mode = "w"
            action_desc = "覆盖写入"
        print(f"\n📄 准备将最近生成的内容（共 {len(last_chat_response)} 字符）{action_desc}到文件：{file_path}")
        confirm = input("是否执行？(Y/n): ").strip().lower()
        if confirm == 'n' or confirm == 'no':
            return True, "已取消写入。"
        success, msg = write_content_to_file(file_path, last_chat_response, mode)
        return True, msg
    return False, ""


def main():
    global last_chat_response, conversation_history

    env = load_env()
    if not env:
        return
    base_url = env.get("LLM_BASE_URL")
    model = env.get("LLM_MODEL")
    api_key = env.get("LLM_API_KEY")
    if not all([base_url, model, api_key]):
        print("错误: .env 文件中缺少必要配置 (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY)")
        return
    temperature = float(env.get("LLM_TEMPERATURE", "0.7"))

    system_prompt = build_system_prompt()

    print("\n" + "=" * 50)
    print(" LLM 命令行工具（支持多轮对话记忆 + 续写 + 命令执行）")
    print(f" API: {base_url}")
    print(f" 模型: {model}")
    print("=" * 50)
    print(" 说明: 输入自然语言，我会自动理解是聊天还是执行命令。")
    print(" 提示: 可以写故事，然后说“继续”自动续写。")
    print(" 命令: /history 查看对话历史，/clear 清空历史，/quit 退出\n")

    while True:
        try:
            user_input = input("你: ").strip()
            if user_input.lower() in ["/quit", "/exit", "q"]:
                print("👋 再见！")
                break
            if not user_input:
                continue

            # 处理特殊命令
            if user_input.lower() == "/history":
                if not conversation_history:
                    print("暂无对话历史。")
                else:
                    print("\n--- 对话历史（最近 {} 轮）---".format(len(conversation_history)))
                    for i, (usr, asst) in enumerate(conversation_history, 1):
                        print(f"{i}. 用户: {usr[:50]}{'...' if len(usr)>50 else ''}")
                        print(f"   助手: {asst[:80]}{'...' if len(asst)>80 else ''}\n")
                print()
                continue

            if user_input.lower() == "/clear":
                conversation_history.clear()
                last_chat_response = ""
                print("✅ 对话历史已清空。\n")
                continue

            # 检测写入文件功能
            handled, msg = handle_write_last_chat(user_input)
            if handled:
                print(msg)
                print()
                continue

            # 构建带历史的消息
            user_msg_with_context = build_user_message_with_history(user_input)

            print("🤖 正在思考...")
            response_text = call_llm(base_url, model, api_key, system_prompt, user_msg_with_context, temperature, timeout=120)
            if not response_text:
                print("❌ LLM 响应失败，请重试。\n")
                continue

            # 尝试解析 JSON
            try:
                clean = response_text.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                if clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                data = json.loads(clean)
            except json.JSONDecodeError:
                # 如果 LLM 没有返回 JSON（比如模型不遵守指令），则当作普通聊天处理
                print("⚠️ LLM 未返回有效 JSON，将作为普通文本处理。")
                print("助手:", response_text)
                # 保存到历史
                conversation_history.append((user_input, response_text))
                last_chat_response = response_text
                print()
                continue

            action = data.get("action")
            if action == "chat":
                response_content = data.get("response", "")
                if not response_content:
                    print("❌ LLM 返回的 response 为空。")
                    continue
                print("助手:", response_content)
                # 保存到历史
                conversation_history.append((user_input, response_content))
                last_chat_response = response_content
                print()
            elif action == "execute":
                commands = data.get("commands", [])
                if not commands:
                    print("❌ 没有生成任何命令。")
                    continue
                print("\n📜 将要执行的命令：")
                for i, cmd in enumerate(commands, 1):
                    print(f"  {i}. {cmd}")
                confirm = input("\n是否执行这些命令？(Y/n): ").strip().lower()
                if confirm == 'n' or confirm == 'no':
                    print("已取消执行。\n")
                    continue
                success, output = execute_commands(commands)
                print("\n执行结果：")
                print(output)
                # 命令执行结果也可以选择是否保存到对话历史（这里不保存，因为不是聊天内容）
                print()
            else:
                print(f"❌ 未知的 action: {action}\n")
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}\n")


if __name__ == "__main__":
    main()
