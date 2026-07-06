"""
校园规章制度问答AI助手 — 一键启动入口
用法：
    python run.py          # 启动服务并自动打开浏览器
    python run.py --no-browser  # 启动但不打开浏览器
"""
import os
import sys
import time
import subprocess
import webbrowser


def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(app_dir, "app.py")
    venv_python = os.path.join(app_dir, ".venv", "Scripts", "python.exe")

    # 优先使用虚拟环境的 Python
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    open_browser = "--no-browser" not in sys.argv

    print("=" * 50)
    print("  校园规章制度问答AI助手")
    print(f"  地址: http://localhost:8501")
    print("  按 Ctrl+C 停止服务")
    print("=" * 50)

    # 自动打开浏览器
    if open_browser:
        def _open():
            time.sleep(2)
            webbrowser.open("http://localhost:8501")

        import threading
        threading.Thread(target=_open, daemon=True).start()

    # 启动 Streamlit
    subprocess.run(
        [python_exe, "-m", "streamlit", "run", app_path,
         "--server.port", "8501",
         "--server.headless", "true",
         "--browser.serverAddress", "localhost"],
        cwd=app_dir,
    )


if __name__ == "__main__":
    main()
