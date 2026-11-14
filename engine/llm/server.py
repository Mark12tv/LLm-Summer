"""Жизненный цикл `llama-server`: поиск бинарника, флаги, запуск, healthcheck.

Флаги собираются из `RuntimeProfile` — единственного источника правды.
Профиль `dev` держит часть слоёв в RAM (тесты идут на 4 ГБ VRAM),
профиль `player` грузит всё на GPU и включает квантованный KV.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from engine.core.config import MODEL_PATH, RUNS_DIR, RuntimeProfile

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

_HINT = (
    "llama-server не найден. Варианты:\n"
    "  1) скачать релиз llama.cpp (win-cuda-x64 или win-avx2-x64) и указать путь\n"
    "     переменной окружения LLAMA_SERVER;\n"
    "  2) положить llama-server.exe в D:\\Projects\\Summer\\tools\\bin\\;\n"
    "  3) использовать in-process бэкенд (engine.llm.backends.llamacpp)."
)


def find_server_binary() -> Path:
    env = os.environ.get("LLAMA_SERVER")
    if env:
        path = Path(env)
        if path.exists():
            return path
        raise FileNotFoundError(f"LLAMA_SERVER указывает на несуществующий файл: {path}")

    in_path = shutil.which("llama-server")
    if in_path:
        return Path(in_path)

    for candidate in (
        RUNS_DIR.parent / "tools" / "bin" / "llama-server.exe",
        Path("D:/LM/LM Studio/resources/app/.webpack/llama/llama-server.exe"),
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(_HINT)


def build_command(
    profile: RuntimeProfile,
    model_path: Path | str = MODEL_PATH,
    *,
    binary: Path | str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> list[str]:
    exe = str(binary) if binary else "llama-server"
    cmd = [
        exe,
        "-m", str(model_path),
        "-c", str(profile.total_ctx),
        "-np", str(profile.parallel),
        "-ngl", str(profile.n_gpu_layers),
        "--host", host,
        "--port", str(port),
        "-b", str(profile.n_batch),
        "-ub", str(profile.n_ubatch),
        "-t", str(profile.n_threads),
        "--cache-type-k", profile.kv_type,
        "--cache-type-v", profile.kv_type,
    ]
    if profile.flash_attn:
        cmd.append("--flash-attn")
    return cmd


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) != 0


@dataclass
class LlamaServer:
    profile: RuntimeProfile
    model_path: Path = MODEL_PATH
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    binary: Path | None = None
    log_path: Path | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self) -> list[str]:
        return build_command(
            self.profile,
            self.model_path,
            binary=self.binary or find_server_binary(),
            host=self.host,
            port=self.port,
        )

    def start(self, *, ready_timeout_s: float = 300.0) -> "LlamaServer":
        if self.process and self.process.poll() is None:
            return self
        if not port_is_free(self.host, self.port):
            raise RuntimeError(f"порт {self.port} занят: останови другой llama-server")

        log_path = self.log_path or (RUNS_DIR / "llama-server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")
        cmd = self.command()
        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            self.wait_ready(ready_timeout_s)
        except Exception:
            self.stop()
            raise
        return self

    def wait_ready(self, timeout_s: float = 300.0) -> None:
        import httpx

        deadline = time.time() + timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server завершился с кодом {self.process.returncode}; "
                    f"смотри лог: {self.log_path or (RUNS_DIR / 'llama-server.log')}"
                )
            try:
                resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.5)
        raise TimeoutError(f"llama-server не поднялся за {timeout_s} c: {last_error}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
