# -*- coding: utf-8 -*-
# Author: eddy
# Model cache for LLM GGUF inference

import os
import re
import sys
import time
import threading
import subprocess
import tempfile
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Callable

# Regex to strip ANSI escape codes
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Regex to strip thinking chain tags
THINK_CHAIN = re.compile(r'<think>.*?</think>', re.DOTALL)

# Try to import llama-cpp-python binding
try:
    from llama_cpp import Llama
    USE_BINDING = True
    logging.info("llama-cpp-python binding available, using native inference")
except ImportError:
    USE_BINDING = False
    logging.warning("llama-cpp-python not found, falling back to subprocess mode")


def normalize_path_for_os(path: str) -> str:
    """Normalize path separators for the current OS (/ on Linux, \\ on Windows)."""
    if not path:
        return path
    if sys.platform == "win32":
        path = path.replace("/", "\\")
    else:
        path = path.replace("\\", "/")
    return os.path.normpath(path)


def resolve_gguf_model_path(model_name: str, folder: str = "LLM") -> str:
    """Resolve a GGUF path from ComfyUI model name, relative path, or absolute path."""
    if not model_name or not str(model_name).strip():
        raise FileNotFoundError("Model name is empty.")

    model_name = normalize_path_for_os(str(model_name).strip())
    tried = []

    def _try(path: str) -> Optional[str]:
        if not path:
            return None
        path = normalize_path_for_os(path)
        tried.append(path)
        if os.path.isfile(path):
            return os.path.abspath(path)
        return None

    found = _try(model_name)
    if found:
        return found

    try:
        import folder_paths
        full = folder_paths.get_full_path(folder, model_name)
        found = _try(full)
        if found:
            return found

        roots = folder_paths.folder_names_and_paths.get(folder, ([], set()))[0]
        basename = os.path.basename(model_name.replace("\\", "/"))
        for root in roots:
            for candidate in (
                os.path.join(root, model_name),
                os.path.join(root, basename),
            ):
                found = _try(candidate)
                if found:
                    return found
    except ImportError:
        roots = []

    msg = (
        f"GGUF model not found: '{model_name}'. "
        f"Checked paths: {tried[:8]}{'...' if len(tried) > 8 else ''}. "
        f"Place the .gguf in ComfyUI/models/LLM/ or use a full path that exists on this machine."
    )
    raise FileNotFoundError(msg)


def _model_file_diagnostics(model_path: str) -> str:
    if not model_path:
        return "path is empty"
    if not os.path.isfile(model_path):
        return f"file does not exist: {model_path}"
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return f"{model_path} ({size_mb:.1f} MB)"


def _find_llama_cli_optional() -> Optional[str]:
    try:
        return SubprocessModel._find_llama_cli_static()
    except FileNotFoundError:
        return None


class SubprocessModel:
    """Fallback model wrapper using llama-cli.exe subprocess."""

    is_subprocess = True

    def __init__(self, model_path: str, llama_cli_path: str = None, **kwargs):
        self.model_path = normalize_path_for_os(model_path)
        cli = llama_cli_path or self._find_llama_cli()
        self.llama_cli_path = normalize_path_for_os(cli)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", 99)
        self.n_ctx = kwargs.get("n_ctx", 32768)
        # Safety net so a stuck subprocess can never hang ComfyUI forever (seconds).
        self.timeout = kwargs.get("timeout", 600)
        # Newer llama.cpp moved non-conversation completion out of llama-cli into a
        # separate llama-completion binary, so prefer it when present.
        self.binary = self._resolve_completion_binary(self.llama_cli_path)
        # Probe --help once so we only pass flags this build actually understands.
        self._help = self._probe_help(self.binary)
        self.supports_no_cnv = ("-no-cnv" in self._help) or ("--no-conversation" in self._help)
        self.is_completion = "completion" in os.path.basename(self.binary).lower()

    @staticmethod
    def _resolve_completion_binary(cli_path: str) -> str:
        """Prefer a sibling llama-completion binary (new llama.cpp) over llama-cli."""
        cli_path = normalize_path_for_os(cli_path)
        name = os.path.basename(cli_path).lower()
        if "completion" in name:
            return cli_path
        exe = ".exe" if sys.platform == "win32" else ""
        candidate = os.path.join(os.path.dirname(cli_path), "llama-completion" + exe)
        if os.path.isfile(candidate):
            logging.info("Using llama-completion binary: %s", candidate)
            return normalize_path_for_os(candidate)
        return cli_path

    @staticmethod
    def _probe_help(binary: str) -> str:
        """Return the --help text of a binary (used to detect supported flags)."""
        try:
            proc = subprocess.run(
                [binary, "--help"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return (proc.stdout or "") + (proc.stderr or "")
        except Exception as err:
            logging.warning("Could not probe %s --help: %s", binary, err)
            return ""

    @staticmethod
    def _find_llama_cli_static() -> str:
        """Find llama-cli in common locations for the current OS."""
        if sys.platform == "win32":
            possible_paths = [
                r"C:\Users\Administrator\Desktop\222\llama.cpp\build\bin\llama-cli.exe",
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama-cli.exe"),
            ]
        else:
            possible_paths = [
                "/usr/local/bin/llama-cli",
                "/usr/bin/llama-cli",
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama-cli"),
            ]
        for path in possible_paths:
            if os.path.exists(path):
                return normalize_path_for_os(path)
        binary = "llama-cli.exe" if sys.platform == "win32" else "llama-cli"
        raise FileNotFoundError(f"{binary} not found. Please set the path for your OS in Load GGUF Model.")

    def _find_llama_cli(self) -> str:
        return self._find_llama_cli_static()

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip prompt echo, control tokens and noise from raw subprocess output."""
        if not text:
            return ""
        # If the prompt was echoed back, keep only what follows the assistant tag.
        marker = "<|im_start|>assistant"
        if marker in text:
            text = text.rsplit(marker, 1)[1]
            if text.startswith("\n"):
                text = text[1:]
        # Cut anything after the assistant's end-of-turn token.
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>", 1)[0]
        text = ANSI_ESCAPE.sub("", text)
        text = THINK_CHAIN.sub("", text)
        text = text.replace("[end of text]", "")
        return text.strip()

    def __call__(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 40, repeat_penalty: float = 1.1,
                 stream: bool = False, callback: Callable = None, **kwargs) -> str:
        """Run inference using subprocess."""
        # Write prompt to temp file
        fd, temp_file = tempfile.mkstemp(suffix=".txt", prefix="llm_prompt_")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(prompt)
        except Exception:
            os.close(fd)
            raise

        try:
            cmd = [
                self.binary,
                "-m", self.model_path,
                "-f", temp_file,
                "-n", str(max_tokens),
                "-ngl", str(self.n_gpu_layers),
                "-c", str(self.n_ctx),
                "--temp", str(temperature),
                "--top-p", str(top_p),
                "--top-k", str(top_k),
                "--repeat-penalty", str(repeat_penalty),
            ]
            # Optional flags differ between llama-cli and llama-completion builds,
            # so only pass the ones this binary actually advertises in --help.
            if "--no-display-prompt" in self._help:
                cmd.append("--no-display-prompt")
            if "-e," in self._help or "--escape" in self._help:
                cmd.append("-e")
            # Disable conversation mode whenever this build advertises support.
            # New llama-completion also enables conversation mode by default.
            if self.supports_no_cnv:
                cmd.append("-no-cnv")

            popen_kwargs = {
                "stdout": subprocess.PIPE,
                # Capture stderr too: some llama.cpp builds emit the generated text
                # (or the reason for empty output) here instead of stdout.
                "stderr": subprocess.PIPE,
                # Critical: give the child an empty stdin so that if llama-cli ever
                # drops into interactive/conversation mode it gets EOF instead of
                # blocking forever waiting for keyboard input (the "hang").
                "stdin": subprocess.DEVNULL,
                "bufsize": 0,
            }
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                popen_kwargs["startupinfo"] = startupinfo

            logging.info("Running llama subprocess: %s", " ".join(cmd))
            process = subprocess.Popen(cmd, **popen_kwargs)

            result = []

            def _reader():
                buffer = b""
                while True:
                    chunk = process.stdout.read(64)
                    if not chunk:
                        break

                    buffer += chunk
                    try:
                        text = buffer.decode("utf-8")
                        buffer = b""

                        result.append(text)
                        if callback:
                            callback(text)

                    except UnicodeDecodeError:
                        for i in range(min(4, len(buffer)), 0, -1):
                            try:
                                text = buffer[:-i].decode("utf-8")
                                buffer = buffer[-i:]
                                result.append(text)
                                if callback:
                                    callback(text)
                                break
                            except UnicodeDecodeError:
                                continue

                if buffer:
                    try:
                        text = buffer.decode("utf-8", errors="replace")
                        if "<|im_end|>" not in text:
                            result.append(text)
                    except Exception:
                        pass

            stderr_chunks = []

            def _stderr_reader():
                while True:
                    chunk = process.stderr.read(4096)
                    if not chunk:
                        break
                    stderr_chunks.append(chunk)

            reader = threading.Thread(target=_reader, daemon=True)
            err_reader = threading.Thread(target=_stderr_reader, daemon=True)
            reader.start()
            err_reader.start()

            deadline = time.monotonic() + self.timeout if self.timeout else None
            while reader.is_alive():
                if deadline is not None and time.monotonic() > deadline:
                    logging.warning(
                        "llama subprocess exceeded timeout of %ss, terminating.", self.timeout
                    )
                    break
                reader.join(timeout=0.5)

            # Make sure the child is gone (timeout, stop token, or natural EOF).
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()

            err_reader.join(timeout=2)
            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

            output = self._clean_output("".join(result))

            if not output:
                # No usable stdout: surface diagnostics and try stderr as a fallback,
                # since some builds print the completion there.
                logging.warning(
                    "llama subprocess produced empty stdout (exit=%s). stderr tail:\n%s",
                    process.returncode,
                    stderr_text[-2000:],
                )
                fallback = self._clean_output(stderr_text)
                if fallback:
                    output = fallback

            return output

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class ModelCache:
    """Singleton cache for loaded LLM models."""

    _instance = None
    _store: OrderedDict = None
    _max_items: int = 2  # Keep max 2 models in memory

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._store = OrderedDict()
        return cls._instance

    def get(self, model_path: str, llama_cli_path: str = None, **kwargs) -> Any:
        """Get or load a model from cache."""
        model_path = normalize_path_for_os(model_path)
        if llama_cli_path:
            llama_cli_path = normalize_path_for_os(llama_cli_path)
        # Create cache key from path and important parameters
        key = (model_path, kwargs.get("n_gpu_layers", 99), kwargs.get("n_ctx", 32768))

        if key in self._store:
            # Move to end (LRU)
            self._store.move_to_end(key)
            logging.info(f"Using cached model: {model_path}")
            return self._store[key]

        # Load new model
        logging.info(f"Loading model: {_model_file_diagnostics(model_path)}")
        use_subprocess = kwargs.get("use_subprocess", False)

        if USE_BINDING and not use_subprocess:
            try:
                model = Llama(
                    model_path=model_path,
                    n_gpu_layers=kwargs.get("n_gpu_layers", 99),
                    n_ctx=kwargs.get("n_ctx", 32768),
                    verbose=False,
                )
            except ValueError as err:
                cli = llama_cli_path or _find_llama_cli_optional()
                if cli:
                    logging.warning(
                        "llama-cpp-python failed (%s); using llama-cli subprocess instead.",
                        err,
                    )
                    model = SubprocessModel(
                        model_path=model_path,
                        llama_cli_path=cli,
                        **kwargs,
                    )
                else:
                    raise RuntimeError(
                        f"llama-cpp-python could not load the model ({_model_file_diagnostics(model_path)}). "
                        f"Original error: {err}. "
                        f"If llama-cli.exe works for you, set win/linux llama-cli path in Load GGUF Model "
                        f"or enable 'use_subprocess'."
                    ) from err
        else:
            cli = llama_cli_path or _find_llama_cli_optional()
            if not cli:
                binary = "llama-cli.exe" if sys.platform == "win32" else "llama-cli"
                raise FileNotFoundError(
                    f"{binary} not found. Set win/linux llama-cli path or install llama-cpp-python."
                )
            model = SubprocessModel(
                model_path=model_path,
                llama_cli_path=cli,
                **kwargs,
            )

        self._store[key] = model

        # Enforce LRU limit
        while len(self._store) > self._max_items:
            old_key, old_model = self._store.popitem(last=False)
            logging.info(f"Evicting cached model: {old_key[0]}")
            # Clean up old model if possible
            if hasattr(old_model, "close"):
                old_model.close()
            del old_model

        return model

    def clear(self):
        """Clear all cached models."""
        for key, model in self._store.items():
            if hasattr(model, "close"):
                model.close()
        self._store.clear()
        logging.info("Model cache cleared")


# Global cache instance
model_cache = ModelCache()
