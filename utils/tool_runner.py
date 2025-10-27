import importlib
import multiprocessing as mp
import traceback
from typing import Any, Dict, Optional


def _worker(module_path: str, attr_name: str, kwargs: Dict[str, Any], out_q: mp.Queue) -> None:
    try:
        mod = importlib.import_module(module_path)
        attr = getattr(mod, attr_name)
        # LangChain Tool objects expose .invoke; raw functions can be called directly
        if hasattr(attr, "invoke"):
            result = attr.invoke(kwargs or {})
        else:
            result = attr(**(kwargs or {}))
        out_q.put({"ok": True, "result": result})
    except Exception:
        out_q.put({"ok": False, "error": traceback.format_exc()})


def run_tool_in_subprocess(module_path: str, attr_name: str, kwargs: Optional[Dict[str, Any]] = None, timeout_s: Optional[float] = None) -> Any:
    """
    Execute a tool/function in a separate process with optional timeout.

    Returns the tool's result on success; raises TimeoutError on timeout;
    raises RuntimeError with original traceback on failure.
    """
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    proc = ctx.Process(target=_worker, args=(module_path, attr_name, kwargs or {}, out_q))
    try:
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(1)
            raise TimeoutError(f"Tool {module_path}.{attr_name} timed out after {timeout_s}s")

        try:
            msg = out_q.get_nowait()
        except Exception:
            raise RuntimeError(f"Tool {module_path}.{attr_name} failed without output")

        if not msg.get("ok"):
            raise RuntimeError(msg.get("error", "Unknown error"))

        return msg.get("result")
    finally:
        try:
            out_q.close()
        except Exception:
            pass
        try:
            out_q.join_thread()
        except Exception:
            pass
