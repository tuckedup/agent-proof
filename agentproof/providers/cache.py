"""Disk-based response cache for LLM provider calls.

Stores completed LLM responses on local disk keyed by the SHA-256 hash
of model identifier, message history, tool specifications, and temperature.
Provides fast retrieval during testing and simulation runs to eliminate
redundant API requests and control expenses.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


def _normalize_obj(obj: Any) -> Any:
    """Recursively convert objects to standard JSON-serializable structures."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _normalize_obj(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_obj(x) for x in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


class DiskCache:
    """Disk-based cache for LLM responses stored in .cache/litellm_cache/.

    Responses are serialized as JSON and keyed by SHA-256 hashes of the request parameters.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path(".cache") / "litellm_cache"

        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        """Filesystem directory where cached responses are stored."""
        return self._cache_dir

    def compute_key(
        self,
        model: str,
        messages: List[Any],
        tools: Optional[List[Any]] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Compute deterministic SHA-256 hash for an LLM request payload.

        Normalizes model name, messages, tools, temperature, and extra arguments.
        """
        payload: Dict[str, Any] = {
            "model": model.strip().lower(),
            "messages": _normalize_obj(messages),
            "tools": _normalize_obj(tools) if tools is not None else None,
            "temperature": round(float(temperature), 4) if temperature is not None else 0.0,
        }
        if kwargs:
            payload["kwargs"] = _normalize_obj(kwargs)

        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached response by key.

        Returns the deserialized response object or dictionary if found, None otherwise.
        Reconstructs litellm.ModelResponse if litellm is installed.
        """
        file_path = self._cache_dir / f"{key}.json"
        if not file_path.is_file():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            response_data = data.get("response", data)

            # Reconstruct litellm.ModelResponse if applicable
            if litellm is not None and isinstance(response_data, dict) and "choices" in response_data:
                try:
                    return litellm.ModelResponse(**response_data)
                except Exception:
                    pass

            return response_data
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, response: Any) -> None:
        """Store an LLM response on disk under the specified key.

        Handles litellm.ModelResponse, Pydantic models, dictionaries, and primitives.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._cache_dir / f"{key}.json"

        # Serialize response to standard JSON dictionary
        if hasattr(response, "model_dump") and callable(getattr(response, "model_dump")):
            serialized_response = response.model_dump()
        elif hasattr(response, "dict") and callable(getattr(response, "dict")):
            serialized_response = response.dict()
        elif isinstance(response, dict):
            serialized_response = _normalize_obj(response)
        else:
            serialized_response = _normalize_obj(response)

        entry = {
            "key": key,
            "response": serialized_response,
        }

        # Write to temporary file then replace for atomic persistence
        temp_file = self._cache_dir / f"{key}.tmp.{os.getpid()}"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, sort_keys=True)
            temp_file.replace(file_path)
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            # Fallback direct write
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, sort_keys=True)

    def exists(self, key: str) -> bool:
        """Check if an entry with the specified key is present in the cache."""
        file_path = self._cache_dir / f"{key}.json"
        return file_path.is_file()

    def clear(self) -> int:
        """Remove all cached entries from the cache directory.

        Returns the number of deleted cache files.
        """
        count = 0
        if not self._cache_dir.exists():
            return count

        for item in self._cache_dir.glob("*.json"):
            try:
                item.unlink()
                count += 1
            except OSError:
                pass

        for temp_item in self._cache_dir.glob("*.tmp.*"):
            try:
                temp_item.unlink()
            except OSError:
                pass

        return count

    def size(self) -> int:
        """Return the total number of cached entries currently stored on disk."""
        if not self._cache_dir.exists():
            return 0
        return len(list(self._cache_dir.glob("*.json")))
