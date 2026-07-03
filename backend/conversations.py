from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_title(value: str) -> str:
    title = re.sub(r"\s+", " ", str(value or "").strip())
    title = title.rstrip("?.!").strip()
    if not title:
        return "New chat"
    words = title.split()
    if len(words) > 8:
        title = " ".join(words[:8])
    if len(title) > 64:
        title = title[:61].rstrip() + "..."
    return title


def _preview_text(value: str, limit: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _default_store() -> Dict[str, Any]:
    return {"conversations": {}}


@dataclass
class ConversationStore:
    path: Path
    _lock: Lock = field(default_factory=Lock)
    _data: Dict[str, Any] = field(default_factory=_default_store)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return _default_store()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return _default_store()
        if not isinstance(payload, dict):
            return _default_store()
        payload.setdefault("conversations", {})
        if not isinstance(payload.get("conversations"), dict):
            payload["conversations"] = {}
        return payload

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, ensure_ascii=False)
        tmp_path.replace(self.path)

    def _record(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._data.setdefault("conversations", {}).get(conversation_id)

    def _ensure_record(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        domain: str = "auto",
        selected_switch: str = "",
        selected_version: str = "",
        selected_sub_version: str = "",
    ) -> Dict[str, Any]:
        conversations = self._data.setdefault("conversations", {})
        record = conversations.get(conversation_id)
        if record is None:
            record = {
                "id": conversation_id,
                "title": title or "New chat",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "domain": domain or "auto",
                "selected_switch": selected_switch or "",
                "selected_version": selected_version or "",
                "selected_sub_version": selected_sub_version or "",
                "messages": [],
                "session_state": None,
            }
            conversations[conversation_id] = record
        else:
            if title and (record.get("title") in {None, "", "New chat", "Untitled chat"}):
                record["title"] = title
            if domain is not None:
                record["domain"] = domain or "auto"
            if selected_switch is not None:
                record["selected_switch"] = selected_switch or ""
            if selected_version is not None:
                record["selected_version"] = selected_version or ""
            if selected_sub_version is not None:
                record["selected_sub_version"] = selected_sub_version or ""
        return record

    @staticmethod
    def _message_preview(messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return _preview_text(message.get("content", ""))
        for message in reversed(messages):
            if message.get("content"):
                return _preview_text(message.get("content", ""))
        return ""

    @staticmethod
    def _conversation_summary(record: Dict[str, Any]) -> Dict[str, Any]:
        messages = record.get("messages", []) or []
        last_message = messages[-1] if messages else {}
        return {
            "id": record.get("id"),
            "title": record.get("title") or "New chat",
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "domain": record.get("domain") or "auto",
            "selected_switch": record.get("selected_switch") or "",
            "selected_version": record.get("selected_version") or "",
            "selected_sub_version": record.get("selected_sub_version") or "",
            "message_count": len(messages),
            "last_message_preview": ConversationStore._message_preview(messages),
            "last_message_role": last_message.get("role"),
        }

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._data.setdefault("conversations", {}).values())
        records.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return [self._conversation_summary(record) for record in records]

    def create_conversation(
        self,
        *,
        title: Optional[str] = None,
        domain: str = "auto",
        selected_switch: str = "",
        selected_version: str = "",
        selected_sub_version: str = "",
    ) -> Dict[str, Any]:
        conversation_id = uuid4().hex
        with self._lock:
            record = self._ensure_record(
                conversation_id,
                title=title,
                domain=domain,
                selected_switch=selected_switch,
                selected_version=selected_version,
                selected_sub_version=selected_sub_version,
            )
            self._save()
            return self._conversation_summary(record)

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._record(conversation_id)
            if record is None:
                return None
            payload = deepcopy(record)
        return payload

    def get_public_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        record = self.get_conversation(conversation_id)
        if record is None:
            return None
        record.pop("session_state", None)
        return record

    def ensure_conversation(
        self,
        conversation_id: Optional[str] = None,
        *,
        title: Optional[str] = None,
        domain: str = "auto",
        selected_switch: str = "",
        selected_version: str = "",
        selected_sub_version: str = "",
    ) -> Dict[str, Any]:
        if not conversation_id:
            return self.create_conversation(
                title=title,
                domain=domain,
                selected_switch=selected_switch,
                selected_version=selected_version,
                selected_sub_version=selected_sub_version,
            )
        with self._lock:
            record = self._ensure_record(
                conversation_id,
                title=title,
                domain=domain,
                selected_switch=selected_switch,
                selected_version=selected_version,
                selected_sub_version=selected_sub_version,
            )
            self._save()
            return self._conversation_summary(record)

    def rename_conversation(self, conversation_id: str, title: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._record(conversation_id)
            if record is None:
                return None
            record["title"] = _normalize_title(title)
            record["updated_at"] = _now_iso()
            self._save()
            return self._conversation_summary(record)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            conversations = self._data.setdefault("conversations", {})
            if conversation_id not in conversations:
                return False
            del conversations[conversation_id]
            self._save()
            return True

    def set_session_state(self, conversation_id: str, session_state: Dict[str, Any]) -> None:
        with self._lock:
            record = self._ensure_record(conversation_id)
            record["session_state"] = deepcopy(session_state)
            record["updated_at"] = _now_iso()
            self._save()

    def restore_session_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._record(conversation_id)
            if record is None:
                return None
            session_state = record.get("session_state")
            if session_state is None:
                return None
            return deepcopy(session_state)

    def append_turn(
        self,
        conversation_id: str,
        *,
        question: str,
        result: Dict[str, Any],
        session_state: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        selected_switch: Optional[str] = None,
        selected_version: Optional[str] = None,
        selected_sub_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._ensure_record(conversation_id)
            messages = record.setdefault("messages", [])

            first_user_message = next((message for message in messages if message.get("role") == "user"), None)
            if first_user_message and record.get("title") in {None, "", "New chat", "Untitled chat"}:
                record["title"] = _normalize_title(first_user_message.get("content", ""))

            user_message = {
                "role": "user",
                "content": question,
                "created_at": _now_iso(),
            }
            assistant_message = {
                "role": "assistant",
                "content": result.get("final_answer")
                or result.get("qwen_answer")
                or result.get("lookup_answer")
                or result.get("answer")
                or "",
                "created_at": _now_iso(),
                "predicted_intent": result.get("predicted_intent"),
                "lookup_status": result.get("lookup_status"),
                "answer_source": result.get("answer_source"),
                "source_type": result.get("source_type"),
                "data_family": result.get("data_family"),
                "qwen_used": result.get("qwen_used"),
                "qwen_validation_passed": result.get("qwen_validation_passed"),
                "slots": result.get("slots", {}),
            }

            messages.append(user_message)
            messages.append(assistant_message)

            if session_state is not None:
                record["session_state"] = deepcopy(session_state)

            if domain is not None:
                record["domain"] = domain or "auto"
            if selected_switch is not None:
                record["selected_switch"] = selected_switch or ""
            if selected_version is not None:
                record["selected_version"] = selected_version or ""
            if selected_sub_version is not None:
                record["selected_sub_version"] = selected_sub_version or ""

            if record.get("title") in {None, "", "New chat", "Untitled chat"}:
                record["title"] = _normalize_title(question)

            record["updated_at"] = _now_iso()
            self._save()
            return self._conversation_summary(record)

