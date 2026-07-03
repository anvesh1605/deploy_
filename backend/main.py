from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import ALLOWED_ORIGINS, CHAT_CONVERSATIONS_PATH, FRONTEND_DIR, PORT  # noqa: E402
from backend.conversations import ConversationStore  # noqa: E402
from backend.runtime import AnswerService  # noqa: E402


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    domain: str = Field(default="auto", description="auto, release, or product")
    selected_switch: str = ""
    selected_version: str = ""
    selected_sub_version: str = ""
    show_debug: bool = False


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None
    domain: str = "auto"
    selected_switch: str = ""
    selected_version: str = ""
    selected_sub_version: str = ""


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    domain: Optional[str] = None
    selected_switch: Optional[str] = None
    selected_version: Optional[str] = None
    selected_sub_version: Optional[str] = None


service = AnswerService.create()
conversation_store = ConversationStore(CHAT_CONVERSATIONS_PATH)
app = FastAPI(title="Aruba QA Local Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/")
def root() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return FileResponse(str(ROOT / "README.md"))
    return FileResponse(index)


@app.get("/api/session")
def new_session() -> dict:
    return {"session_id": service.new_session_id()}


@app.get("/api/conversations")
def list_conversations() -> dict:
    return {"conversations": conversation_store.list_conversations()}


@app.post("/api/conversations")
def create_conversation(request: ConversationCreateRequest) -> dict:
    return conversation_store.create_conversation(
        title=request.title,
        domain=request.domain,
        selected_switch=request.selected_switch,
        selected_version=request.selected_version,
        selected_sub_version=request.selected_sub_version,
    )


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    conversation = conversation_store.get_public_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.patch("/api/conversations/{conversation_id}")
def update_conversation(conversation_id: str, request: ConversationUpdateRequest) -> dict:
    conversation = conversation_store.get_public_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if request.title is not None:
        renamed = conversation_store.rename_conversation(conversation_id, request.title)
        if renamed is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    updated = conversation_store.ensure_conversation(
        conversation_id,
        domain=request.domain if request.domain is not None else conversation.get("domain", "auto"),
        selected_switch=request.selected_switch if request.selected_switch is not None else conversation.get("selected_switch", ""),
        selected_version=request.selected_version if request.selected_version is not None else conversation.get("selected_version", ""),
        selected_sub_version=request.selected_sub_version if request.selected_sub_version is not None else conversation.get("selected_sub_version", ""),
    )
    return updated


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    deleted = conversation_store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "conversation_id": conversation_id}


@app.get("/api/health")
def health() -> dict:
    return service.health()


@app.get("/api/models")
def models() -> dict:
    return service.health()


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    conversation_id = request.conversation_id or request.session_id
    if conversation_id:
        stored_session = conversation_store.restore_session_state(conversation_id)
        if stored_session is not None:
            service.sessions[conversation_id] = stored_session
    else:
        conversation_id = service.new_session_id()

    result = service.chat(
        question=request.question,
        session_id=conversation_id,
        domain=request.domain,
        selected_switch=request.selected_switch,
        selected_version=request.selected_version,
        selected_sub_version=request.selected_sub_version,
        show_debug=request.show_debug,
    )
    conversation_summary = conversation_store.append_turn(
        conversation_id,
        question=request.question,
        result=result,
        session_state=service.sessions.get(conversation_id),
        domain=result.get("domain") or request.domain,
        selected_switch=request.selected_switch,
        selected_version=request.selected_version,
        selected_sub_version=request.selected_sub_version,
    )
    result["conversation_id"] = conversation_id
    result["conversation_title"] = conversation_summary.get("title")
    result["conversation_updated_at"] = conversation_summary.get("updated_at")
    result["conversation_summary"] = conversation_summary
    return result


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    icon = FRONTEND_DIR / "favicon.ico"
    if icon.exists():
        return FileResponse(icon)
    return FileResponse(str(FRONTEND_DIR / "index.html"))


def main() -> None:
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=PORT, reload=False)


if __name__ == "__main__":
    main()
