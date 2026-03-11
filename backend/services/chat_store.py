from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List
from uuid import uuid4


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatSession:
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)


class InMemoryChatStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = Lock()

    def create_session(self) -> ChatSession:
        with self._lock:
            session_id = str(uuid4())
            session = ChatSession(session_id=session_id)
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            session.messages.append(ChatMessage(role=role, content=content))

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            return list(session.messages)


chat_store = InMemoryChatStore()