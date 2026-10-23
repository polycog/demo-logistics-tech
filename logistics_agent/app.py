"""
Chat webapp core
"""

import os
import socket
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory

from logistics_agent.backend import ChatAppBackend
from logistics_agent.sessions import SessionStore

# ===

STATIC_DIR = Path(__file__).parent / "static"
INSTANCE_ID = uuid.uuid4().hex[:6]

MAX_SESSIONS: int = 100
SESSION_TTL: int = 3600

MAX_CHARS: int = 4000

# ===


def _valid_session_id(value: Any) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def create_app[T](backend: ChatAppBackend[T]) -> Flask:
    """
    produces the flask app given
    a backend
    """

    app = Flask(__name__, static_folder=None)

    # Refuse request bodies over 64 KB before reading them. This protects
    # the server; the character check below is for the user's message.
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    # ===

    store = SessionStore[T](
        backend.session_init,
        ttl_seconds=int(os.environ.get("SESSION_TTL", f"{SESSION_TTL}")),
        max_sessions=int(os.environ.get("MAX_SESSIONS", f"{MAX_SESSIONS}")),
    )

    # ===

    @app.errorhandler(413)
    def _too_large(_err: Any) -> tuple[Response, int]:
        return jsonify(error="Request is too large."), 413

    @app.get("/")
    def _index() -> Response:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/config")
    def _config() -> Response:
        return jsonify(
            title=backend.title,
            greeting=backend.greeting,
            instance=INSTANCE_ID,
            allowunsafe=backend.allow_unsafe_output,
        )

    @app.post("/api/chat")
    def _chat() -> Response | tuple[Response, int]:
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id")
        message = str(body.get("message", "")).strip()

        if not _valid_session_id(session_id):
            return jsonify(error="Missing or invalid session_id."), 400
        if not message:
            return jsonify(error="Message is empty."), 400
        if len(message) > MAX_CHARS:
            return jsonify(error=f"Message is longer than {MAX_CHARS} characters."), 400

        with store.session(str(session_id)) as (state, is_new):
            try:
                reply = backend.respond(message, state)
                return jsonify(
                    reply=str(reply), new_session=is_new, instance=INSTANCE_ID
                )
            except Exception:  # pylint: disable=broad-exception-caught
                app.logger.exception("Backend failed")
                return (
                    jsonify(
                        error="The backend hit an error responding to that message."
                    ),
                    500,
                )

    @app.post("/api/reset")
    def _reset() -> Response:
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id")
        if _valid_session_id(session_id):
            store.reset(str(session_id))
        return jsonify(ok=True)

    @app.get("/health")
    def _health() -> Response:
        return jsonify(
            ok=True,
            sessions=len(store),
            instance=INSTANCE_ID,
            host=socket.gethostname(),
        )

    return app
