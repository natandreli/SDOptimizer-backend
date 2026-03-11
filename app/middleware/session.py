import secrets
from typing import Callable

from fastapi import Request, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings


class SessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for session management using secure signed cookies.

    Creates and validates session IDs for each user, storing them in HTTP-only cookies.
    Session IDs are used to isolate instance storage per user.
    """

    def __init__(self, app, secret_key: str, cookie_name: str = "session_id"):
        """
        Initialize session middleware.

        Args:
            app: FastAPI application
            secret_key: Secret key for signing session tokens
            cookie_name: Name of the session cookie
        """
        super().__init__(app)
        self.serializer = URLSafeTimedSerializer(secret_key)
        self.cookie_name = cookie_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and inject session ID.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response with session cookie set
        """
        session_id = None

        cookie_value = request.cookies.get(self.cookie_name)
        if cookie_value:
            try:
                session_id = self.serializer.loads(cookie_value, max_age=86400)
            except BadSignature:
                pass

        if not session_id:
            session_id = secrets.token_urlsafe(16)

        request.state.session_id = session_id

        response = await call_next(request)

        if not cookie_value or cookie_value != self.serializer.dumps(session_id):
            signed_session = self.serializer.dumps(session_id)
            response.set_cookie(
                key=self.cookie_name,
                value=signed_session,
                httponly=True,
                secure=settings.DEBUG is False,
                samesite="lax",
                max_age=settings.SESSION_MAX_AGE_HOURS * 3600,
            )

        return response
