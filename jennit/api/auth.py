"""Optional bearer-token authentication for the batch API."""
import hmac

from fastapi import HTTPException, Request


def require_token(request: Request) -> None:
    """FastAPI dependency: 401 unless Authorization: Bearer <JENNIT_API_TOKEN>.

    When no token is configured (settings.api_token is None) every request
    is allowed — the deployment is considered internal.
    """
    expected = request.app.state.settings.api_token
    if expected is None:
        return
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401,
                            detail="invalid or missing bearer token")
