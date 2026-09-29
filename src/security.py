import asyncio
import datetime
import json
import logging
from time import monotonic
from typing import Annotated, Any, cast

import httpx
import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.config import settings
from src.utils.formatting import utc_now_iso

from .exceptions import AuthenticationException

logger = logging.getLogger(__name__)

security = HTTPBearer(
    auto_error=False,
)


#
#  jwt params
#  all optional, used to produce tokens valid for different routes
#  hierarchy: app > user > ( session / collection )
#  routes that involve a 'name' parameter require permissions for the parent object
#  name routes are considered 'queries' as names are mutable properties
#
#  note: add routes without parameters that assume the most immediately scoped key is providing
#
class JWTParams(BaseModel):
    """
    JWT parameters used to produce tokens valid for different routes.
    Workspaces are the top level of the hierarchy -- a workspace key will
    give access to all peers/sessions/collections in that workspace.

    A session key will allow the listing and creation of messages in
    that session.

    A peer key will allow the listing and creation of peer-level messages
    and querying the peer's dialectic endpoint.

    Names shortened to minimize token size. Timestamp is included
    so that many unique tokens can be generated for the same resource.
    Note that the timestamp itself is not used for security, and can
    be omitted, such as when Honcho generates the initial admin JWT.

    Fields (all optional other than `t`):

    `t`: a string timestamp of when the JWT was created
    `exp`: when the JWT expires (optional) — a standard NumericDate claim
    `ad`: a boolean flag indicating if the JWT is an admin JWT
    `w`: (string) workspace name
    `p`: (string) peer name
    `s`: (string) session name
    """

    t: str = Field(default_factory=utc_now_iso)
    exp: datetime.datetime | None = None
    ad: bool | None = None
    w: str | None = None
    p: str | None = None
    s: str | None = None


def create_admin_jwt() -> str:
    """Create a JWT for admin operations."""
    params = JWTParams(t="", ad=True)
    key = create_jwt(params)
    return key


def create_jwt(params: JWTParams) -> str:
    """Create a JWT from the given parameters."""
    payload = {k: v for k, v in params.__dict__.items() if v is not None}
    if not settings.AUTH.JWT_SECRET:
        raise ValueError("AUTH_JWT_SECRET is not set, cannot create JWT.")
    return jwt.encode(
        payload, settings.AUTH.JWT_SECRET.encode("utf-8"), algorithm="HS256"
    )


def scope_requires_workspace(
    *, peer: str | None, session: str | None, workspace: str | None
) -> bool:
    """Return whether a peer- or session-scoped claim lacks its parent workspace.

    A peer or session scope is meaningless without a workspace: the route-level
    check cannot rule out cross-workspace use (a ``{p: "alice"}`` token would
    match ``alice`` in any workspace). Truthiness-based so empty-string claims
    count as absent. Shared by `verify_jwt` (the token-shape invariant) and the
    keys API (the creation-time guard) so the two rules cannot drift apart.

    Args:
        peer: The peer claim, if any.
        session: The session claim, if any.
        workspace: The workspace claim, if any.

    Returns:
        True when a peer/session scope is present but the workspace is not.
    """
    return bool(peer or session) and not workspace


def verify_jwt(token: str) -> JWTParams:
    """Verify a shared-secret (HS256) JWT and return the decoded parameters."""

    try:
        if not settings.AUTH.JWT_SECRET:
            raise ValueError("AUTH_JWT_SECRET is not set, cannot verify JWT.")
        decoded = jwt.decode(
            token, settings.AUTH.JWT_SECRET.encode("utf-8"), algorithms=["HS256"]
        )
        return _params_from_claims(decoded)
    except jwt.ExpiredSignatureError:
        raise AuthenticationException("JWT expired") from None
    except jwt.PyJWTError:
        raise AuthenticationException("Invalid JWT") from None


def _params_from_claims(decoded: dict[str, Any]) -> JWTParams:
    """Map verified claims onto JWTParams, enforcing the token-shape invariant."""
    params = JWTParams()
    if "t" in decoded:
        params.t = decoded["t"]
    if "ad" in decoded:
        params.ad = decoded["ad"]
    # Normalize empty-string scope claims to None so a blank `w`/`p`/`s`
    # cannot masquerade as a present claim in the checks below.
    if "w" in decoded:
        params.w = decoded["w"] or None
    if "p" in decoded:
        params.p = decoded["p"] or None
    if "s" in decoded:
        params.s = decoded["s"] or None
    # Token-shape invariant: a peer- or session-scoped token MUST also
    # carry its parent workspace, otherwise the route-level check cannot
    # rule out cross-workspace use.
    if scope_requires_workspace(peer=params.p, session=params.s, workspace=params.w):
        raise AuthenticationException(
            "Invalid JWT scope: peer/session token missing workspace"
        )
    return params


#
#  issuer-signed tokens (AUTH_JWKS_URL / AUTH_JWKS)
#
#  A token whose header `kid` names a key in the configured JWK Set is verified
#  against that key, with `iss`, `aud` and `exp` required. Any other token goes
#  to verify_jwt unchanged. The claims are Honcho's own (`w`, `p`, `s`, `ad`,
#  `t`), so authorization in auth() is identical for both kinds of token.
#
#  A fetched key set is reused until AUTH_JWKS_CACHE_TTL_SECONDS pass. After
#  that, a failed refetch rejects kid-bearing tokens (fail closed) rather than
#  serving stale keys. An unknown `kid` does not trigger a refetch, so an
#  issuer should publish a new key at least one TTL before signing with it.
#

# After a failed fetch, requests fail fast for this long instead of each one
# queueing behind another fetch of an unreachable key set.
_JWKS_FETCH_RETRY_SECONDS = 5.0
_JWKS_FETCH_TIMEOUT_SECONDS = 5.0


class _JWKSCache:
    def __init__(self) -> None:
        self.source: str | None = None
        self.key_set: jwt.PyJWKSet | None = None
        self.fetched_at: float = 0.0
        self.failed_at: float | None = None
        self.lock: asyncio.Lock = asyncio.Lock()


_jwks_cache = _JWKSCache()


def parse_jwk_set(data: object) -> jwt.PyJWKSet:
    """Build a JWK Set from a decoded JSON document.

    Raises:
        ValueError: When the document is not a JWK Set with a usable key.
    """
    if not isinstance(data, dict):
        raise ValueError("JWK Set must be a JSON object")
    try:
        return jwt.PyJWKSet.from_dict(cast(dict[str, Any], data))
    except jwt.PyJWTError as exc:
        raise ValueError(str(exc)) from None


async def _fetch_jwks(url: str) -> jwt.PyJWKSet:
    async with httpx.AsyncClient(timeout=_JWKS_FETCH_TIMEOUT_SECONDS) as client:
        response = await client.get(url)
        response.raise_for_status()
        return parse_jwk_set(response.json())


async def get_jwk_set() -> jwt.PyJWKSet:
    """Return the configured JWK Set, fetching or reusing the cached copy.

    Raises:
        AuthenticationException: When the key set cannot be loaded.
    """
    auth_settings = settings.AUTH
    source = auth_settings.JWKS or auth_settings.JWKS_URL
    cache = _jwks_cache
    if cache.source != source:
        cache.source, cache.key_set, cache.failed_at = source, None, None

    if auth_settings.JWKS:
        if cache.key_set is None:
            try:
                cache.key_set = parse_jwk_set(json.loads(auth_settings.JWKS))
            except ValueError:
                logger.error("AUTH_JWKS is not a valid JWK Set")
                raise AuthenticationException("JWT signing keys unavailable") from None
        return cache.key_set

    url = auth_settings.JWKS_URL
    if url is None:
        raise AuthenticationException("JWT signing keys unavailable")
    ttl = auth_settings.JWKS_CACHE_TTL_SECONDS

    def fresh() -> jwt.PyJWKSet | None:
        if cache.key_set is not None and monotonic() - cache.fetched_at < ttl:
            return cache.key_set
        return None

    if (key_set := fresh()) is not None:
        return key_set
    async with cache.lock:
        if (key_set := fresh()) is not None:
            return key_set
        now = monotonic()
        if (
            cache.failed_at is not None
            and now - cache.failed_at < _JWKS_FETCH_RETRY_SECONDS
        ):
            raise AuthenticationException("JWT signing keys unavailable")
        try:
            key_set = await _fetch_jwks(url)
        except (httpx.HTTPError, ValueError) as exc:
            cache.failed_at = monotonic()
            logger.warning("Failed to fetch JWKS from AUTH_JWKS_URL: %s", exc)
            raise AuthenticationException("JWT signing keys unavailable") from None
        cache.key_set, cache.fetched_at, cache.failed_at = key_set, now, None
        return key_set


def _verify_issuer_jwt(token: str, key: jwt.PyJWK) -> JWTParams:
    auth_settings = settings.AUTH
    algorithms = auth_settings.JWKS_ALGORITHMS
    try:
        if key.algorithm_name not in algorithms:
            raise jwt.InvalidAlgorithmError("Key algorithm is not allowed")
        decoded = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience=auth_settings.JWKS_AUDIENCE,
            issuer=auth_settings.JWKS_ISSUER,
            options={"require": ["exp", "iss", "aud"]},
        )
        return _params_from_claims(decoded)
    except jwt.ExpiredSignatureError:
        raise AuthenticationException("JWT expired") from None
    except jwt.PyJWTError:
        raise AuthenticationException("Invalid JWT") from None


async def verify_token(token: str) -> JWTParams:
    """Verify a bearer token, issuer-signed or shared-secret.

    When a JWK Set is configured and the token's header `kid` names one of its
    keys, the token is verified against that key. Otherwise it is verified as
    a shared-secret token by `verify_jwt`.
    """
    if settings.AUTH.jwks_enabled:
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError:
            raise AuthenticationException("Invalid JWT") from None
        if kid:
            key_set = await get_jwk_set()
            key = next((k for k in key_set.keys if k.key_id == kid), None)
            if key is not None:
                return _verify_issuer_jwt(token, key)
    return verify_jwt(token)


def require_auth(
    admin: bool | None = None,
    workspace_name: str | None = None,
    peer_name: str | None = None,
    session_name: str | None = None,
    allow_member_read: bool = False,
):
    """
    Generate a dependency that requires authentication for the given parameters.

    Set `allow_member_read=True` on read-only session routes to additionally
    grant access to peer-scoped keys whose peer is an active member of the
    session. Never set it on routes that mutate state.
    """

    async def auth_dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ):
        workspace_name_param = (
            request.path_params.get(workspace_name)
            or request.query_params.get(workspace_name)
            if workspace_name
            else None
        )
        peer_name_param = (
            request.path_params.get(peer_name) or request.query_params.get(peer_name)
            if peer_name
            else None
        )
        session_name_param = (
            request.path_params.get(session_name)
            or request.query_params.get(session_name)
            if session_name
            else None
        )

        return await auth(
            credentials=credentials,
            admin=admin,
            workspace_name=workspace_name_param,
            peer_name=peer_name_param,
            session_name=session_name_param,
            allow_member_read=allow_member_read,
        )

    # Tag the closure so route-policy tests can introspect which routes opt into
    # member read without re-deriving it from HTTP method (an unreliable
    # read/write signal here — some read routes use POST for a richer body).
    auth_dependency.honcho_allow_member_read = allow_member_read  # pyright: ignore[reportFunctionMemberAccess]

    return auth_dependency


async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    admin: bool | None = None,
    workspace_name: str | None = None,
    peer_name: str | None = None,
    session_name: str | None = None,
    allow_member_read: bool = False,
) -> JWTParams:
    """Authenticate the given JWT and return the decoded parameters."""
    if not settings.AUTH.USE_AUTH:
        return JWTParams(t="", ad=True)
    if not credentials or not credentials.credentials:
        logger.warning("No access token provided")
        raise AuthenticationException("No access token provided")

    jwt_params = await verify_token(credentials.credentials)

    # Authorize by the token's narrowest scope, not by the route's. A
    # narrower-than-workspace token must NOT fall back to workspace access:
    # `{w: ws, p: alice}` may only act on `alice`, never on a sibling peer.
    if jwt_params.ad:
        return jwt_params
    if admin:
        raise AuthenticationException("Resource requires admin privileges")

    if not any([session_name, peer_name, workspace_name]):
        # Self-authorizing routes decode the token here and compare the claims
        # against body/path data inside the handler. This is needed for routes
        # whose resource identifier is not available to require_auth().
        return jwt_params

    # Every scoped, non-admin path requires the token's workspace to match the
    # route's. Check it once here so no individual branch below can forget it
    # and silently re-open cross-workspace access (the bug this module fixes).
    if workspace_name and jwt_params.w != workspace_name:
        raise AuthenticationException("JWT not permissioned for this resource")

    if jwt_params.s is not None:
        # Session-scoped token: confined to its own session. It gets no
        # cross-scope access to peer routes.
        if not session_name or jwt_params.s != session_name:
            raise AuthenticationException("JWT not permissioned for this resource")
        return jwt_params

    if jwt_params.p is not None:
        # Peer-scoped token: its own peer routes...
        if peer_name and jwt_params.p == peer_name:
            return jwt_params
        # ...plus read-only access to the sessions the peer is a member of.
        # Gated on `allow_member_read` so only read routes opt in; writes stay
        # denied. Requires the route's workspace so the membership lookup is
        # scoped (every session route declares workspace_name); the workspace
        # match itself was already verified above.
        if allow_member_read and session_name and workspace_name:
            # Lazy imports avoid an import cycle with the crud/db layers and
            # keep this DB round-trip off the common (same-scope) auth path.
            from src.crud.session import is_peer_in_session
            from src.dependencies import tracked_db

            # Membership is read on a separate committed-only (read_only)
            # connection, so a peer added to the session in a not-yet-committed
            # transaction reads as a non-member: writes must commit before a
            # member-scoped read. Fails closed.
            async with tracked_db(
                "auth.is_peer_in_session", read_only=True
            ) as member_db:
                is_member = await is_peer_in_session(
                    member_db, workspace_name, session_name, jwt_params.p
                )
            if is_member:
                return jwt_params
        raise AuthenticationException("JWT not permissioned for this resource")

    if jwt_params.w is not None:
        # Workspace tokens reach any route inside their workspace (the workspace
        # match was verified above). Routes without a declared workspace (e.g.
        # POST /v3/workspaces) self-authorize by reading jwt_params.w themselves.
        return jwt_params

    raise AuthenticationException("JWT not permissioned for this resource")
