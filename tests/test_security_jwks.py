"""Issuer-signed tokens verified against a JWK Set (AUTH_JWKS_URL / AUTH_JWKS).

A token whose header `kid` names a key in the configured set is verified with
that key and must carry the expected `iss`, `aud` and an integer `exp`. Every
other token takes the shared-secret (HS256) path, which must stay unchanged.
"""

import hashlib
import hmac
import json
import time
from typing import Any

import httpx
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.security import HTTPAuthorizationCredentials
from jwt.utils import base64url_encode
from pydantic import ValidationError

from src import security
from src.config import AuthSettings, settings
from src.exceptions import AuthenticationException
from src.security import JWTParams, auth, create_jwt, verify_token

ISSUER = "https://issuer.example.test"
AUDIENCE = "https://honcho.example.test"
KID = "test-key-1"


def _ec_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _jwk(private_key: Any, kid: str, alg: str) -> dict[str, Any]:
    algorithm = pyjwt.get_algorithm_by_name(alg)
    jwk = json.loads(algorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": kid, "alg": alg, "use": "sig"})
    return jwk


def _jwks(*jwks: dict[str, Any]) -> str:
    return json.dumps({"keys": list(jwks)})


SIGNING_KEY = _ec_key()
KEY_SET = _jwks(_jwk(SIGNING_KEY, KID, "ES256"))


def _issuer_token(
    claims: dict[str, Any] | None = None,
    *,
    key: Any = SIGNING_KEY,
    kid: str | None = KID,
    algorithm: str = "ES256",
) -> str:
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 300,
        "w": "ws-a",
    }
    payload.update(claims or {})
    payload = {k: v for k, v in payload.items() if v is not None}
    headers = {"kid": kid} if kid else None
    return pyjwt.encode(payload, key, algorithm=algorithm, headers=headers)


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.fixture(autouse=True)
def _enable_auth(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(security, "_jwks_cache", security._JWKSCache())  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def inline_jwks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.AUTH, "JWKS", KEY_SET)
    monkeypatch.setattr(settings.AUTH, "JWKS_ISSUER", ISSUER)
    monkeypatch.setattr(settings.AUTH, "JWKS_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(settings.AUTH, "JWKS_ALGORITHMS", ["ES256"])


class _FakeFetch:
    """Stands in for the JWKS HTTP fetch; counts calls and can be made to fail."""

    def __init__(self, key_set: str = KEY_SET) -> None:
        self.key_set: str = key_set
        self.calls: int = 0
        self.fail: bool = False

    async def __call__(self, url: str) -> pyjwt.PyJWKSet:
        assert url == "https://issuer.example.test/.well-known/jwks.json"
        self.calls += 1
        if self.fail:
            raise httpx.ConnectError("issuer unreachable")
        return security.parse_jwk_set(json.loads(self.key_set))


class _Clock:
    def __init__(self) -> None:
        self.now: float = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def jwks_url(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeFetch, _Clock]:
    monkeypatch.setattr(
        settings.AUTH, "JWKS_URL", "https://issuer.example.test/.well-known/jwks.json"
    )
    monkeypatch.setattr(settings.AUTH, "JWKS_ISSUER", ISSUER)
    monkeypatch.setattr(settings.AUTH, "JWKS_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(settings.AUTH, "JWKS_ALGORITHMS", ["ES256"])
    monkeypatch.setattr(settings.AUTH, "JWKS_CACHE_TTL_SECONDS", 300)
    fetch = _FakeFetch()
    clock = _Clock()
    monkeypatch.setattr(security, "_fetch_jwks", fetch)
    monkeypatch.setattr(security, "monotonic", clock)
    return fetch, clock


@pytest.mark.usefixtures("inline_jwks")
class TestIssuerTokens:
    @pytest.mark.asyncio
    async def test_workspace_only_token_accepted(self):
        params = await verify_token(_issuer_token())
        assert params.w == "ws-a"
        assert params.p is None
        assert params.ad is None

    @pytest.mark.asyncio
    async def test_authorizes_like_a_shared_secret_token(self):
        creds = _bearer(_issuer_token({"p": "alice"}))
        params = await auth(credentials=creds, workspace_name="ws-a", peer_name="alice")
        assert params.p == "alice"
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-a", peer_name="bob")
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-b", peer_name="alice")

    @pytest.mark.asyncio
    async def test_wrong_audience_rejected(self):
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(_issuer_token({"aud": "https://other.example.test"}))

    @pytest.mark.asyncio
    async def test_wrong_issuer_rejected(self):
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(_issuer_token({"iss": "https://other.example.test"}))

    @pytest.mark.asyncio
    async def test_expired_rejected(self):
        with pytest.raises(AuthenticationException, match="JWT expired"):
            await verify_token(_issuer_token({"exp": int(time.time()) - 10}))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("claim", ["exp", "iss", "aud"])
    async def test_missing_required_claim_rejected(self, claim: str):
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(_issuer_token({claim: None}))

    @pytest.mark.asyncio
    async def test_unknown_kid_falls_through_and_is_rejected(self):
        # Not in the key set, so it takes the HS256 path, which refuses ES256.
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(_issuer_token(kid="not-in-the-set"))

    @pytest.mark.asyncio
    async def test_signed_by_another_key_rejected(self):
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(_issuer_token(key=_ec_key()))

    @pytest.mark.asyncio
    async def test_peer_token_without_workspace_rejected(self):
        with pytest.raises(AuthenticationException, match="missing workspace"):
            await verify_token(_issuer_token({"w": None, "p": "alice"}))

    @pytest.mark.asyncio
    async def test_hs256_token_signed_with_the_public_key_rejected(self):
        # Algorithm confusion: an HS256 token "signed" with the public key and
        # carrying its kid must not verify against the key set.
        public_pem = SIGNING_KEY.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        payload = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300}
        payload["w"] = "ws-a"
        # PyJWT refuses to HMAC with a PEM key, so sign it by hand.
        header = {"alg": "HS256", "typ": "JWT", "kid": KID}
        signing_input = b".".join(
            base64url_encode(json.dumps(part).encode()) for part in (header, payload)
        )
        signature = hmac.new(public_pem, signing_input, hashlib.sha256).digest()
        forged = (signing_input + b"." + base64url_encode(signature)).decode()
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(forged)

    @pytest.mark.asyncio
    async def test_key_with_disallowed_algorithm_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        monkeypatch.setattr(
            settings.AUTH, "JWKS", _jwks(_jwk(rsa_key, "rsa-key", "RS256"))
        )
        token = _issuer_token(key=rsa_key, kid="rsa-key", algorithm="RS256")
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(token)

    @pytest.mark.asyncio
    async def test_shared_secret_tokens_still_accepted(self):
        params = await verify_token(create_jwt(JWTParams(w="ws-a", p="alice")))
        assert (params.w, params.p) == ("ws-a", "alice")
        admin = await verify_token(create_jwt(JWTParams(t="", ad=True)))
        assert admin.ad is True

    @pytest.mark.asyncio
    async def test_malformed_token_rejected(self):
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token("not-a-jwt")


class TestWithoutJWKS:
    @pytest.mark.asyncio
    async def test_issuer_token_rejected_when_no_key_set_configured(self):
        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_token(_issuer_token())

    @pytest.mark.asyncio
    async def test_shared_secret_token_with_kid_header_unchanged(self):
        token = pyjwt.encode(
            {"w": "ws-a"}, b"test-secret", algorithm="HS256", headers={"kid": KID}
        )
        assert (await verify_token(token)).w == "ws-a"


class TestJWKSURLCache:
    @pytest.mark.asyncio
    async def test_key_set_reused_within_ttl(self, jwks_url: tuple[_FakeFetch, _Clock]):
        fetch, clock = jwks_url
        assert (await verify_token(_issuer_token())).w == "ws-a"
        clock.now += 299
        assert (await verify_token(_issuer_token())).w == "ws-a"
        assert fetch.calls == 1

    @pytest.mark.asyncio
    async def test_key_set_refetched_after_ttl(
        self, jwks_url: tuple[_FakeFetch, _Clock]
    ):
        fetch, clock = jwks_url
        await verify_token(_issuer_token())
        clock.now += 301
        await verify_token(_issuer_token())
        assert fetch.calls == 2

    @pytest.mark.asyncio
    async def test_rotated_key_picked_up_after_ttl(
        self, jwks_url: tuple[_FakeFetch, _Clock]
    ):
        fetch, clock = jwks_url
        await verify_token(_issuer_token())
        new_key = _ec_key()
        fetch.key_set = _jwks(_jwk(new_key, "test-key-2", "ES256"))
        token = _issuer_token(key=new_key, kid="test-key-2")
        # Unknown kid within the TTL does not force a refetch.
        with pytest.raises(AuthenticationException):
            await verify_token(token)
        assert fetch.calls == 1
        clock.now += 301
        assert (await verify_token(token)).w == "ws-a"

    @pytest.mark.asyncio
    async def test_cached_keys_honored_while_issuer_is_down(
        self, jwks_url: tuple[_FakeFetch, _Clock]
    ):
        fetch, clock = jwks_url
        await verify_token(_issuer_token())
        fetch.fail = True
        clock.now += 100
        assert (await verify_token(_issuer_token())).w == "ws-a"

    @pytest.mark.asyncio
    async def test_fails_closed_once_ttl_passes_and_fetch_fails(
        self, jwks_url: tuple[_FakeFetch, _Clock]
    ):
        fetch, clock = jwks_url
        await verify_token(_issuer_token())
        fetch.fail = True
        clock.now += 301
        with pytest.raises(AuthenticationException, match="signing keys unavailable"):
            await verify_token(_issuer_token())
        # Shared-secret tokens carry no kid and are unaffected by the outage.
        assert (await verify_token(create_jwt(JWTParams(w="ws-a")))).w == "ws-a"

    @pytest.mark.asyncio
    async def test_failed_fetch_not_retried_immediately(
        self, jwks_url: tuple[_FakeFetch, _Clock]
    ):
        fetch, clock = jwks_url
        fetch.fail = True
        for _ in range(3):
            with pytest.raises(AuthenticationException):
                await verify_token(_issuer_token())
        assert fetch.calls == 1
        fetch.fail = False
        clock.now += 6
        assert (await verify_token(_issuer_token())).w == "ws-a"
        assert fetch.calls == 2


class TestParseJWKSet:
    def test_accepts_a_key_set(self):
        key_set = security.parse_jwk_set(json.loads(KEY_SET))
        assert [k.key_id for k in key_set.keys] == [KID]

    @pytest.mark.parametrize("document", [[], {"keys": []}, {"keys": "x"}, "x"])
    def test_rejects_non_key_sets(self, document: object):
        with pytest.raises(ValueError):
            security.parse_jwk_set(document)


class TestAuthSettingsJWKS:
    def _settings(self, **kwargs: Any) -> AuthSettings:
        return AuthSettings(USE_AUTH=True, JWT_SECRET="test-secret", **kwargs)

    def test_defaults_leave_jwks_off(self):
        auth_settings = self._settings()
        assert auth_settings.jwks_enabled is False
        assert auth_settings.JWKS_ALGORITHMS == ["ES256"]
        assert auth_settings.JWKS_CACHE_TTL_SECONDS == 300

    def test_valid_inline_key_set(self):
        auth_settings = self._settings(
            JWKS=KEY_SET, JWKS_ISSUER=ISSUER, JWKS_AUDIENCE=AUDIENCE
        )
        assert auth_settings.jwks_enabled is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"JWKS": KEY_SET, "JWKS_AUDIENCE": AUDIENCE},
            {"JWKS_URL": "https://issuer.example.test/jwks", "JWKS_ISSUER": ISSUER},
            {
                "JWKS": KEY_SET,
                "JWKS_URL": "https://issuer.example.test/jwks",
                "JWKS_ISSUER": ISSUER,
                "JWKS_AUDIENCE": AUDIENCE,
            },
            {
                "JWKS": '{"keys": []}',
                "JWKS_ISSUER": ISSUER,
                "JWKS_AUDIENCE": AUDIENCE,
            },
            {
                "JWKS": KEY_SET,
                "JWKS_ISSUER": ISSUER,
                "JWKS_AUDIENCE": AUDIENCE,
                "JWKS_ALGORITHMS": ["HS256"],
            },
            {
                "JWKS": KEY_SET,
                "JWKS_ISSUER": ISSUER,
                "JWKS_AUDIENCE": AUDIENCE,
                "JWKS_ALGORITHMS": ["none"],
            },
            {
                "JWKS": KEY_SET,
                "JWKS_ISSUER": ISSUER,
                "JWKS_AUDIENCE": AUDIENCE,
                "JWKS_ALGORITHMS": [],
            },
        ],
    )
    def test_invalid_configurations_rejected(self, kwargs: dict[str, Any]):
        with pytest.raises(ValidationError):
            self._settings(**kwargs)

    def test_algorithms_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AUTH_JWKS_ALGORITHMS", '["ES256", "EdDSA"]')
        auth_settings = self._settings(
            JWKS=KEY_SET, JWKS_ISSUER=ISSUER, JWKS_AUDIENCE=AUDIENCE
        )
        assert auth_settings.JWKS_ALGORITHMS == ["ES256", "EdDSA"]
