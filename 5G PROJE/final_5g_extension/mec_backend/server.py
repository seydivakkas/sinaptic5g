"""SİNAPTİC5G canlı kontrol, kimlik, QoD ve telemetri BFF'i.

REST kontrol düzlemidir. Yalnız SDP signaling taşır; telemetri GPU servisinde,
sürekli medya ise doğrudan Android-GPU WebRTC peer bağlantısındadır.
"""

import time
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import redis.asyncio as aioredis
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException, Depends, Header, Query, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qod_engine"))

from qod_client import CamaraQodAdapter, FlowDescriptor
from qod_orchestrator import (
    BenefitSignals,
    CalibratedBenefitModel,
    QodOrchestrator,
)
from number_verification import NumberVerificationClient
from webrtc_signaling import (
    RedisSignalingMailbox,
    SignalingConflictError,
    SignalingExpiredError,
)
from config import cfg
from auth_utils import create_access_token, get_current_device

logger = logging.getLogger(__name__)


def _load_ice_servers(raw_json: str) -> list[dict]:
    """Parse and validate runtime ICE server configuration once."""
    try:
        ice_servers = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError("WEBRTC_ICE_SERVERS_JSON must be valid JSON") from exc
    if not isinstance(ice_servers, list):
        raise ValueError("WEBRTC_ICE_SERVERS_JSON must decode to a list")
    return ice_servers


def _live_health_payload(*, status: str, readiness: str, redis_ok: bool) -> dict:
    configuration_issues = getattr(app.state, "configuration_issues", [])
    ice_issue = getattr(app.state, "ice_configuration_issue", None)
    qod = getattr(app.state, "qod", None)
    signaling_ready = getattr(app.state, "signaling", None) is not None
    return {
        "mode": "live_5g",
        "status": status,
        "readiness": readiness,
        "timestamp": time.time(),
        "redis": "connected" if redis_ok else "disconnected",
        "pipeline": "external_gpu_media_service",
        "configuration_issue_count": len(configuration_issues) + (1 if ice_issue else 0),
        "capabilities": {
            "qod_benefit_model": "measured" if qod and qod.model else "unavailable",
            "webrtc_signaling": "ready" if signaling_ready else "unavailable",
            "webrtc_media_plane": "peer_to_peer_only",
            "webrtc_ice_configuration": "ready" if ice_issue is None else "unavailable",
        },
        "dependencies": {
            "redis": "connected" if redis_ok else "disconnected",
            "webrtc_signaling": "ready" if signaling_ready else "unavailable",
            "webrtc_ice_configuration": "ready" if ice_issue is None else "unavailable",
        },
    }

# ─── Yaşam Döngüsü ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama başlangıç ve bitiş olayları.

    Bu süreç yalnız kontrol/auth/QoD/telemetri BFF'idir. GPU modelleri ve
    WebRTC medya alıcısı ayrı servis sınırındadır.
    """
    logger.info("[START] Sinaptic5G Backend başlatılıyor...")

    app.state.configuration_issues = cfg.validate()
    try:
        app.state.ice_servers = _load_ice_servers(cfg.media.ice_servers_json)
        app.state.ice_configuration_issue = None
    except ValueError as exc:
        app.state.ice_servers = []
        app.state.ice_configuration_issue = str(exc)
        app.state.configuration_issues.append("WEBRTC_ICE_SERVERS_JSON invalid — runtime ICE config unavailable")
    for issue in app.state.configuration_issues:
        logger.warning("Configuration readiness issue: %s", issue)

    # API İstemcileri
    qod_adapter = CamaraQodAdapter(
        base_url=cfg.camara.qod_base_url,
        timeout_seconds=cfg.camara.api_timeout,
    )
    benefit_model = None
    benefit_path = Path(cfg.camara.benefit_model_path)
    if benefit_path.is_file():
        try:
            benefit_model = CalibratedBenefitModel.load(benefit_path)
            logger.info("QoD benefit model loaded: run_id=%s", benefit_model.run_id)
        except Exception as exc:
            logger.warning("QoD benefit model rejected: %s", exc)
    else:
        logger.warning("QoD disabled until a measured benefit artifact is installed")
    app.state.qod_client = qod_adapter
    app.state.qod = QodOrchestrator(
        qod_adapter,
        benefit_model,
        flow_descriptor=FlowDescriptor(
            application_server_ipv4=cfg.camara.application_server_ipv4,
            application_server_port=cfg.camara.application_server_port,
            device_port=cfg.camara.device_port,
            protocol=cfg.camara.transport_protocol,
        ),
        qos_profile=cfg.camara.qos_profile,
        duration_seconds=cfg.camara.qod_duration_seconds,
        notification_sink=cfg.camara.notification_url or None,
    )
    app.state.num_verify  = NumberVerificationClient()

    # Redis Bağlantısı
    app.state.signaling = None
    try:
        app.state.redis = await aioredis.from_url(cfg.database.redis_url)
        await app.state.redis.ping()
        app.state.signaling = RedisSignalingMailbox(
            app.state.redis,
            ttl_seconds=cfg.media.signaling_ttl_seconds,
        )
        logger.info(f"[PASS] Redis bağlantısı: {cfg.database.redis_url}")
    except Exception as e:
        logger.warning(f"Redis bağlantısı kurulamadı ({e}) — oturumsuz çalışılıyor")
        app.state.redis = None

    logger.info("[READY] Sinaptic5G Backend hazır!")

    yield  # Uygulama çalışıyor

    # Temiz kapatma
    logger.info("Backend kapatılıyor...")
    if app.state.redis:
        await app.state.redis.aclose()


# ─── FastAPI Uygulaması ───────────────────────────────────────────────────────

app = FastAPI(
    title       = "Sinaptic5G API",
    description = "5G & Yapay Zeka ile Akıllı Yol Güvenliği — TEKNOFEST 2026",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = cfg.server.allowed_origins,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

security = HTTPBearer(auto_error=False)

# ─── Statik Dosya Sunumu ───────────────────────────────────────────────────────
# Demo web arayüzünü '/' adresinden erişilebilir hale getir
app.mount("/demo", StaticFiles(directory="demo"), name="demo")

@app.get("/")
async def root_redirect():
    """Ana dizini demo arayüzüne yönlendirir."""
    return RedirectResponse(url="/demo/index.html")


# ─── Pydantic Şemaları ────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    phone_number: str   # E.164 formatı: +905xxxxxxxxx
    device_id: str


class VerifyResponse(BaseModel):
    verified:     bool
    phone_number: Optional[str] = None
    latency_ms:   float = 0.0
    app_session_token: Optional[str] = None


class QoDStartResponse(BaseModel):
    session_id: Optional[str] = None
    status:     str
    state: str
    reason: Optional[str] = None
    qos_profile: Optional[str] = None
    expected_benefit: float = 0.0
    benefit_run_id: Optional[str] = None


class QoDExtendRequest(BaseModel):
    device_id: str
    additional_duration: int


class QoDStartRequest(BaseModel):
    device_id: str
    phone_number: str
    target_present: bool
    vehicle_is_approaching: bool
    recognizability_gap: float
    model_uncertainty: float
    media_degradation: float
    network_degradation: float


class WebRtcOfferRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)
    type: Literal["offer"]
    sdp: str = Field(min_length=1, max_length=65_536)


class WebRtcOfferResponse(BaseModel):
    signaling_id: str
    expires_in: int


class WebRtcClaimedOffer(BaseModel):
    signaling_id: str
    device_id: str
    type: Literal["offer"] = "offer"
    sdp: str
    expires_at: float


class WebRtcAnswerRequest(BaseModel):
    signaling_id: str = Field(min_length=1, max_length=128)
    type: Literal["answer"]
    sdp: str = Field(min_length=1, max_length=65_536)


# ─── REST Endpoint'leri ───────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü."""
    redis_ok = False
    redis = getattr(app.state, "redis", None)
    if redis:
        try:
            await redis.ping()
            redis_ok = True
        except Exception:
            pass
    readiness = "ready" if len(getattr(app.state, "configuration_issues", [])) == 0 and getattr(app.state, "ice_configuration_issue", None) is None else "degraded"
    return _live_health_payload(status="ok", readiness=readiness, redis_ok=redis_ok)


@app.get("/health/live")
async def live_health_check():
    """Minimal liveness probe for container orchestration."""
    return {"mode": "live_5g", "status": "ok", "timestamp": time.time()}


@app.get("/health/ready")
async def ready_health_check():
    """Readiness probe with dependency detail for deployment gates."""
    redis_ok = False
    redis = getattr(app.state, "redis", None)
    if redis:
        try:
            await redis.ping()
            redis_ok = True
        except Exception:
            pass
    readiness = "ready" if len(getattr(app.state, "configuration_issues", [])) == 0 and getattr(app.state, "ice_configuration_issue", None) is None and redis_ok else "degraded"
    return _live_health_payload(status="ok", readiness=readiness, redis_ok=redis_ok)


def require_media_service(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> None:
    supplied = credentials.credentials if credentials else ""
    if not cfg.media.service_token or not secrets.compare_digest(supplied, cfg.media.service_token):
        raise HTTPException(status_code=401, detail="valid media service token required")


def signaling_mailbox() -> RedisSignalingMailbox:
    mailbox = app.state.signaling
    if mailbox is None:
        raise HTTPException(status_code=503, detail="signaling_unavailable")
    return mailbox


@app.get("/webrtc/config")
async def get_webrtc_config(_: dict = Depends(get_current_device)):
    """Return runtime ICE configuration; credentials are never built into the APK."""
    if getattr(app.state, "ice_configuration_issue", None):
        raise HTTPException(status_code=503, detail="ice_configuration_unavailable")
    ice_servers = getattr(app.state, "ice_servers", None)
    if ice_servers is None:
        raise HTTPException(status_code=503, detail="ice_configuration_unavailable")
    return {"ice_servers": ice_servers}


@app.post("/webrtc/offer", response_model=WebRtcOfferResponse)
async def create_webrtc_offer(
    request: WebRtcOfferRequest,
    x_device_id: str = Header(alias="X-Device-Id"),
    device: dict = Depends(get_current_device),
):
    if device.get("device_id") != request.device_id or x_device_id != request.device_id:
        raise HTTPException(status_code=403, detail="device token mismatch")
    try:
        offer = await signaling_mailbox().create_offer(device_id=request.device_id, sdp=request.sdp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return WebRtcOfferResponse(
        signaling_id=offer.signaling_id,
        expires_in=cfg.media.signaling_ttl_seconds,
    )


@app.get("/webrtc/offer", response_model=WebRtcClaimedOffer)
async def claim_webrtc_offer(
    wait_seconds: int = Query(default=0, ge=0, le=20),
    _: None = Depends(require_media_service),
):
    offer = await signaling_mailbox().claim_offer(wait_seconds=wait_seconds)
    if offer is None:
        return Response(status_code=204)
    return WebRtcClaimedOffer(
        signaling_id=offer.signaling_id,
        device_id=offer.device_id,
        sdp=offer.sdp,
        expires_at=offer.expires_at,
    )


@app.post("/webrtc/answer", status_code=201)
async def create_webrtc_answer(
    request: WebRtcAnswerRequest,
    _: None = Depends(require_media_service),
):
    try:
        answer = await signaling_mailbox().put_answer(
            signaling_id=request.signaling_id,
            sdp=request.sdp,
        )
    except SignalingExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except SignalingConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"signaling_id": answer.signaling_id, "expires_at": answer.expires_at}


@app.get("/webrtc/answer")
async def consume_webrtc_answer(
    signaling_id: str = Query(min_length=1, max_length=128),
    device: dict = Depends(get_current_device),
):
    try:
        answer = await signaling_mailbox().consume_answer(
            signaling_id=signaling_id,
            device_id=str(device["device_id"]),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except SignalingExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except SignalingConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if answer is None:
        return JSONResponse(status_code=202, content={"status": "pending"})
    return {
        "signaling_id": answer.signaling_id,
        "type": "answer",
        "sdp": answer.sdp,
        "expires_at": answer.expires_at,
    }


@app.post("/auth/verify", response_model=VerifyResponse)
async def verify_number(
    request: VerifyRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Number Verification API ile telefon numarası doğrular.

    SMS kodu gerekmez; SIM eşleşmesi ile sessizce doğrulama yapılır.
    Yalnızca mobil veri bağlantısında (5G/4G) çalışır.

    İstemcinin OIDC Authorization Code + PKCE ile aldığı kullanıcı tokenı
    zorunludur; backend bu akış için client-credentials tokenı üretmez.
    """
    start = time.perf_counter()
    client: NumberVerificationClient = app.state.num_verify

    try:
        if credentials is None or not credentials.credentials:
            raise HTTPException(status_code=401, detail="User-bound access token required")
        verified = await client.verify(
            phone_number = request.phone_number,
            access_token = credentials.credentials,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return VerifyResponse(
            verified     = verified,
            phone_number = request.phone_number if verified else None,
            latency_ms   = round(latency_ms, 1),
            app_session_token = create_access_token(
                {"device_id": request.device_id, "phone_number": request.phone_number},
                expires_in=900,
            ) if verified else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Number Verification failed: %s", type(e).__name__)
        raise HTTPException(
            status_code = 502,
            detail      = "Number Verification API unavailable"
        )


@app.post("/qod/start", response_model=QoDStartResponse)
async def start_qod_session(
    request: QoDStartRequest,
    device: dict = Depends(get_current_device),
):
    """
    QoD oturumu başlatır.
    Sağlayıcının opaque qosProfile değeriyle oturum ister. QoD faydası
    ölçülmüş model tarafından onaylanmıyorsa Best Effort devam eder.
    """
    redis_lock_key = f"qod_request_lock:{request.device_id}"
    redis_lock_token = secrets.token_urlsafe(24)
    redis_lock_acquired = False
    try:
        if device.get("device_id") != request.device_id:
            raise HTTPException(status_code=403, detail="device token mismatch")
        if app.state.redis:
            try:
                redis_lock_acquired = bool(await app.state.redis.set(
                    redis_lock_key,
                    redis_lock_token,
                    nx=True,
                    ex=60,
                ))
                if not redis_lock_acquired:
                    return QoDStartResponse(
                        status="best_effort",
                        state="REQUESTING",
                        reason="request_in_progress",
                    )
            except Exception as exc:
                logger.warning(
                    "Redis idempotency lock unavailable; reconciling with provider: %s",
                    exc,
                )
        orchestrator: QodOrchestrator = app.state.qod
        result = await orchestrator.request(
            request.device_id,
            request.phone_number,
            BenefitSignals(
                target_present=request.target_present,
                vehicle_is_approaching=request.vehicle_is_approaching,
                recognizability_gap=request.recognizability_gap,
                model_uncertainty=request.model_uncertainty,
                media_degradation=request.media_degradation,
                network_degradation=request.network_degradation,
            ),
        )
        if app.state.redis and result.get("session_id"):
            try:
                await app.state.redis.setex(
                    f"qod_session:{request.device_id}",
                    cfg.camara.qod_duration_seconds,
                    result["session_id"],
                )
            except Exception as exc:
                logger.warning("Redis QoD mirror update failed; provider remains authoritative: %s", exc)
        return QoDStartResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("QoD start failed: %s", type(e).__name__)
        raise HTTPException(status_code=502, detail="QoD API unavailable")
    finally:
        if app.state.redis and redis_lock_acquired:
            try:
                await app.state.redis.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) else return 0 end",
                    1,
                    redis_lock_key,
                    redis_lock_token,
                )
            except Exception as exc:
                logger.warning("Redis idempotency lock release failed: %s", exc)


@app.delete("/qod/{session_id}")
async def stop_qod_session(
    session_id: str,
    device_id: str,
    device: dict = Depends(get_current_device),
):
    """
    QoD oturumunu sonlandırır.
    Şebeke kaynakları serbest bırakılır; sistem Best Effort moduna döner.
    """
    try:
        if device.get("device_id") != device_id:
            raise HTTPException(status_code=403, detail="device token mismatch")
        orchestrator: QodOrchestrator = app.state.qod
        deleted = await orchestrator.stop(device_id, session_id)

        if app.state.redis:
            try:
                await app.state.redis.delete(f"qod_session:{device_id}")
            except Exception as exc:
                logger.warning("Redis QoD mirror delete failed; provider remains authoritative: %s", exc)

        return {
            "status":     "deleted" if deleted else "not_found",
            "session_id": session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("QoD delete failed: %s", type(e).__name__)
        raise HTTPException(status_code=502, detail="QoD delete unavailable")


@app.get("/qod/{session_id}/status")
async def get_qod_status(
    session_id: str,
    device_id: str,
    device: dict = Depends(get_current_device),
):
    """QoD oturum durumunu sorgular."""
    if device.get("device_id") != device_id:
        raise HTTPException(status_code=403, detail="device token mismatch")
    orchestrator: QodOrchestrator = app.state.qod
    if not orchestrator.owns_session(device_id, session_id):
        raise HTTPException(status_code=404, detail="session not owned by device")
    qod: CamaraQodAdapter = app.state.qod_client
    status = await qod.get_session(session_id)
    return status


@app.post("/qod/{session_id}/extend")
async def extend_qod_session(
    session_id: str,
    request: QoDExtendRequest,
    device: dict = Depends(get_current_device),
):
    if device.get("device_id") != request.device_id:
        raise HTTPException(status_code=403, detail="device token mismatch")
    orchestrator: QodOrchestrator = app.state.qod
    try:
        return await orchestrator.extend(
            request.device_id,
            session_id,
            request.additional_duration,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ─── Media boundary ───────────────────────────────────────────────────────────

# Continuous media and result telemetry are not accepted by this BFF. They are
# owned by the separately deployed GPU media service.
# ─── İstatistik Endpoint'i ────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    """Sistem istatistikleri (debug amaçlı)."""
    redis_info = {}
    if app.state.redis:
        try:
            # Aktif QoD oturumlarını say
            keys = await app.state.redis.keys("qod_session:*")
            redis_info["active_qod_sessions"] = len(keys)
        except Exception:
            pass

    return {
        "timestamp": time.time(),
        "media_plane": "not_hosted_by_bff",
        "redis": redis_info,
    }


# ─── Çalıştırma ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("=" * 60)
    print("  Sinaptic5G Backend Sunucu")
    print(f"  http://{cfg.server.host}:{cfg.server.port}")
    print("  Media: direct Android <-> GPU WebRTC (not hosted here)")
    print("=" * 60)

    config_issues = cfg.validate()
    if config_issues:
        print("\n⚠️  Yapılandırma uyarıları:")
        for issue in config_issues:
            print(f"   • {issue}")
        print()

    uvicorn.run(
        "server:app",
        host    = cfg.server.host,
        port    = cfg.server.port,
        reload  = cfg.server.reload,
        workers = cfg.server.workers,
        log_level = "info",
    )
