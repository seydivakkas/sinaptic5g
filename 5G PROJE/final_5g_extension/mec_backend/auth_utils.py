# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
#
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.
#
# YASAKLAR:
#   1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
#   2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
#   3. Alt lisanslanamaz, satılamaz veya devredilemez.
#   4. Tersine mühendislik yapılamaz.
#
# İZİN VERİLEN KULLANIM:
#   - GitHub üzerinde görüntüleme ve okuma.
#   - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).
#
# YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
# İzin talepleri için: GitHub @seydivakkas

from __future__ import annotations

import os
import time
from typing import Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "super-secret-key-that-needs-to-be-secure-32-chars")

JOSE_AVAILABLE = False
try:
    from jose import jwt, JWTError
    JOSE_AVAILABLE = True
except ImportError:
    pass

security = HTTPBearer()

def create_access_token(data: dict, expires_in: int = 900) -> str:
    if not SECRET_KEY:
        raise RuntimeError("AUTH_SECRET_KEY is empty")
    
    to_encode = data.copy()
    expire = time.time() + expires_in
    to_encode.update({"exp": expire})
    
    if JOSE_AVAILABLE:
        return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    else:
        # Fallback to simple signed token if jose is not available
        import json
        import hmac
        import hashlib
        import base64
        
        payload_b64 = base64.urlsafe_b64encode(json.dumps(to_encode).encode()).decode().rstrip("=")
        sig = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
        return f"{payload_b64}.{sig_b64}"

def verify_token(token: str) -> Optional[dict]:
    if not SECRET_KEY:
        return None
    
    if JOSE_AVAILABLE:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except JWTError:
            return None
    else:
        import json
        import hmac
        import hashlib
        import base64
        
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
            payload_b64, sig_b64 = parts
            
            # Recalculate signature
            sig = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
            if not hmac.compare_digest(sig_b64, expected_sig_b64):
                return None
                
            # Decode payload
            # Fix padding
            rem = len(payload_b64) % 4
            if rem > 0:
                payload_b64 += "=" * (4 - rem)
            payload_bytes = base64.urlsafe_b64decode(payload_b64.encode())
            payload = json.loads(payload_bytes.decode())
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except Exception:
            return None

async def get_current_device(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload
