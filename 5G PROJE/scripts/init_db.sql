-- ============================================================
-- Sinaptic5G — PostgreSQL Veritabani Baslatma Scripti
-- ============================================================

-- Uzanti olustur
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ─── Tespit Sonuclari ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS detections (
    id              UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id       VARCHAR(64)  NOT NULL,
    frame_id        BIGINT       NOT NULL,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    class_name      VARCHAR(32)  NOT NULL,
    confidence      FLOAT4       NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    bbox_x1         INTEGER      NOT NULL,
    bbox_y1         INTEGER      NOT NULL,
    bbox_x2         INTEGER      NOT NULL,
    bbox_y2         INTEGER      NOT NULL,
    track_id        INTEGER,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ─── Frame Analiz Sonuclari ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS frame_analyses (
    id                  UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id           VARCHAR(64)  NOT NULL,
    frame_id            BIGINT       NOT NULL,
    timestamp           TIMESTAMPTZ  NOT NULL,
    license_plate       VARCHAR(16),
    speed_kmh           FLOAT4,
    risk_score          FLOAT4       NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_level          VARCHAR(8)   NOT NULL,  -- DUSUK / ORTA / KRITIK
    qod_active          BOOLEAN      NOT NULL DEFAULT FALSE,
    processing_time_ms  FLOAT4,
    driver_is_drowsy    BOOLEAN,
    driver_is_smoking   BOOLEAN,
    driver_has_phone    BOOLEAN,
    ear_value           FLOAT4,
    mar_value           FLOAT4,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ─── QoD Oturum Gecmisi ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qod_sessions (
    id              UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      VARCHAR(128) NOT NULL UNIQUE,
    device_id       VARCHAR(64)  NOT NULL,
    phone_number    VARCHAR(20),
    qos_profile     VARCHAR(128) NOT NULL,
    status          VARCHAR(16)  NOT NULL DEFAULT 'ACTIVE',  -- ACTIVE / DELETED / EXPIRED
    trigger_reason  TEXT,
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    duration_ms     INTEGER
);

-- ─── Number Verification Loglari ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS verification_logs (
    id              UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_hash      VARCHAR(64)  NOT NULL,  -- SHA256 hash (GDPR)
    verified        BOOLEAN      NOT NULL,
    latency_ms      FLOAT4,
    error_code      VARCHAR(16),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ─── Indeksler ─────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_detections_device_time
    ON detections (device_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_frames_device_time
    ON frame_analyses (device_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_frames_risk
    ON frame_analyses (risk_score DESC)
    WHERE risk_score > 70;

CREATE INDEX IF NOT EXISTS idx_qod_sessions_device
    ON qod_sessions (device_id, started_at DESC);

-- ─── Gecmis Veri Yonetimi (30 gunluk veri sakla) ─────────────────────────────
-- Uretim ortaminda pg_partman veya cron job ile calistirilir
-- Asagidaki fonksiyon manuel tetiklenebilir:
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    DELETE FROM detections WHERE created_at < NOW() - INTERVAL '30 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    DELETE FROM frame_analyses WHERE created_at < NOW() - INTERVAL '30 days';

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
