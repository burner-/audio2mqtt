use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{ConnectInfo, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse,
    },
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use reqwest::Client as HttpClient;
use rumqttc::{AsyncClient, EventLoop, MqttOptions, QoS};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    fs,
    io::{Cursor, Read},
    net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream},
    path::Path,
    sync::{Arc, RwLock},
    thread,
    time::{Duration, Instant},
};
use tokio::{
    fs::OpenOptions,
    io::AsyncWriteExt,
    net::TcpListener,
    sync::{broadcast, mpsc, oneshot},
    time,
};
use tokio_stream::wrappers::BroadcastStream;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, reload, util::SubscriberInitExt, EnvFilter};
use tsclientlib::{ChannelId, ClientId, Connection as TsConnection, Identity, StreamItem};
use tsproto_packets::packets::AudioData;
use uuid::Uuid;
use voice_activity_detector::VoiceActivityDetector;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SAMPLE_RATE: usize = 16_000;
const BYTES_PER_SAMPLE: usize = 2;
const VAD_FRAME_SAMPLES: usize = 512;
const TEAMSPEAK_SAMPLE_RATE: usize = 48_000;
const TEAMSPEAK_FRAME_SAMPLES: usize = 960;

#[derive(Debug, Clone)]
struct EnvConfig {
    asr_bind: SocketAddr,
    admin_bind: SocketAddr,
    model_path: String,
    model_url: String,
    default_model_id: String,
    language: String,
    n_threads: i32,
    beam_size: i32,
    vad_threshold: f32,
    min_segment_seconds: f64,
    max_segment_seconds: f64,
    silence_cut_seconds: f64,
    pre_roll_seconds: f64,
    config_path: String,
    output_jsonl: String,
    webhook_timeout_ms: u64,
}

impl EnvConfig {
    fn from_env() -> Result<Self> {
        Ok(Self {
            asr_bind: env_or("ASR_BIND", "0.0.0.0:15000").parse()?,
            admin_bind: env_or("ADMIN_BIND", "0.0.0.0:8080").parse()?,
            model_path: env_or("MODEL_PATH", "/models/ggml-large-v3.bin"),
            model_url: env_or("MODEL_URL", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"),
            default_model_id: env_or("DEFAULT_MODEL_ID", "large-v3"),
            language: env_or("LANGUAGE", "fi"),
            n_threads: env_or("N_THREADS", "8").parse()?,
            beam_size: env_or("BEAM_SIZE", "3").parse()?,
            vad_threshold: env_or("VAD_THRESHOLD", "0.55").parse()?,
            min_segment_seconds: env_or("MIN_SEGMENT_SECONDS", "1.0").parse()?,
            max_segment_seconds: env_or("MAX_SEGMENT_SECONDS", "30.0").parse()?,
            silence_cut_seconds: env_or("SILENCE_CUT_SECONDS", "1.2").parse()?,
            pre_roll_seconds: env_or("PRE_ROLL_SECONDS", "0.4").parse()?,
            config_path: env_or("CONFIG_PATH", "/config/config.json"),
            output_jsonl: env_or("OUTPUT_JSONL", "/output/transcripts.jsonl"),
            webhook_timeout_ms: env_or("WEBHOOK_TIMEOUT_MS", "3000").parse()?,
        })
    }
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn log_filter_from_level(level: &str) -> String {
    let normalized = match level.trim().to_lowercase().as_str() {
        "error" => "error",
        "warn" | "warning" => "warn",
        "debug" => "debug",
        "trace" => "trace",
        _ => "info",
    };
    format!("{normalized},audio2mqtt={normalized}")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeConfig {
    #[serde(default)]
    mqtt: MqttConfig,
    #[serde(default)]
    webhooks: Vec<WebhookConfig>,
    #[serde(default)]
    models: ModelConfig,
    #[serde(default)]
    logging: LoggingConfig,
    #[serde(default)]
    transcription: TranscriptionConfig,
    #[serde(default)]
    teamspeak: TeamSpeakConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            mqtt: MqttConfig::default(),
            webhooks: vec![],
            models: ModelConfig::default(),
            logging: LoggingConfig::default(),
            transcription: TranscriptionConfig::default(),
            teamspeak: TeamSpeakConfig::default(),
        }
    }
}

impl RuntimeConfig {
    fn from_env(env: &EnvConfig) -> Self {
        Self {
            mqtt: MqttConfig::default(),
            webhooks: vec![],
            models: ModelConfig {
                active_model_id: env.default_model_id.clone(),
                profiles: default_model_profiles(&env.default_model_id, &env.model_path, &env.model_url),
            },
            logging: LoggingConfig::default(),
            transcription: TranscriptionConfig::from_env(env),
            teamspeak: TeamSpeakConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TeamSpeakConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_teamspeak_server_address")]
    server_address: String,
    #[serde(default)]
    server_password: Option<String>,
    #[serde(default = "default_teamspeak_nickname")]
    nickname: String,
    #[serde(default)]
    identity: String,
    #[serde(default)]
    channel_id: Option<u64>,
    #[serde(default)]
    channel_path: String,
    #[serde(default)]
    channel_password: Option<String>,
    #[serde(default = "default_teamspeak_reconnect_seconds")]
    reconnect_seconds: u64,
}

fn default_teamspeak_server_address() -> String {
    "localhost:9987".to_string()
}

fn default_teamspeak_nickname() -> String {
    "audio2mqtt".to_string()
}

fn default_teamspeak_reconnect_seconds() -> u64 {
    10
}

impl Default for TeamSpeakConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            server_address: default_teamspeak_server_address(),
            server_password: None,
            nickname: default_teamspeak_nickname(),
            identity: String::new(),
            channel_id: None,
            channel_path: String::new(),
            channel_password: None,
            reconnect_seconds: default_teamspeak_reconnect_seconds(),
        }
    }
}

impl TeamSpeakConfig {
    fn normalized(&self) -> Self {
        let mut cfg = self.clone();
        if cfg.server_address.trim().is_empty() {
            cfg.server_address = default_teamspeak_server_address();
        } else {
            cfg.server_address = cfg.server_address.trim().to_string();
        }
        if cfg.nickname.trim().is_empty() {
            cfg.nickname = default_teamspeak_nickname();
        } else {
            cfg.nickname = cfg.nickname.trim().to_string();
        }
        cfg.identity = cfg.identity.trim().to_string();
        if cfg.identity.is_empty() {
            cfg.identity = create_teamspeak_identity_string();
        }
        cfg.channel_path = cfg.channel_path.trim().trim_matches('/').to_string();
        cfg.server_password = normalize_optional_secret(cfg.server_password.as_deref());
        cfg.channel_password = normalize_optional_secret(cfg.channel_password.as_deref());
        cfg.reconnect_seconds = cfg.reconnect_seconds.clamp(1, 300);
        cfg
    }
}

fn normalize_optional_secret(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
}

fn create_teamspeak_identity_string() -> String {
    identity_to_config_string(&Identity::create()).unwrap_or_default()
}

fn identity_to_config_string(identity: &Identity) -> Result<String> {
    let value = serde_json::to_value(identity)?;
    Ok(match value {
        Value::String(s) => s,
        other => other.to_string(),
    })
}

fn parse_teamspeak_identity(value: &str) -> Result<Identity> {
    let trimmed = value.trim();
    if let Ok(identity) = Identity::new_from_str(trimmed) {
        return Ok(identity);
    }
    if let Ok(identity) = Identity::new_from_ts_str(trimmed) {
        return Ok(identity);
    }
    serde_json::from_str::<Identity>(trimmed).with_context(|| "parsing TeamSpeak identity")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LoggingConfig {
    #[serde(default = "default_log_level")]
    level: String,
    #[serde(default)]
    vad_debug: bool,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            vad_debug: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TranscriptionConfig {
    #[serde(default = "default_language")]
    language: String,
    #[serde(default = "default_n_threads")]
    n_threads: i32,
    #[serde(default = "default_beam_size")]
    beam_size: i32,
    #[serde(default = "default_vad_threshold")]
    vad_threshold: f32,
    #[serde(default = "default_min_segment_seconds")]
    min_segment_seconds: f64,
    #[serde(default = "default_max_segment_seconds")]
    max_segment_seconds: f64,
    #[serde(default = "default_silence_cut_seconds")]
    silence_cut_seconds: f64,
    #[serde(default = "default_pre_roll_seconds")]
    pre_roll_seconds: f64,
}

fn default_language() -> String {
    "fi".to_string()
}

fn default_n_threads() -> i32 {
    8
}

fn default_beam_size() -> i32 {
    3
}

fn default_vad_threshold() -> f32 {
    0.55
}

fn default_min_segment_seconds() -> f64 {
    1.0
}

fn default_max_segment_seconds() -> f64 {
    30.0
}

fn default_silence_cut_seconds() -> f64 {
    1.2
}

fn default_pre_roll_seconds() -> f64 {
    0.4
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            language: default_language(),
            n_threads: default_n_threads(),
            beam_size: default_beam_size(),
            vad_threshold: default_vad_threshold(),
            min_segment_seconds: default_min_segment_seconds(),
            max_segment_seconds: default_max_segment_seconds(),
            silence_cut_seconds: default_silence_cut_seconds(),
            pre_roll_seconds: default_pre_roll_seconds(),
        }
    }
}

impl TranscriptionConfig {
    fn from_env(env: &EnvConfig) -> Self {
        Self {
            language: env.language.clone(),
            n_threads: env.n_threads,
            beam_size: env.beam_size,
            vad_threshold: env.vad_threshold,
            min_segment_seconds: env.min_segment_seconds,
            max_segment_seconds: env.max_segment_seconds,
            silence_cut_seconds: env.silence_cut_seconds,
            pre_roll_seconds: env.pre_roll_seconds,
        }
        .normalized()
    }

    fn normalized(&self) -> Self {
        let language = if self.language.trim().is_empty() {
            default_language()
        } else {
            self.language.trim().to_string()
        };
        let n_threads = self.n_threads.max(1);
        let beam_size = self.beam_size.max(1);
        let vad_threshold = finite_f32_or(self.vad_threshold, default_vad_threshold()).clamp(0.0, 1.0);
        let min_segment_seconds = finite_f64_or(self.min_segment_seconds, default_min_segment_seconds()).max(0.1);
        let max_segment_seconds = finite_f64_or(self.max_segment_seconds, default_max_segment_seconds()).max(min_segment_seconds);
        let silence_cut_seconds = finite_f64_or(self.silence_cut_seconds, default_silence_cut_seconds()).max(0.05);
        let pre_roll_seconds = finite_f64_or(self.pre_roll_seconds, default_pre_roll_seconds()).max(0.0);

        Self {
            language,
            n_threads,
            beam_size,
            vad_threshold,
            min_segment_seconds,
            max_segment_seconds,
            silence_cut_seconds,
            pre_roll_seconds,
        }
    }
}

fn finite_f32_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn finite_f64_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MqttConfig {
    enabled: bool,
    host: String,
    port: u16,
    username: Option<String>,
    password: Option<String>,
    client_id: String,
    topic: String,
    qos: u8,
}

impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "host.docker.internal".to_string(),
            port: 1883,
            username: None,
            password: None,
            client_id: "audio2mqtt".to_string(),
            topic: "audio2mqtt/transcripts".to_string(),
            qos: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WebhookConfig {
    name: String,
    enabled: bool,
    url: String,
    bearer_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelConfig {
    active_model_id: String,
    profiles: Vec<ModelProfile>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            active_model_id: "large-v3".to_string(),
            profiles: default_model_profiles(
                "large-v3",
                "/models/ggml-large-v3.bin",
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelProfile {
    id: String,
    name: String,
    path: String,
    url: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    recommended_vram_gb: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ModelSelectRequest {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ModelActivateRequest {
    id: String,
    #[serde(default = "default_true")]
    download_if_missing: bool,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceModelAddRequest {
    repo_id: String,
    filename: Option<String>,
    id: Option<String>,
    name: Option<String>,
    #[serde(default)]
    set_active: bool,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceModelInfo {
    #[serde(default)]
    siblings: Vec<HuggingFaceSibling>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
}

fn default_true() -> bool {
    true
}

#[derive(Clone)]
struct SharedState {
    runtime_config: Arc<RwLock<RuntimeConfig>>,
    config_path: String,
    events: broadcast::Sender<String>,
    http: HttpClient,
    started_at: DateTime<Utc>,
    env: EnvConfig,
    log_reload: Arc<reload::Handle<EnvFilter, tracing_subscriber::Registry>>,
    job_tx: mpsc::Sender<TranscribeJob>,
    teamspeak: TeamSpeakControl,
}

#[derive(Clone)]
struct TeamSpeakControl {
    tx: mpsc::UnboundedSender<TeamSpeakCommand>,
    status: Arc<RwLock<TeamSpeakStatus>>,
}

#[derive(Debug)]
enum TeamSpeakCommand {
    Connect,
    Disconnect,
    Reconnect,
    Apply(TeamSpeakConfig),
}

#[derive(Debug, Clone, Serialize)]
struct TeamSpeakStatus {
    state: String,
    enabled: bool,
    connected: bool,
    server_address: String,
    active_channel_id: Option<u64>,
    active_channel_path: String,
    last_error: Option<String>,
    connected_at: Option<DateTime<Utc>>,
    channels: Vec<TeamSpeakChannelInfo>,
}

impl Default for TeamSpeakStatus {
    fn default() -> Self {
        Self {
            state: "disabled".to_string(),
            enabled: false,
            connected: false,
            server_address: default_teamspeak_server_address(),
            active_channel_id: None,
            active_channel_path: String::new(),
            last_error: None,
            connected_at: None,
            channels: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct TeamSpeakChannelInfo {
    id: u64,
    parent_id: Option<u64>,
    name: String,
    path: String,
}

#[derive(Debug, Deserialize)]
struct TeamSpeakChannelSelectRequest {
    channel_id: u64,
}

#[derive(Debug, Clone)]
struct TeamSpeakSpeakerMeta {
    client_id: u64,
    client_name: String,
    channel_id: Option<u64>,
    channel_name: String,
    channel_path: String,
}

struct TranscribeJob {
    id: Uuid,
    audio: Vec<f32>,
    start_sample: u64,
    end_sample: u64,
    received_at: DateTime<Utc>,
    source: SourceInfo,
    context: Option<Value>,
    vad: VadInfo,
    reply: Option<oneshot::Sender<Result<TranscriptEvent, String>>>,
}

#[derive(Debug, Clone, Serialize)]
struct SourceInfo {
    #[serde(rename = "type")]
    source_type: String,
    src: String,
    meta: Value,
}

#[derive(Debug, Clone, Serialize)]
struct ModelInfo {
    id: String,
    name: String,
    path: String,
}

#[derive(Debug, Clone, Serialize)]
struct AudioInfo {
    start_sec: f64,
    end_sec: f64,
    duration_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
struct VadInfo {
    enabled: bool,
    engine: String,
    threshold: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct TranscriptSegment {
    start_sec: f64,
    end_sec: f64,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
struct TranscriptResult {
    language: String,
    text: String,
    segments: Vec<TranscriptSegment>,
}

#[derive(Debug, Clone, Serialize)]
struct TranscriptEvent {
    #[serde(rename = "type")]
    event_type: String,
    schema_version: String,
    id: Uuid,
    ts: DateTime<Utc>,
    source: SourceInfo,
    context: Option<Value>,
    model: ModelInfo,
    audio: AudioInfo,
    vad: VadInfo,
    result: TranscriptResult,
}

#[derive(Debug, Deserialize)]
struct RestTranscribeRequest {
    audio: RestAudioRequest,
    context: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RestAudioRequest {
    format: String,
    encoding: String,
    data: String,
    sample_rate: Option<usize>,
    channels: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let initial_log_level = env_or("AUDIO2MQTT_LOG_LEVEL", "info");
    let (filter_layer, log_reload) = reload::Layer::new(EnvFilter::new(log_filter_from_level(&initial_log_level)));
    tracing_subscriber::registry()
        .with(filter_layer)
        .with(tracing_subscriber::fmt::layer())
        .init();

    let env = EnvConfig::from_env()?;
    let runtime_config = load_or_create_runtime_config(&env.config_path, &env)?;
    let _ = log_reload.reload(EnvFilter::new(log_filter_from_level(&runtime_config.logging.level)));
    info!(?env, "starting audio2mqtt");
    let (events_tx, _) = broadcast::channel::<String>(128);
    let (job_tx, job_rx) = mpsc::channel::<TranscribeJob>(8);
    let (transcript_tx, transcript_rx) = mpsc::unbounded_channel::<TranscriptEvent>();
    let (teamspeak_tx, teamspeak_rx) = mpsc::unbounded_channel::<TeamSpeakCommand>();
    let teamspeak_status = Arc::new(RwLock::new(TeamSpeakStatus::default()));
    let teamspeak_control = TeamSpeakControl {
        tx: teamspeak_tx.clone(),
        status: teamspeak_status.clone(),
    };

    let shared = SharedState {
        runtime_config: Arc::new(RwLock::new(runtime_config)),
        config_path: env.config_path.clone(),
        events: events_tx.clone(),
        http: HttpClient::builder()
            .timeout(Duration::from_millis(env.webhook_timeout_ms))
            .build()?,
        started_at: Utc::now(),
        env: env.clone(),
        log_reload: Arc::new(log_reload),
        job_tx: job_tx.clone(),
        teamspeak: teamspeak_control,
    };

    start_transcriber_thread(env.clone(), shared.runtime_config.clone(), job_rx, transcript_tx)?;

    tokio::spawn(run_teamspeak_manager(
        shared.runtime_config.clone(),
        job_tx.clone(),
        teamspeak_status,
        teamspeak_rx,
    ));

    let dispatcher_state = shared.clone();
    tokio::spawn(async move {
        if let Err(e) = dispatch_transcripts(dispatcher_state, transcript_rx).await {
            error!(error = %e, "transcript dispatcher stopped");
        }
    });

    let audio_env = env.clone();
    let audio_runtime_config = shared.runtime_config.clone();
    thread::Builder::new()
        .name("audio-tcp-server".to_string())
        .spawn(move || {
            if let Err(e) = run_audio_server_blocking(audio_env, audio_runtime_config, job_tx) {
                error!(error = %e, "audio server stopped");
            }
        })?;

    run_web_admin(shared).await?;
    Ok(())
}

fn load_or_create_runtime_config(path: &str, env: &EnvConfig) -> Result<RuntimeConfig> {
    let mut changed = false;
    let mut cfg = if Path::new(path).exists() {
        let raw = fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let raw_value: Value = serde_json::from_str(&raw).with_context(|| format!("parsing {path}"))?;
        let has_models = raw_value.get("models").is_some();
        let has_transcription = raw_value.get("transcription").is_some();
        let has_teamspeak = raw_value.get("teamspeak").is_some();
        let mut parsed: RuntimeConfig = serde_json::from_value(raw_value).with_context(|| format!("parsing {path}"))?;
        if !has_models {
            parsed.models = ModelConfig {
                active_model_id: env.default_model_id.clone(),
                profiles: default_model_profiles(&env.default_model_id, &env.model_path, &env.model_url),
            };
            changed = true;
        }
        if !has_transcription {
            parsed.transcription = TranscriptionConfig::from_env(env);
            changed = true;
        }
        if !has_teamspeak {
            parsed.teamspeak = TeamSpeakConfig::default();
            changed = true;
        }
        parsed
    } else {
        changed = true;
        RuntimeConfig::from_env(env)
    };

    if cfg.models.profiles.is_empty() {
        cfg.models = ModelConfig {
            active_model_id: env.default_model_id.clone(),
            profiles: default_model_profiles(&env.default_model_id, &env.model_path, &env.model_url),
        };
        changed = true;
    }
    cfg.transcription = cfg.transcription.normalized();
    let normalized_teamspeak = cfg.teamspeak.normalized();
    if serde_json::to_value(&cfg.teamspeak)? != serde_json::to_value(&normalized_teamspeak)? {
        cfg.teamspeak = normalized_teamspeak;
        changed = true;
    }
    changed |= ensure_finnish_nlp_profile(&mut cfg.models.profiles);

    if cfg.models.active_model_id.trim().is_empty()
        || !cfg.models.profiles.iter().any(|p| p.id.as_str() == cfg.models.active_model_id.as_str())
    {
        cfg.models.active_model_id = cfg
            .models
            .profiles
            .first()
            .map(|p| p.id.clone())
            .unwrap_or_else(|| env.default_model_id.clone());
        changed = true;
    }

    if changed {
        save_runtime_config(path, &cfg)?;
    }

    Ok(cfg)
}

fn ensure_finnish_nlp_profile(profiles: &mut Vec<ModelProfile>) -> bool {
    let profile = finnish_nlp_large_v3_profile();
    if profiles.iter().any(|p| p.id.as_str() == profile.id.as_str()) {
        return false;
    }
    profiles.push(profile);
    true
}

fn default_model_profiles(default_id: &str, default_path: &str, default_url: &str) -> Vec<ModelProfile> {
    let mut profiles = vec![
        ModelProfile {
            id: "large-v3".to_string(),
            name: "Whisper large-v3".to_string(),
            path: "/models/ggml-large-v3.bin".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin".to_string(),
            description: "Best default quality. RTX 3080 10/12 GB: usually works, but the 10 GB variant has less headroom.".to_string(),
            recommended_vram_gb: Some(10),
        },
        ModelProfile {
            id: "large-v3-turbo".to_string(),
            name: "Whisper large-v3-turbo".to_string(),
            path: "/models/ggml-large-v3-turbo.bin".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin".to_string(),
            description: "Faster and lighter large-v3 turbo variant.".to_string(),
            recommended_vram_gb: Some(8),
        },
        ModelProfile {
            id: "medium".to_string(),
            name: "Whisper medium".to_string(),
            path: "/models/ggml-medium.bin".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".to_string(),
            description: "Light fallback. Finnish accuracy is clearly weaker than large-v3.".to_string(),
            recommended_vram_gb: Some(5),
        },
        finnish_nlp_large_v3_profile(),
    ];

    if !profiles.iter().any(|p| p.id.as_str() == default_id) {
        profiles.insert(
            0,
            ModelProfile {
                id: default_id.to_string(),
                name: format!("Custom default: {default_id}"),
                path: default_path.to_string(),
                url: default_url.to_string(),
                description: "Default from environment".to_string(),
                recommended_vram_gb: None,
            },
        );
    }

    profiles
}

fn finnish_nlp_large_v3_profile() -> ModelProfile {
    ModelProfile {
        id: "finnish-nlp-large-v3".to_string(),
        name: "Finnish-NLP Finnish Whisper large-v3".to_string(),
        path: "/models/ggml-model-fi-large-v3.bin".to_string(),
        url: "https://huggingface.co/Finnish-NLP/Finnish-finetuned-whisper-models-ggml-format/resolve/main/ggml-model-fi-large-v3.bin".to_string(),
        description: "Finnish fine-tuned GGML large-v3 model from Finnish-NLP.".to_string(),
        recommended_vram_gb: Some(10),
    }
}

fn active_model_profile(cfg: &RuntimeConfig, env: &EnvConfig) -> ModelProfile {
    cfg.models
        .profiles
        .iter()
        .find(|p| p.id.as_str() == cfg.models.active_model_id.as_str())
        .cloned()
        .or_else(|| cfg.models.profiles.first().cloned())
        .unwrap_or_else(|| ModelProfile {
            id: env.default_model_id.clone(),
            name: env.default_model_id.clone(),
            path: env.model_path.clone(),
            url: env.model_url.clone(),
            description: "Fallback model from environment".to_string(),
            recommended_vram_gb: None,
        })
}

fn save_runtime_config(path: &str, cfg: &RuntimeConfig) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(cfg)?)?;
    Ok(())
}

fn start_transcriber_thread(
    env: EnvConfig,
    runtime_config: Arc<RwLock<RuntimeConfig>>,
    mut job_rx: mpsc::Receiver<TranscribeJob>,
    transcript_tx: mpsc::UnboundedSender<TranscriptEvent>,
) -> Result<()> {
    thread::Builder::new()
        .name("whisper-transcriber".to_string())
        .spawn(move || {
            let mut loaded_model_id: Option<String> = None;
            let mut loaded_model_path: Option<String> = None;
            let mut ctx: Option<WhisperContext> = None;

            while let Some(mut job) = job_rx.blocking_recv() {
                let reply = job.reply.take();
                let (profile, transcription) = {
                    let cfg = runtime_config.read().unwrap().clone();
                    (active_model_profile(&cfg, &env), cfg.transcription.normalized())
                };

                let must_reload = loaded_model_id.as_deref() != Some(profile.id.as_str())
                    || loaded_model_path.as_deref() != Some(profile.path.as_str())
                    || ctx.is_none();

                if must_reload {
                    if !Path::new(&profile.path).exists() {
                        let message = format!(
                            "active model file is missing: {}. Download it from web admin before using it",
                            profile.path
                        );
                        error!(model_id = %profile.id, path = %profile.path, message = %message, "active model file is missing");
                        if let Some(tx) = reply {
                            let _ = tx.send(Err(message));
                        }
                        continue;
                    }

                    info!(model_id = %profile.id, model_path = %profile.path, "loading whisper model");
                    match WhisperContext::new_with_params(&profile.path, WhisperContextParameters::default()) {
                        Ok(new_ctx) => {
                            ctx = Some(new_ctx);
                            loaded_model_id = Some(profile.id.clone());
                            loaded_model_path = Some(profile.path.clone());
                            info!(model_id = %profile.id, "whisper model loaded");
                        }
                        Err(e) => {
                            let message = format!("failed to load whisper model {}: {e}", profile.path);
                            error!(model_id = %profile.id, path = %profile.path, error = %e, "failed to load whisper model");
                            ctx = None;
                            if let Some(tx) = reply {
                                let _ = tx.send(Err(message));
                            }
                            continue;
                        }
                    }
                }

                let start = Instant::now();
                match transcribe_one(ctx.as_ref().expect("ctx loaded"), &transcription, &profile, job) {
                    Ok(event) => {
                        let elapsed = start.elapsed().as_millis();
                        info!(id = %event.id, model_id = %profile.id, elapsed_ms = elapsed, text = %event.result.text, "transcribed segment");
                        if let Some(tx) = reply {
                            let _ = tx.send(Ok(event.clone()));
                        }
                        if let Err(e) = transcript_tx.send(event) {
                            error!(error = %e, "failed to send transcript event");
                        }
                    }
                    Err(e) => {
                        let message = e.to_string();
                        error!(model_id = %profile.id, error = %e, "transcription failed");
                        if let Some(tx) = reply {
                            let _ = tx.send(Err(message));
                        }
                    }
                }
            }
        })?;
    Ok(())
}

fn transcribe_one(
    ctx: &WhisperContext,
    transcription: &TranscriptionConfig,
    profile: &ModelProfile,
    job: TranscribeJob,
) -> Result<TranscriptEvent> {
    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: transcription.beam_size,
        patience: -1.0,
    });

    let language = transcription.language.clone();
    params.set_language(Some(&language));
    params.set_n_threads(transcription.n_threads);
    params.set_translate(false);
    params.set_no_context(true);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_print_special(false);

    state.full(params, &job.audio)?;

    let base_start_sec = job.start_sample as f64 / SAMPLE_RATE as f64;
    let base_end_sec = job.end_sample as f64 / SAMPLE_RATE as f64;
    let mut segments = Vec::new();

    for segment in state.as_iter() {
        let text = segment.to_str_lossy()?.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let t0 = segment.start_timestamp() as f64 * 0.01;
        let t1 = segment.end_timestamp() as f64 * 0.01;
        segments.push(TranscriptSegment {
            start_sec: round3(base_start_sec + t0),
            end_sec: round3(base_start_sec + t1),
            text,
        });
    }

    let text = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();

    Ok(TranscriptEvent {
        event_type: "transcript".to_string(),
        schema_version: "1.0".to_string(),
        id: job.id,
        ts: job.received_at,
        source: job.source,
        context: job.context,
        model: ModelInfo {
            id: profile.id.clone(),
            name: profile.name.clone(),
            path: profile.path.clone(),
        },
        audio: AudioInfo {
            start_sec: round3(base_start_sec),
            end_sec: round3(base_end_sec),
            duration_sec: round3(base_end_sec - base_start_sec),
        },
        vad: job.vad,
        result: TranscriptResult {
            language,
            text,
            segments,
        },
    })
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn run_audio_server_blocking(env: EnvConfig, runtime_config: Arc<RwLock<RuntimeConfig>>, job_tx: mpsc::Sender<TranscribeJob>) -> Result<()> {
    let listener = StdTcpListener::bind(env.asr_bind)?;
    info!(addr = %env.asr_bind, "audio TCP server listening");

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let peer = stream.peer_addr().ok();
                info!(?peer, "audio stream connected");
                let runtime_config_for_conn = runtime_config.clone();
                let job_tx_for_conn = job_tx.clone();
                thread::Builder::new()
                    .name("audio-tcp-connection".to_string())
                    .spawn(move || {
                        if let Err(e) = handle_audio_connection_blocking(stream, runtime_config_for_conn, job_tx_for_conn) {
                            warn!(?peer, error = %e, "audio connection ended");
                        }
                    })?;
            }
            Err(e) => warn!(error = %e, "audio accept failed"),
        }
    }

    Ok(())
}

fn handle_audio_connection_blocking(
    mut stream: StdTcpStream,
    runtime_config: Arc<RwLock<RuntimeConfig>>,
    job_tx: mpsc::Sender<TranscribeJob>,
) -> Result<()> {
    let peer_src = stream
        .peer_addr()
        .map(|p| p.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let source = SourceInfo {
        source_type: "stream".to_string(),
        src: peer_src,
        meta: json!({
            "transport": "tcp",
            "format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "channels": 1
        }),
    };
    let context: Option<Value> = None;
    let mut segmenter = AudioSegmenter::new(source, context)?;

    let mut byte_buf = vec![0u8; 8192];
    let mut pending_samples: Vec<i16> = Vec::new();

    loop {
        let n = stream.read(&mut byte_buf)?;
        if n == 0 {
            info!("audio TCP stream closed");
            break;
        }

        for chunk in byte_buf[..n].chunks_exact(BYTES_PER_SAMPLE) {
            pending_samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }

        while pending_samples.len() >= VAD_FRAME_SAMPLES {
            let frame_i16: Vec<i16> = pending_samples.drain(..VAD_FRAME_SAMPLES).collect();
            let frame_f32: Vec<f32> = frame_i16.iter().map(|s| (*s as f32) / 32768.0).collect();
            segmenter.push_frame_blocking(frame_f32, frame_i16, &runtime_config, &job_tx)?;
        }
    }

    segmenter.finish_blocking(&runtime_config, &job_tx)?;

    Ok(())
}

struct AudioSegmenter {
    vad: VoiceActivityDetector,
    source: SourceInfo,
    context: Option<Value>,
    pending_samples: Vec<f32>,
    pre_roll: VecDeque<f32>,
    current_segment: Vec<f32>,
    current_start_sample: u64,
    total_samples_seen: u64,
    silence_samples_in_segment: usize,
    in_speech: bool,
}

impl AudioSegmenter {
    fn new(source: SourceInfo, context: Option<Value>) -> Result<Self> {
        Ok(Self {
            vad: VoiceActivityDetector::builder()
                .sample_rate(SAMPLE_RATE as u32)
                .chunk_size(VAD_FRAME_SAMPLES)
                .build()
                .context("creating Silero VAD V5 detector")?,
            source,
            context,
            pending_samples: Vec::new(),
            pre_roll: VecDeque::new(),
            current_segment: Vec::new(),
            current_start_sample: 0,
            total_samples_seen: 0,
            silence_samples_in_segment: 0,
            in_speech: false,
        })
    }

    fn set_source(&mut self, source: SourceInfo) {
        self.source = source;
    }

    fn push_samples_blocking(
        &mut self,
        samples: &[f32],
        runtime_config: &Arc<RwLock<RuntimeConfig>>,
        job_tx: &mpsc::Sender<TranscribeJob>,
    ) -> Result<()> {
        self.pending_samples.extend_from_slice(samples);
        while self.pending_samples.len() >= VAD_FRAME_SAMPLES {
            let frame_f32: Vec<f32> = self.pending_samples.drain(..VAD_FRAME_SAMPLES).collect();
            let frame_i16 = frame_f32_to_i16(&frame_f32);
            self.push_frame_blocking(frame_f32, frame_i16, runtime_config, job_tx)?;
        }
        Ok(())
    }

    fn push_frame_blocking(
        &mut self,
        frame_f32: Vec<f32>,
        frame_i16: Vec<i16>,
        runtime_config: &Arc<RwLock<RuntimeConfig>>,
        job_tx: &mpsc::Sender<TranscribeJob>,
    ) -> Result<()> {
        let frame_start_sample = self.total_samples_seen;
        self.total_samples_seen += VAD_FRAME_SAMPLES as u64;

        let probability = self.vad.predict(frame_i16);
        let (transcription, vad_debug) = current_transcription_settings(runtime_config);
        let vad_threshold = transcription.vad_threshold;
        let pre_roll_len = seconds_to_samples_allow_zero(transcription.pre_roll_seconds);
        let max_segment_samples = seconds_to_samples(transcription.max_segment_seconds);
        let min_segment_samples = seconds_to_samples(transcription.min_segment_seconds);
        let silence_cut_samples = seconds_to_samples(transcription.silence_cut_seconds);
        let vad_info = VadInfo {
            enabled: true,
            engine: "silero".to_string(),
            threshold: Some(vad_threshold),
        };
        let is_voice = probability >= vad_threshold;
        if vad_debug {
            debug!(probability = probability, threshold = vad_threshold, is_voice = is_voice, "vad frame");
        }

        if !self.in_speech {
            if pre_roll_len == 0 {
                self.pre_roll.clear();
            } else {
                for &sample in &frame_f32 {
                    while self.pre_roll.len() >= pre_roll_len {
                        self.pre_roll.pop_front();
                    }
                    self.pre_roll.push_back(sample);
                }
            }

            if is_voice {
                self.in_speech = true;
                self.silence_samples_in_segment = 0;
                self.current_segment.clear();
                let pre_len = self.pre_roll.len();
                self.current_start_sample = frame_start_sample.saturating_sub(pre_len as u64);
                self.current_segment.extend(self.pre_roll.iter().copied());
                self.current_segment.extend_from_slice(&frame_f32);
                debug!(start_sample = self.current_start_sample, probability = probability, "speech started");
            }
            return Ok(());
        }

        self.current_segment.extend_from_slice(&frame_f32);
        if is_voice {
            self.silence_samples_in_segment = 0;
        } else {
            self.silence_samples_in_segment += VAD_FRAME_SAMPLES;
        }

        let should_flush_for_max = self.current_segment.len() >= max_segment_samples;
        let should_flush_for_silence = self.current_segment.len() >= min_segment_samples
            && self.silence_samples_in_segment >= silence_cut_samples;

        if should_flush_for_max || should_flush_for_silence {
            let end_sample = self.total_samples_seen;
            flush_segment_if_valid_blocking(
                job_tx,
                &mut self.current_segment,
                self.current_start_sample,
                end_sample,
                min_segment_samples,
                &self.source,
                &self.context,
                &vad_info,
                None,
            )?;
            self.in_speech = false;
            self.silence_samples_in_segment = 0;
            self.pre_roll.clear();
            debug!(end_sample = end_sample, max = should_flush_for_max, silence = should_flush_for_silence, "speech ended");
        }

        Ok(())
    }

    fn finish_blocking(
        &mut self,
        runtime_config: &Arc<RwLock<RuntimeConfig>>,
        job_tx: &mpsc::Sender<TranscribeJob>,
    ) -> Result<()> {
        if self.in_speech && !self.current_segment.is_empty() {
            let (transcription, _) = current_transcription_settings(runtime_config);
            let min_segment_samples = seconds_to_samples(transcription.min_segment_seconds);
            let vad_info = VadInfo {
                enabled: true,
                engine: "silero".to_string(),
                threshold: Some(transcription.vad_threshold),
            };
            flush_segment_if_valid_blocking(
                job_tx,
                &mut self.current_segment,
                self.current_start_sample,
                self.total_samples_seen,
                min_segment_samples,
                &self.source,
                &self.context,
                &vad_info,
                None,
            )?;
        }
        self.in_speech = false;
        self.pending_samples.clear();
        self.pre_roll.clear();
        self.current_segment.clear();
        self.silence_samples_in_segment = 0;
        Ok(())
    }
}

fn frame_f32_to_i16(frame: &[f32]) -> Vec<i16> {
    frame
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect()
}

async fn run_teamspeak_manager(
    runtime_config: Arc<RwLock<RuntimeConfig>>,
    job_tx: mpsc::Sender<TranscribeJob>,
    status: Arc<RwLock<TeamSpeakStatus>>,
    mut command_rx: mpsc::UnboundedReceiver<TeamSpeakCommand>,
) {
    let mut cfg = runtime_config.read().unwrap().teamspeak.clone().normalized();
    update_teamspeak_status(&status, |s| {
        s.enabled = cfg.enabled;
        s.server_address = cfg.server_address.clone();
        s.state = if cfg.enabled { "disconnected" } else { "disabled" }.to_string();
    });

    loop {
        if !cfg.enabled {
            match command_rx.recv().await {
                Some(TeamSpeakCommand::Apply(new_cfg)) => cfg = new_cfg.normalized(),
                Some(TeamSpeakCommand::Connect) | Some(TeamSpeakCommand::Reconnect) => {
                    cfg = runtime_config.read().unwrap().teamspeak.clone().normalized();
                    cfg.enabled = true;
                }
                Some(TeamSpeakCommand::Disconnect) => {}
                None => break,
            }
            update_teamspeak_status(&status, |s| {
                s.enabled = cfg.enabled;
                s.server_address = cfg.server_address.clone();
                if !cfg.enabled {
                    s.state = "disabled".to_string();
                    s.connected = false;
                    s.connected_at = None;
                }
            });
            continue;
        }

        match run_teamspeak_session(cfg.clone(), runtime_config.clone(), job_tx.clone(), status.clone(), &mut command_rx).await {
            TeamSpeakSessionEnd::Apply(new_cfg) => cfg = new_cfg.normalized(),
            TeamSpeakSessionEnd::Disconnected => {
                cfg = runtime_config.read().unwrap().teamspeak.clone().normalized();
                update_teamspeak_status(&status, |s| {
                    s.state = if cfg.enabled { "disconnected" } else { "disabled" }.to_string();
                    s.enabled = cfg.enabled;
                    s.connected = false;
                    s.connected_at = None;
                });
            }
            TeamSpeakSessionEnd::Reconnect => {
                cfg = runtime_config.read().unwrap().teamspeak.clone().normalized();
                update_teamspeak_status(&status, |s| {
                    s.state = "reconnecting".to_string();
                    s.enabled = cfg.enabled;
                    s.connected = false;
                    s.connected_at = None;
                });
                time::sleep(Duration::from_secs(cfg.reconnect_seconds)).await;
            }
            TeamSpeakSessionEnd::Fatal(err) => {
                if is_unsupported_teamspeak_license_error(&err) {
                    update_teamspeak_status(&status, |s| {
                        s.state = "unsupported_server_license".to_string();
                        s.enabled = cfg.enabled;
                        s.connected = false;
                        s.connected_at = None;
                        s.last_error = Some(format!(
                            "{err}. The current tsclientlib/tsproto stack could not parse this server's license handshake."
                        ));
                    });
                    match command_rx.recv().await {
                        Some(TeamSpeakCommand::Apply(new_cfg)) => cfg = new_cfg.normalized(),
                        Some(TeamSpeakCommand::Connect) | Some(TeamSpeakCommand::Reconnect) => {
                            cfg = runtime_config.read().unwrap().teamspeak.clone().normalized();
                        }
                        Some(TeamSpeakCommand::Disconnect) => cfg.enabled = false,
                        None => break,
                    }
                    continue;
                }
                let reconnect_seconds = cfg.reconnect_seconds;
                update_teamspeak_status(&status, |s| {
                    s.state = "error".to_string();
                    s.enabled = cfg.enabled;
                    s.connected = false;
                    s.connected_at = None;
                    s.last_error = Some(err);
                });
                time::sleep(Duration::from_secs(reconnect_seconds)).await;
                cfg = runtime_config.read().unwrap().teamspeak.clone().normalized();
            }
        }
    }
}

enum TeamSpeakSessionEnd {
    Apply(TeamSpeakConfig),
    Disconnected,
    Reconnect,
    Fatal(String),
}

async fn run_teamspeak_session(
    cfg: TeamSpeakConfig,
    runtime_config: Arc<RwLock<RuntimeConfig>>,
    job_tx: mpsc::Sender<TranscribeJob>,
    status: Arc<RwLock<TeamSpeakStatus>>,
    command_rx: &mut mpsc::UnboundedReceiver<TeamSpeakCommand>,
) -> TeamSpeakSessionEnd {
    update_teamspeak_status(&status, |s| {
        s.state = "connecting".to_string();
        s.enabled = true;
        s.connected = false;
        s.server_address = cfg.server_address.clone();
        s.last_error = None;
    });

    let mut con = match connect_teamspeak(&cfg) {
        Ok(con) => con,
        Err(e) => {
            return TeamSpeakSessionEnd::Fatal(e.to_string());
        }
    };

    let logger = slog::Logger::root(slog::Discard, slog::o!());
    let mut audio_handler = tsclientlib::audio::AudioHandler::<ClientId>::new(logger);
    let mut segmenters: HashMap<ClientId, AudioSegmenter> = HashMap::new();
    let mut speakers: HashMap<ClientId, TeamSpeakSpeakerMeta> = HashMap::new();
    let mut audio_tick = time::interval(Duration::from_millis(20));

    loop {
        tokio::select! {
            command = command_rx.recv() => {
                match command {
                    Some(TeamSpeakCommand::Apply(new_cfg)) => {
                        return TeamSpeakSessionEnd::Apply(new_cfg);
                    }
                    Some(TeamSpeakCommand::Disconnect) => {
                        return TeamSpeakSessionEnd::Disconnected;
                    }
                    Some(TeamSpeakCommand::Connect) | Some(TeamSpeakCommand::Reconnect) => {
                        return TeamSpeakSessionEnd::Reconnect;
                    }
                    None => {
                        return TeamSpeakSessionEnd::Disconnected;
                    }
                }
            }
            _ = audio_tick.tick() => {
                let mut output = vec![0.0f32; TEAMSPEAK_FRAME_SAMPLES];
                let ended = audio_handler.fill_buffer_with_proc(&mut output, |client_id, samples| {
                    let source = teamspeak_source_for_client(&cfg, client_id, &speakers);
                    let mono_16k = downsample_teamspeak_to_asr(samples);
                    if !segmenters.contains_key(client_id) {
                        match AudioSegmenter::new(source.clone(), None) {
                            Ok(segmenter) => {
                                segmenters.insert(*client_id, segmenter);
                            }
                            Err(e) => {
                                warn!(client_id = ?client_id, error = %e, "failed to create TeamSpeak audio segmenter");
                                return;
                            }
                        }
                    }
                    if let Some(entry) = segmenters.get_mut(client_id) {
                        entry.set_source(source);
                        if let Err(e) = entry.push_samples_blocking(&mono_16k, &runtime_config, &job_tx) {
                            warn!(client_id = ?client_id, error = %e, "failed to segment TeamSpeak audio");
                        }
                    }
                });
                for client_id in ended {
                    if let Some(mut segmenter) = segmenters.remove(&client_id) {
                        if let Err(e) = segmenter.finish_blocking(&runtime_config, &job_tx) {
                            warn!(client_id = ?client_id, error = %e, "failed to flush TeamSpeak audio segment");
                        }
                    }
                }
            }
            event = async {
                let mut events = con.events();
                events.next().await
            } => {
                let Some(event) = event else {
                    return TeamSpeakSessionEnd::Reconnect;
                };
                match event {
                    Ok(StreamItem::BookEvents(_)) => {
                        if let Ok(state) = con.get_state() {
                            let channels = collect_teamspeak_channels(state);
                            let channels_by_id: HashMap<u64, TeamSpeakChannelInfo> =
                                channels.iter().map(|c| (c.id, c.clone())).collect();
                            speakers = state
                                .clients
                                .iter()
                                .map(|(id, client)| {
                                    (*id, teamspeak_speaker_meta(client, &channels_by_id))
                                })
                                .collect();
                            let own_channel = state
                                .clients
                                .get(&state.own_client)
                                .and_then(|c| channels_by_id.get(&channel_id_u64(c.channel)));
                            let configured_path_exists = !cfg.channel_path.trim().is_empty()
                                && channels.iter().any(|channel| channel.path == cfg.channel_path);
                            let configured_id_exists = cfg.channel_id
                                .map(|id| channels_by_id.contains_key(&id))
                                .unwrap_or(false);
                            let selected_missing = (cfg.channel_id.is_some() || !cfg.channel_path.trim().is_empty())
                                && !configured_id_exists
                                && !configured_path_exists;
                            update_teamspeak_status(&status, |s| {
                                s.state = if selected_missing { "channel_missing" } else { "connected" }.to_string();
                                s.enabled = true;
                                s.connected = true;
                                s.server_address = cfg.server_address.clone();
                                s.connected_at.get_or_insert_with(Utc::now);
                                s.channels = channels.clone();
                                s.active_channel_id = own_channel.map(|c| c.id);
                                s.active_channel_path = own_channel.map(|c| c.path.clone()).unwrap_or_default();
                                if selected_missing {
                                    s.last_error = Some(format!("configured TeamSpeak channel is missing: {}", cfg.channel_path));
                                } else {
                                    s.last_error = None;
                                }
                            });
                        }
                    }
                    Ok(StreamItem::Audio(packet)) => {
                        if let Some(from) = teamspeak_audio_sender(&packet) {
                            if let Err(e) = audio_handler.handle_packet(from, packet) {
                                warn!(client_id = ?from, error = %e, "failed to handle TeamSpeak audio packet");
                            }
                        }
                    }
                    Ok(StreamItem::DisconnectedTemporarily(reason)) => {
                        update_teamspeak_status(&status, |s| {
                            s.state = "reconnecting".to_string();
                            s.connected = false;
                            s.connected_at = None;
                            s.last_error = Some(format!("temporarily disconnected: {reason:?}"));
                        });
                    }
                    Ok(_) => {}
                    Err(e) => return TeamSpeakSessionEnd::Fatal(e.to_string()),
                }
            }
        }
    }
}

fn connect_teamspeak(cfg: &TeamSpeakConfig) -> Result<TsConnection> {
    let primary = if let Some(channel_id) = cfg.channel_id {
        Some(TeamSpeakConnectChannel::Id(channel_id))
    } else if !cfg.channel_path.trim().is_empty() {
        Some(TeamSpeakConnectChannel::Path(cfg.channel_path.clone()))
    } else {
        None
    };
    let options = build_teamspeak_connect_options(cfg, primary.clone())?;

    match options.connect() {
        Ok(con) => Ok(con),
        Err(primary_error) if primary.is_some() => {
            if cfg.channel_id.is_some() && !cfg.channel_path.trim().is_empty() {
                warn!(error = %primary_error, "TeamSpeak channel id failed, retrying saved channel path");
                let path_options = build_teamspeak_connect_options(cfg, Some(TeamSpeakConnectChannel::Path(cfg.channel_path.clone())))?;
                if let Ok(con) = path_options.connect() {
                    return Ok(con);
                }
            }
            warn!(error = %primary_error, "TeamSpeak configured channel failed, retrying default channel");
            build_teamspeak_connect_options(cfg, None)?
                .connect()
                .with_context(|| format!("TeamSpeak channel connect failed first: {primary_error}"))
        }
        Err(e) => Err(e.into()),
    }
}

#[derive(Clone)]
enum TeamSpeakConnectChannel {
    Id(u64),
    Path(String),
}

fn build_teamspeak_connect_options(
    cfg: &TeamSpeakConfig,
    channel: Option<TeamSpeakConnectChannel>,
) -> Result<tsclientlib::ConnectOptions> {
    let identity = parse_teamspeak_identity(&cfg.identity)?;
    let mut options = TsConnection::build(cfg.server_address.clone())
        .name(cfg.nickname.clone())
        .identity(identity)
        .input_muted(true)
        .output_muted(false);
    if let Some(password) = &cfg.server_password {
        options = options.password(password.clone());
    }
    match channel {
        Some(TeamSpeakConnectChannel::Id(channel_id)) => {
            options = options.channel_id(ChannelId(channel_id));
        }
        Some(TeamSpeakConnectChannel::Path(path)) => {
            options = options.channel(path);
        }
        None => {}
    }
    if let Some(channel_password) = &cfg.channel_password {
        options = options.channel_password(channel_password.clone());
    }
    Ok(options)
}

fn teamspeak_audio_sender(packet: &tsproto_packets::packets::InAudioBuf) -> Option<ClientId> {
    match packet.data().data() {
        AudioData::S2C { from, .. } | AudioData::S2CWhisper { from, .. } => Some(ClientId(*from)),
        _ => None,
    }
}

fn teamspeak_source_for_client(
    cfg: &TeamSpeakConfig,
    client_id: &ClientId,
    speakers: &HashMap<ClientId, TeamSpeakSpeakerMeta>,
) -> SourceInfo {
    let meta = speakers.get(client_id).cloned().unwrap_or_else(|| TeamSpeakSpeakerMeta {
        client_id: client_id_u64(*client_id),
        client_name: "unknown".to_string(),
        channel_id: None,
        channel_name: String::new(),
        channel_path: String::new(),
    });
    SourceInfo {
        source_type: "teamspeak".to_string(),
        src: cfg.server_address.clone(),
        meta: json!({
            "transport": "teamspeak",
            "client_id": meta.client_id,
            "client_name": meta.client_name,
            "channel_id": meta.channel_id,
            "channel_name": meta.channel_name,
            "channel_path": meta.channel_path,
            "sample_rate_original": TEAMSPEAK_SAMPLE_RATE,
            "sample_rate": SAMPLE_RATE,
            "channels": 1
        }),
    }
}

fn downsample_teamspeak_to_asr(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks(3)
        .filter(|chunk| chunk.len() == 3)
        .map(|chunk| (chunk[0] + chunk[1] + chunk[2]) / 3.0)
        .collect()
}

fn update_teamspeak_status(status: &Arc<RwLock<TeamSpeakStatus>>, update: impl FnOnce(&mut TeamSpeakStatus)) {
    let mut guard = status.write().unwrap();
    update(&mut guard);
}

fn collect_teamspeak_channels(state: &tsclientlib::data::Connection) -> Vec<TeamSpeakChannelInfo> {
    let mut channels = state
        .channels
        .iter()
        .map(|(id, channel)| {
            let id_u64 = channel_id_u64(*id);
            let channel_value = serde_json::to_value(channel).unwrap_or(Value::Null);
            let parent_id = teamspeak_channel_parent_from_value(&channel_value);
            let name = teamspeak_channel_name_from_value(&channel_value).unwrap_or_else(|| format!("Channel {id_u64}"));
            TeamSpeakChannelInfo {
                id: id_u64,
                parent_id,
                name,
                path: String::new(),
            }
        })
        .collect::<Vec<_>>();

    let names = channels
        .iter()
        .map(|c| (c.id, (c.parent_id, c.name.clone())))
        .collect::<HashMap<_, _>>();
    for channel in &mut channels {
        channel.path = build_teamspeak_channel_path(channel.id, &names);
    }
    channels.sort_by(|a, b| a.path.cmp(&b.path).then_with(|| a.id.cmp(&b.id)));
    channels
}

fn build_teamspeak_channel_path(id: u64, channels: &HashMap<u64, (Option<u64>, String)>) -> String {
    let mut parts = Vec::new();
    let mut current = Some(id);
    let mut guard = 0;
    while let Some(channel_id) = current {
        guard += 1;
        if guard > 64 {
            break;
        }
        let Some((parent, name)) = channels.get(&channel_id) else {
            break;
        };
        parts.push(name.clone());
        current = *parent;
    }
    parts.reverse();
    parts.join("/")
}

fn teamspeak_speaker_meta(
    client: &tsclientlib::data::Client,
    channels_by_id: &HashMap<u64, TeamSpeakChannelInfo>,
) -> TeamSpeakSpeakerMeta {
    let channel_id = channel_id_u64(client.channel);
    let channel = channels_by_id.get(&channel_id);
    TeamSpeakSpeakerMeta {
        client_id: client_id_u64(client.id),
        client_name: client.name.clone(),
        channel_id: Some(channel_id),
        channel_name: channel.map(|c| c.name.clone()).unwrap_or_default(),
        channel_path: channel.map(|c| c.path.clone()).unwrap_or_default(),
    }
}

fn teamspeak_channel_name_from_value(value: &Value) -> Option<String> {
    value
        .get("name")
        .or_else(|| value.get("channel_name"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn teamspeak_channel_parent_from_value(value: &Value) -> Option<u64> {
    let parent = value
        .get("parent")
        .or_else(|| value.get("parent_id"))
        .or_else(|| value.get("pid"))
        .or_else(|| value.get("channel_parent_id"));
    match parent {
        Some(Value::Number(n)) => n.as_u64().filter(|id| *id != 0),
        Some(Value::Object(obj)) => obj
            .get("0")
            .or_else(|| obj.get("id"))
            .and_then(Value::as_u64)
            .filter(|id| *id != 0),
        _ => None,
    }
}

fn channel_id_u64(id: ChannelId) -> u64 {
    id.0
}

fn client_id_u64(id: ClientId) -> u64 {
    id.0 as u64
}

fn is_unsupported_teamspeak_license_error(error: &str) -> bool {
    let normalized = error.to_ascii_lowercase();
    normalized.contains("failed to parse license") || normalized.contains("intermediate license")
}

fn current_transcription_settings(runtime_config: &Arc<RwLock<RuntimeConfig>>) -> (TranscriptionConfig, bool) {
    let cfg = runtime_config.read().unwrap();
    (cfg.transcription.normalized(), cfg.logging.vad_debug)
}

fn seconds_to_samples(seconds: f64) -> usize {
    (seconds * SAMPLE_RATE as f64).round().max(1.0) as usize
}

fn seconds_to_samples_allow_zero(seconds: f64) -> usize {
    (seconds * SAMPLE_RATE as f64).round().max(0.0) as usize
}

fn flush_segment_if_valid_blocking(
    job_tx: &mpsc::Sender<TranscribeJob>,
    current_segment: &mut Vec<f32>,
    start_sample: u64,
    end_sample: u64,
    min_segment_samples: usize,
    source: &SourceInfo,
    context: &Option<Value>,
    vad: &VadInfo,
    reply: Option<oneshot::Sender<Result<TranscriptEvent, String>>>,
) -> Result<()> {
    if current_segment.len() < min_segment_samples {
        debug!(samples = current_segment.len(), "dropping too short segment");
        current_segment.clear();
        return Ok(());
    }

    let audio = std::mem::take(current_segment);
    let id = Uuid::new_v4();
    let duration = audio.len() as f64 / SAMPLE_RATE as f64;
    info!(%id, duration_sec = duration, "queueing segment for transcription");
    send_transcribe_job(
        job_tx,
        TranscribeJob {
            id,
            audio,
            start_sample,
            end_sample,
            received_at: Utc::now(),
            source: source.clone(),
            context: context.clone(),
            vad: vad.clone(),
            reply,
        },
    )?;
    Ok(())
}

fn send_transcribe_job(job_tx: &mpsc::Sender<TranscribeJob>, job: TranscribeJob) -> Result<()> {
    if tokio::runtime::Handle::try_current().is_ok() {
        job_tx
            .try_send(job)
            .map_err(|e| anyhow!("transcriber queue unavailable: {e}"))?;
    } else {
        job_tx
            .blocking_send(job)
            .map_err(|e| anyhow!("transcriber queue closed: {e}"))?;
    }
    Ok(())
}

async fn dispatch_transcripts(
    state: SharedState,
    mut rx: mpsc::UnboundedReceiver<TranscriptEvent>,
) -> Result<()> {
    if let Some(parent) = Path::new(&state.env.output_jsonl).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&state.env.output_jsonl)
        .await?;

    while let Some(event) = rx.recv().await {
        let payload = serde_json::to_string(&event)?;
        println!("{payload}");
        file.write_all(payload.as_bytes()).await?;
        file.write_all(b"\n").await?;
        file.flush().await?;

        let _ = state.events.send(payload.clone());

        let cfg = state.runtime_config.read().unwrap().clone();
        let webhook_state = state.clone();
        let mqtt_state = state.clone();
        let payload_for_webhook = payload.clone();
        let payload_for_mqtt = payload.clone();
        tokio::spawn(async move {
            if let Err(e) = send_webhooks(&webhook_state, &cfg.webhooks, payload_for_webhook).await {
                warn!(error = %e, "webhook delivery error");
            }
        });
        tokio::spawn(async move {
            if let Err(e) = publish_mqtt(&mqtt_state, &cfg.mqtt, payload_for_mqtt).await {
                warn!(error = %e, "mqtt delivery error");
            }
        });
    }

    Ok(())
}

async fn send_webhooks(state: &SharedState, hooks: &[WebhookConfig], payload: String) -> Result<()> {
    for hook in hooks.iter().filter(|h| h.enabled) {
        let mut req = state
            .http
            .post(&hook.url)
            .header("content-type", "application/json")
            .body(payload.clone());
        if let Some(token) = &hook.bearer_token {
            if !token.trim().is_empty() {
                req = req.bearer_auth(token);
            }
        }
        let resp = req.send().await?;
        if !resp.status().is_success() {
            warn!(name = %hook.name, status = %resp.status(), "webhook returned non-success status");
        }
    }
    Ok(())
}

async fn publish_mqtt(_state: &SharedState, cfg: &MqttConfig, payload: String) -> Result<()> {
    if !cfg.enabled {
        return Ok(());
    }

    let client_id = if cfg.client_id.trim().is_empty() {
        format!("audio2mqtt-{}", Uuid::new_v4())
    } else {
        cfg.client_id.clone()
    };

    let mut options = MqttOptions::new(client_id, cfg.host.clone(), cfg.port);
    options.set_keep_alive(Duration::from_secs(10));
    if let Some(username) = &cfg.username {
        if !username.trim().is_empty() {
            options.set_credentials(username, cfg.password.clone().unwrap_or_default());
        }
    }

    let (client, mut eventloop): (AsyncClient, EventLoop) = AsyncClient::new(options, 10);
    let qos = match cfg.qos {
        0 => QoS::AtMostOnce,
        2 => QoS::ExactlyOnce,
        _ => QoS::AtLeastOnce,
    };

    client.publish(cfg.topic.clone(), qos, false, payload).await?;

    let _ = tokio::time::timeout(Duration::from_secs(3), async move {
        for _ in 0..10 {
            let _ = eventloop.poll().await?;
        }
        Ok::<_, rumqttc::ConnectionError>(())
    })
    .await;

    Ok(())
}

async fn run_web_admin(state: SharedState) -> Result<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/status", get(status))
        .route("/api/config", get(get_config).post(post_config))
        .route("/api/models", get(get_models))
        .route("/api/models/download", post(download_model))
        .route("/api/models/activate", post(activate_model))
        .route("/api/models/huggingface", post(add_huggingface_model))
        .route("/api/teamspeak/status", get(teamspeak_status))
        .route("/api/teamspeak/connect", post(teamspeak_connect))
        .route("/api/teamspeak/disconnect", post(teamspeak_disconnect))
        .route("/api/teamspeak/reconnect", post(teamspeak_reconnect))
        .route("/api/teamspeak/channel", post(teamspeak_select_channel))
        .route("/api/test/webhook", post(test_webhook))
        .route("/api/test/mqtt", post(test_mqtt))
        .route("/api/transcribe", post(rest_transcribe))
        .route("/api/events", get(events_sse))
        .nest_service("/static", ServeDir::new("/app/static"))
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let listener = TcpListener::bind(state.env.admin_bind).await?;
    info!(addr = %state.env.admin_bind, "web admin listening");
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn status(State(state): State<SharedState>) -> impl IntoResponse {
    let cfg = state.runtime_config.read().unwrap().clone();
    let active_model = active_model_profile(&cfg, &state.env);
    let active_model_exists = Path::new(&active_model.path).exists();
    let transcription = cfg.transcription.normalized();
    let language = transcription.language.clone();
    Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "started_at": state.started_at,
        "asr_bind": state.env.asr_bind.to_string(),
        "admin_bind": state.env.admin_bind.to_string(),
        "active_model_id": active_model.id,
        "active_model_name": active_model.name,
        "active_model_path": active_model.path,
        "active_model_exists": active_model_exists,
        "language": language,
        "vad_backend": "voice_activity_detector/Silero VAD V5",
        "vad_threshold": transcription.vad_threshold,
        "transcription": transcription,
        "output_jsonl": state.env.output_jsonl.clone(),
        "logging": cfg.logging,
        "teamspeak": state.teamspeak.status.read().unwrap().clone(),
    }))
}

async fn get_config(State(state): State<SharedState>) -> impl IntoResponse {
    let cfg = state.runtime_config.read().unwrap().clone();
    Json(cfg)
}

async fn post_config(State(state): State<SharedState>, Json(mut new_cfg): Json<RuntimeConfig>) -> impl IntoResponse {
    normalize_runtime_config(&mut new_cfg, &state.env);
    {
        let mut guard = state.runtime_config.write().unwrap();
        *guard = new_cfg.clone();
    }
    let _ = state.teamspeak.tx.send(TeamSpeakCommand::Apply(new_cfg.teamspeak.clone()));
    let reload_result = state
        .log_reload
        .reload(EnvFilter::new(log_filter_from_level(&new_cfg.logging.level)))
        .map_err(|e| e.to_string());

    match save_runtime_config(&state.config_path, &new_cfg) {
        Ok(_) => Json(json!({ "ok": true, "log_reload": reload_result.is_ok() })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string(), "log_reload_error": reload_result.err() })),
    }
}

async fn teamspeak_status(State(state): State<SharedState>) -> impl IntoResponse {
    Json(state.teamspeak.status.read().unwrap().clone())
}

async fn teamspeak_connect(State(state): State<SharedState>) -> impl IntoResponse {
    let mut cfg = state.runtime_config.read().unwrap().clone();
    cfg.teamspeak.enabled = true;
    normalize_runtime_config(&mut cfg, &state.env);
    {
        let mut guard = state.runtime_config.write().unwrap();
        *guard = cfg.clone();
    }
    let saved = save_runtime_config(&state.config_path, &cfg).map_err(|e| e.to_string());
    let sent = state.teamspeak.tx.send(TeamSpeakCommand::Connect).is_ok();
    Json(json!({"ok": saved.is_ok() && sent, "saved": saved.is_ok(), "command_sent": sent, "error": saved.err()}))
}

async fn teamspeak_disconnect(State(state): State<SharedState>) -> impl IntoResponse {
    let mut cfg = state.runtime_config.read().unwrap().clone();
    cfg.teamspeak.enabled = false;
    normalize_runtime_config(&mut cfg, &state.env);
    {
        let mut guard = state.runtime_config.write().unwrap();
        *guard = cfg.clone();
    }
    let saved = save_runtime_config(&state.config_path, &cfg).map_err(|e| e.to_string());
    let sent = state.teamspeak.tx.send(TeamSpeakCommand::Disconnect).is_ok();
    Json(json!({"ok": saved.is_ok() && sent, "saved": saved.is_ok(), "command_sent": sent, "error": saved.err()}))
}

async fn teamspeak_reconnect(State(state): State<SharedState>) -> impl IntoResponse {
    let sent = state.teamspeak.tx.send(TeamSpeakCommand::Reconnect).is_ok();
    Json(json!({"ok": sent, "command_sent": sent}))
}

async fn teamspeak_select_channel(
    State(state): State<SharedState>,
    Json(req): Json<TeamSpeakChannelSelectRequest>,
) -> impl IntoResponse {
    let channel = {
        let status = state.teamspeak.status.read().unwrap();
        status.channels.iter().find(|c| c.id == req.channel_id).cloned()
    };

    let Some(channel) = channel else {
        return Json(json!({"ok": false, "error": format!("unknown TeamSpeak channel id {}", req.channel_id)}));
    };

    let mut cfg = state.runtime_config.read().unwrap().clone();
    cfg.teamspeak.channel_id = Some(channel.id);
    cfg.teamspeak.channel_path = channel.path.clone();
    normalize_runtime_config(&mut cfg, &state.env);
    {
        let mut guard = state.runtime_config.write().unwrap();
        *guard = cfg.clone();
    }
    let saved = save_runtime_config(&state.config_path, &cfg).map_err(|e| e.to_string());
    let sent = state.teamspeak.tx.send(TeamSpeakCommand::Reconnect).is_ok();
    Json(json!({
        "ok": saved.is_ok() && sent,
        "channel": channel,
        "saved": saved.is_ok(),
        "command_sent": sent,
        "error": saved.err(),
    }))
}

fn normalize_runtime_config(cfg: &mut RuntimeConfig, env: &EnvConfig) {
    cfg.transcription = cfg.transcription.normalized();
    cfg.teamspeak = cfg.teamspeak.normalized();
    if cfg.models.profiles.is_empty() {
        cfg.models.profiles = default_model_profiles(&env.default_model_id, &env.model_path, &env.model_url);
    }
    ensure_finnish_nlp_profile(&mut cfg.models.profiles);
    if cfg.models.active_model_id.trim().is_empty()
        || !cfg.models.profiles.iter().any(|p| p.id.as_str() == cfg.models.active_model_id.as_str())
    {
        cfg.models.active_model_id = cfg
            .models
            .profiles
            .first()
            .map(|p| p.id.clone())
            .unwrap_or_else(|| env.default_model_id.clone());
    }
}

async fn get_models(State(state): State<SharedState>) -> impl IntoResponse {
    let cfg = state.runtime_config.read().unwrap().clone();
    let profiles = cfg
        .models
        .profiles
        .iter()
        .map(|p| {
            json!({
                "id": &p.id,
                "name": &p.name,
                "path": &p.path,
                "url": &p.url,
                "description": &p.description,
                "recommended_vram_gb": p.recommended_vram_gb,
                "exists": Path::new(&p.path).exists(),
                "active": p.id.as_str() == cfg.models.active_model_id.as_str(),
            })
        })
        .collect::<Vec<Value>>();

    Json(json!({
        "active_model_id": cfg.models.active_model_id,
        "profiles": profiles,
    }))
}

async fn download_model(State(state): State<SharedState>, Json(req): Json<ModelSelectRequest>) -> impl IntoResponse {
    let profile = {
        let cfg = state.runtime_config.read().unwrap().clone();
        cfg.models.profiles.iter().find(|p| p.id.as_str() == req.id.as_str()).cloned()
    };

    let Some(profile) = profile else {
        return Json(json!({"ok": false, "error": format!("unknown model id: {}", req.id)}));
    };

    match ensure_model_downloaded(&profile).await {
        Ok(_) => Json(json!({"ok": true, "id": profile.id, "path": profile.path})),
        Err(e) => Json(json!({"ok": false, "id": profile.id, "error": e.to_string()})),
    }
}

async fn activate_model(State(state): State<SharedState>, Json(req): Json<ModelActivateRequest>) -> impl IntoResponse {
    let mut cfg = state.runtime_config.read().unwrap().clone();
    let profile = cfg.models.profiles.iter().find(|p| p.id.as_str() == req.id.as_str()).cloned();

    let Some(profile) = profile else {
        return Json(json!({"ok": false, "error": format!("unknown model id: {}", req.id)}));
    };

    if !Path::new(&profile.path).exists() {
        if req.download_if_missing {
            if let Err(e) = ensure_model_downloaded(&profile).await {
                return Json(json!({"ok": false, "id": profile.id, "error": e.to_string()}));
            }
        } else {
            return Json(json!({"ok": false, "id": profile.id, "error": "model file is missing"}));
        }
    }

    cfg.models.active_model_id = profile.id.clone();
    {
        let mut guard = state.runtime_config.write().unwrap();
        *guard = cfg.clone();
    }

    match save_runtime_config(&state.config_path, &cfg) {
        Ok(_) => Json(json!({
            "ok": true,
            "active_model_id": profile.id,
            "active_model_path": profile.path,
            "reload": "model reloads before the next transcription job"
        })),
        Err(e) => Json(json!({"ok": false, "error": e.to_string()})),
    }
}

async fn add_huggingface_model(
    State(state): State<SharedState>,
    Json(req): Json<HuggingFaceModelAddRequest>,
) -> impl IntoResponse {
    let (repo_id, requested_filename) = match split_huggingface_repo_and_filename(&req.repo_id, req.filename.as_deref()) {
        Ok(v) => v,
        Err(e) => return Json(json!({"ok": false, "error": e.to_string()})),
    };

    let api_url = format!("https://huggingface.co/api/models/{repo_id}");
    let resp = match reqwest::get(&api_url).await {
        Ok(resp) => resp,
        Err(e) => return Json(json!({"ok": false, "repo_id": repo_id, "error": e.to_string()})),
    };
    if !resp.status().is_success() {
        return Json(json!({
            "ok": false,
            "repo_id": repo_id,
            "error": format!("Hugging Face returned {}", resp.status())
        }));
    }

    let info: HuggingFaceModelInfo = match resp.json().await {
        Ok(info) => info,
        Err(e) => return Json(json!({"ok": false, "repo_id": repo_id, "error": e.to_string()})),
    };

    let filename = match select_huggingface_model_file(&info.siblings, requested_filename.as_deref()) {
        Ok(filename) => filename,
        Err(e) => return Json(json!({"ok": false, "repo_id": repo_id, "error": e.to_string()})),
    };

    let profile = huggingface_model_profile(&repo_id, &filename, req.id.as_deref(), req.name.as_deref());
    let mut cfg = state.runtime_config.read().unwrap().clone();
    let replaced = if let Some(existing) = cfg.models.profiles.iter_mut().find(|p| p.id.as_str() == profile.id.as_str()) {
        *existing = profile.clone();
        true
    } else {
        cfg.models.profiles.push(profile.clone());
        false
    };
    if req.set_active {
        cfg.models.active_model_id = profile.id.clone();
    }
    normalize_runtime_config(&mut cfg, &state.env);

    {
        let mut guard = state.runtime_config.write().unwrap();
        *guard = cfg.clone();
    }

    match save_runtime_config(&state.config_path, &cfg) {
        Ok(_) => Json(json!({
            "ok": true,
            "repo_id": repo_id,
            "filename": filename,
            "profile": profile,
            "replaced": replaced,
            "active_model_id": cfg.models.active_model_id.clone(),
        })),
        Err(e) => Json(json!({"ok": false, "repo_id": repo_id, "error": e.to_string()})),
    }
}

fn split_huggingface_repo_and_filename(repo_input: &str, filename_input: Option<&str>) -> Result<(String, Option<String>)> {
    let mut input = repo_input.trim().to_string();
    if input.is_empty() {
        return Err(anyhow!("Hugging Face repo name is empty; expected owner/model"));
    }

    input = input
        .split('?')
        .next()
        .unwrap_or("")
        .split('#')
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('/')
        .to_string();

    let stripped_url = ["https://huggingface.co/", "http://huggingface.co/"]
        .iter()
        .find_map(|prefix| input.strip_prefix(*prefix).map(|rest| rest.trim_matches('/').to_string()));
    if let Some(stripped_url) = stripped_url {
        input = stripped_url;
    }

    let mut filename = filename_input
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_start_matches('/').to_string());

    for marker in ["/resolve/", "/blob/"] {
        if let Some(pos) = input.find(marker) {
            let repo = input[..pos].trim_matches('/').to_string();
            let revision_and_file = &input[pos + marker.len()..];
            if filename.is_none() {
                filename = filename_after_revision(revision_and_file);
            }
            input = repo;
            break;
        }
    }

    if let Some(pos) = input.find("/tree/") {
        input = input[..pos].trim_matches('/').to_string();
    }

    if filename.is_none() {
        if let Some(pos) = input.find(':') {
            let maybe_file = input[pos + 1..].trim().trim_start_matches('/');
            if !maybe_file.is_empty() {
                filename = Some(maybe_file.to_string());
            }
            input = input[..pos].trim_matches('/').to_string();
        }
    }

    if filename.is_none() {
        let parts = input.split('/').map(ToOwned::to_owned).collect::<Vec<_>>();
        if parts.len() > 2 {
            let maybe_file = parts[2..].join("/");
            if maybe_file.to_lowercase().ends_with(".bin") {
                filename = Some(maybe_file);
                input = format!("{}/{}", parts[0], parts[1]);
            }
        }
    }

    let repo_id = input.trim_matches('/').to_string();
    validate_huggingface_repo_id(&repo_id)?;
    Ok((repo_id, filename))
}

fn filename_after_revision(revision_and_file: &str) -> Option<String> {
    revision_and_file
        .split_once('/')
        .map(|(_, file)| file.trim().trim_start_matches('/').to_string())
        .filter(|file| !file.is_empty())
}

fn validate_huggingface_repo_id(repo_id: &str) -> Result<()> {
    let parts = repo_id.split('/').collect::<Vec<_>>();
    if parts.len() != 2 || parts.iter().any(|part| part.trim().is_empty()) {
        return Err(anyhow!("expected Hugging Face repo name in owner/model form"));
    }
    if !repo_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.'))
    {
        return Err(anyhow!("Hugging Face repo name contains unsupported characters"));
    }
    Ok(())
}

fn select_huggingface_model_file(siblings: &[HuggingFaceSibling], requested_filename: Option<&str>) -> Result<String> {
    let bin_files = siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|name| name.to_lowercase().ends_with(".bin"))
        .collect::<Vec<_>>();

    if let Some(requested) = requested_filename.map(str::trim).filter(|s| !s.is_empty()) {
        let requested = requested.trim_start_matches('/');
        if bin_files.iter().any(|name| *name == requested) {
            return Ok(requested.to_string());
        }
        return Err(anyhow!(
            "file '{}' was not found in the Hugging Face repo; available .bin files: {}",
            requested,
            summarize_candidates(&bin_files)
        ));
    }

    let mut candidates = bin_files;
    if candidates.is_empty() {
        return Err(anyhow!("no .bin model file was found in the Hugging Face repo"));
    }

    candidates.sort_by(|a, b| {
        huggingface_file_score(b)
            .cmp(&huggingface_file_score(a))
            .then_with(|| a.cmp(b))
    });
    Ok(candidates[0].to_string())
}

fn huggingface_file_score(filename: &str) -> i32 {
    let name = filename.to_lowercase();
    let mut score = 0;
    if name.contains("ggml") {
        score += 100;
    }
    if name.contains("large-v3") {
        score += 90;
    } else if name.contains("large-v2") {
        score += 85;
    } else if name.contains("large") {
        score += 80;
    } else if name.contains("medium") {
        score += 70;
    } else if name.contains("small") {
        score += 60;
    } else if name.contains("base") {
        score += 50;
    } else if name.contains("tiny") {
        score += 40;
    }
    score
}

fn summarize_candidates(candidates: &[&str]) -> String {
    if candidates.is_empty() {
        return "none".to_string();
    }
    let mut names = candidates.iter().take(8).copied().collect::<Vec<_>>().join(", ");
    if candidates.len() > 8 {
        names.push_str(", ...");
    }
    names
}

fn huggingface_model_profile(repo_id: &str, filename: &str, requested_id: Option<&str>, requested_name: Option<&str>) -> ModelProfile {
    let basename = filename.rsplit('/').next().unwrap_or(filename);
    let stem = basename.strip_suffix(".bin").unwrap_or(basename);
    let id = requested_id
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("hf-{}-{}", sanitize_identifier(repo_id), sanitize_identifier(stem)));
    let name = requested_name
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("{repo_id} / {basename}"));

    ModelProfile {
        id,
        name,
        path: format!("/models/{}-{}", sanitize_identifier(repo_id), sanitize_filename(basename)),
        url: format!("https://huggingface.co/{repo_id}/resolve/main/{}", encode_huggingface_path(filename)),
        description: format!("Hugging Face GGML model: {repo_id}/{filename}"),
        recommended_vram_gb: None,
    }
}

fn sanitize_identifier(value: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in value.chars().flat_map(|c| c.to_lowercase()) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

fn sanitize_filename(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn encode_huggingface_path(path: &str) -> String {
    path.split('/')
        .map(percent_encode_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn percent_encode_path_segment(segment: &str) -> String {
    let mut out = String::new();
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

async fn ensure_model_downloaded(profile: &ModelProfile) -> Result<()> {
    let path = Path::new(&profile.path);
    if path.exists() {
        return Ok(());
    }
    if profile.url.trim().is_empty() {
        return Err(anyhow!("model URL is empty for {}", profile.id));
    }
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let tmp_path = format!("{}.tmp-download", profile.path);
    let tmp = Path::new(&tmp_path);
    let _ = tokio::fs::remove_file(tmp).await;

    info!(model_id = %profile.id, url = %profile.url, path = %profile.path, "downloading model");
    let resp = reqwest::get(&profile.url).await?.error_for_status()?;
    let mut stream = resp.bytes_stream();
    let mut file = OpenOptions::new().create(true).truncate(true).write(true).open(tmp).await?;

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        downloaded += chunk.len() as u64;
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    tokio::fs::rename(tmp, path).await?;
    info!(model_id = %profile.id, bytes = downloaded, "model downloaded");
    Ok(())
}

async fn test_webhook(State(state): State<SharedState>) -> impl IntoResponse {
    let payload = json!({
        "id": Uuid::new_v4(),
        "ts": Utc::now(),
        "test": true,
        "text": "Webhook test from audio2mqtt"
    })
    .to_string();
    let cfg = state.runtime_config.read().unwrap().clone();
    match send_webhooks(&state, &cfg.webhooks, payload).await {
        Ok(_) => Json(json!({"ok": true})),
        Err(e) => Json(json!({"ok": false, "error": e.to_string()})),
    }
}

async fn test_mqtt(State(state): State<SharedState>) -> impl IntoResponse {
    let payload = json!({
        "id": Uuid::new_v4(),
        "ts": Utc::now(),
        "test": true,
        "text": "MQTT test from audio2mqtt"
    })
    .to_string();
    let cfg = state.runtime_config.read().unwrap().clone();
    match publish_mqtt(&state, &cfg.mqtt, payload).await {
        Ok(_) => Json(json!({"ok": true})),
        Err(e) => Json(json!({"ok": false, "error": e.to_string()})),
    }
}

async fn rest_transcribe(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<SharedState>,
    Json(req): Json<RestTranscribeRequest>,
) -> impl IntoResponse {
    let (audio, sample_rate, channels, format, encoding) = match decode_rest_audio(&req.audio) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"ok": false, "error": e.to_string()})),
            );
        }
    };

    let source = SourceInfo {
        source_type: "rest".to_string(),
        src: addr.to_string(),
        meta: json!({
            "endpoint": "/api/transcribe",
            "format": format,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "channels": channels
        }),
    };
    let vad = VadInfo {
        enabled: false,
        engine: "none".to_string(),
        threshold: None,
    };

    let end_sample = audio.len() as u64;
    let (tx, rx) = oneshot::channel();
    let job = TranscribeJob {
        id: Uuid::new_v4(),
        audio,
        start_sample: 0,
        end_sample,
        received_at: Utc::now(),
        source,
        context: req.context,
        vad,
        reply: Some(tx),
    };

    if let Err(e) = state.job_tx.send(job).await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok": false, "error": format!("transcriber queue unavailable: {e}")})),
        );
    }

    match tokio::time::timeout(Duration::from_secs(300), rx).await {
        Ok(Ok(Ok(event))) => (StatusCode::OK, Json(json!(event))),
        Ok(Ok(Err(e))) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"ok": false, "error": e}))),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": format!("transcriber response channel closed: {e}")})),
        ),
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(json!({"ok": false, "error": "transcription timed out"})),
        ),
    }
}

fn decode_rest_audio(audio: &RestAudioRequest) -> Result<(Vec<f32>, usize, usize, String, String)> {
    let format = audio.format.trim().to_lowercase();
    let encoding = audio.encoding.trim().to_lowercase();
    if encoding != "base64" {
        return Err(anyhow!("unsupported audio.encoding '{}'; expected 'base64'", audio.encoding));
    }
    let bytes = general_purpose::STANDARD
        .decode(audio.data.trim())
        .context("decoding base64 audio.data")?;

    match format.as_str() {
        "pcm_s16le" | "s16le" => {
            let sample_rate = audio.sample_rate.unwrap_or(SAMPLE_RATE);
            let channels = audio.channels.unwrap_or(1);
            if sample_rate != SAMPLE_RATE {
                return Err(anyhow!("unsupported sample_rate {sample_rate}; expected {SAMPLE_RATE}"));
            }
            if channels != 1 {
                return Err(anyhow!("unsupported channels {channels}; expected 1"));
            }
            if bytes.len() % 2 != 0 {
                return Err(anyhow!("pcm_s16le byte length must be divisible by 2"));
            }
            let samples = bytes
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect::<Vec<f32>>();
            Ok((samples, sample_rate, channels, "pcm_s16le".to_string(), encoding))
        }
        "wav" => decode_wav_bytes(&bytes, encoding),
        other => Err(anyhow!("unsupported audio.format '{other}'; expected 'wav' or 'pcm_s16le'")),
    }
}

fn decode_wav_bytes(bytes: &[u8], encoding: String) -> Result<(Vec<f32>, usize, usize, String, String)> {
    let cursor = Cursor::new(bytes.to_vec());
    let mut reader = hound::WavReader::new(cursor).context("reading WAV data")?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as usize;
    let channels = spec.channels as usize;
    if sample_rate != SAMPLE_RATE {
        return Err(anyhow!("unsupported WAV sample_rate {sample_rate}; expected {SAMPLE_RATE}"));
    }
    if channels == 0 {
        return Err(anyhow!("WAV channels must be at least 1"));
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(anyhow!(
            "unsupported WAV encoding; expected 16-bit PCM integer, got {:?} {} bits",
            spec.sample_format,
            spec.bits_per_sample
        ));
    }

    let raw = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<i16>, _>>()
        .context("reading WAV samples")?;

    let samples = if channels == 1 {
        raw.iter().map(|s| *s as f32 / 32768.0).collect::<Vec<f32>>()
    } else {
        raw.chunks(channels)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|s| *s as i32).sum();
                (sum as f32 / frame.len() as f32) / 32768.0
            })
            .collect::<Vec<f32>>()
    };

    Ok((samples, sample_rate, channels, "wav".to_string(), encoding))
}

async fn events_sse(
    State(state): State<SharedState>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| async move {
        match msg {
            Ok(payload) => Some(Ok(Event::default().event("transcript").data(payload))),
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
