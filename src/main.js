import './style.css';
import { FilesetResolver, FaceLandmarker, GestureRecognizer } from '@mediapipe/tasks-vision';
import {
  computeFaceDescriptor,
  matchVisitorProfile,
  updateRunningDescriptor,
  extractVisitorName,
  classifyFaceExpression,
} from './visitorMemory.js';

const stageEl = document.getElementById('stage');
const videoFrameEl = document.getElementById('video-frame');
const avatarCanvas = document.getElementById('avatar-canvas');
const avatarCtx = avatarCanvas.getContext('2d');
const fallbackEl = document.getElementById('avatar-fallback');
const livePill = document.getElementById('live-pill');
const gazePill = document.getElementById('gaze-pill');
const audiencePill = document.getElementById('audience-pill');
const gesturePill = document.getElementById('gesture-pill');
const avatarDock = document.getElementById('avatar-dock');
const brandTitle = document.getElementById('brand-title');
const brandSub = document.getElementById('brand-sub');
const statusDot = document.getElementById('status-dot');
const statusBadge = document.getElementById('status-badge');
const telemetryBadge = document.getElementById('telemetry-badge');
const bannerEl = document.getElementById('banner');
const bannerText = document.getElementById('banner-text');

const liveBtn = document.getElementById('live-btn');
const cameraBtn = document.getElementById('camera-btn');
const mouseBtn = document.getElementById('mouse-btn');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const portraitInput = document.getElementById('portrait-input');

const cameraPip = document.getElementById('camera-pip');
const webcamVideo = document.getElementById('webcam-video');
const webcamOverlay = document.getElementById('webcam-overlay');
const webcamOverlayCtx = webcamOverlay.getContext('2d');
const pipLabel = document.getElementById('pip-label');

const transcriptText = document.getElementById('transcript-text');
const topicTag = document.getElementById('topic-tag');
const micStateEl = document.getElementById('mic-state');
const chatForm = document.getElementById('chat-form');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const interruptBtn = document.getElementById('interrupt-btn');
const micBtn = document.getElementById('mic-btn');
const chips = document.getElementById('chips');

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const clientSessionId =
  typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

let ws = null;
let liveMode = false;
let cameraActive = false;
let mouseGazeActive = false;
let webcamStream = null;
let faceLandmarker = null;
let gestureRecognizer = null;
let webcamLoopId = null;
let lastGazeSentMs = 0;
let lastSmileSentMs = 0;

let avatarList = [];
let activeAvatarId = 'sibem';
let audience = 'adult';
let audienceManualOverride = false;
let lastAudienceCheckMs = 0;
let audienceCheckInFlight = false;
let faceMissingSinceMs = 0;
let visitorGreeted = false;
let stableFaceFrames = 0;
let awaitingNewVisitorGreet = false;
let hadPreviousVisitor = false;
let awaitingVisitorName = false;
let gestureHideTimer = null;
let lastVocalReactionMs = 0;
const lastGestureTriggerMs = {};
let candidateGesture = '';
let candidateGestureFrames = 0;
let candidateExpr = '';
let exprHoldStartMs = 0;
let lastDescriptorRefreshMs = 0;

const VISITOR_STORE_KEY = 'sibem_visitors_v1';
let visitorProfiles = [];
try {
  const saved = JSON.parse(localStorage.getItem(VISITOR_STORE_KEY) || '[]');
  if (Array.isArray(saved)) visitorProfiles = saved;
} catch {
  visitorProfiles = [];
}
let currentVisitor = null;

function saveVisitorProfiles() {
  try {
    localStorage.setItem(VISITOR_STORE_KEY, JSON.stringify(visitorProfiles.slice(-30)));
  } catch {
    // Ignore storage quota errors
  }
}

function bindCurrentVisitorFromLandmarks(landmarks) {
  const desc = computeFaceDescriptor(landmarks);
  const matched = desc ? matchVisitorProfile(visitorProfiles, desc) : null;
  if (matched) {
    matched.profile.lastSeenAt = Date.now();
    matched.profile.descriptor = updateRunningDescriptor(matched.profile.descriptor, desc, 0.15);
    currentVisitor = matched.profile;
    saveVisitorProfiles();
    return { profile: currentVisitor, isReturning: true };
  }
  const newProfile = {
    id: `v_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
    name: '',
    audience: 'adult',
    descriptor: desc,
    history: [],
    lastSeenAt: Date.now(),
  };
  visitorProfiles.push(newProfile);
  if (visitorProfiles.length > 30) visitorProfiles = visitorProfiles.slice(-30);
  currentVisitor = newProfile;
  saveVisitorProfiles();
  return { profile: newProfile, isReturning: false };
}

function rememberVisitorName(name) {
  const clean = String(name || '').trim();
  if (!clean) return;
  if (!currentVisitor) {
    currentVisitor = {
      id: `v_${Date.now().toString(36)}`,
      name: clean,
      audience,
      descriptor: null,
      history: [],
      lastSeenAt: Date.now(),
    };
    visitorProfiles.push(currentVisitor);
  } else {
    currentVisitor.name = clean;
    currentVisitor.lastSeenAt = Date.now();
  }
  awaitingVisitorName = false;
  saveVisitorProfiles();
}

let activeAudio = null;
const audioQueue = [];
let speaking = false;
let busy = false;
let activeAbort = null;
let history = [];
let micPaused = false;
let recognition = null;
let recognitionActive = false;
let micRestartTimer = null;
let speechCommitTimer = null;
let speechTurnPendingText = '';
let userSpeaking = false;
let lastSpeakEndedMs = 0;

function setUiState(state, label) {
  statusDot.className = `dot ${state}`;
  statusBadge.textContent = label;
  videoFrameEl.classList.toggle('speaking', state === 'speaking');
}

function setBanner(visible, text = '') {
  bannerEl.classList.toggle('hidden', !visible);
  if (text) bannerText.textContent = text;
}

function updateLiveButton() {
  if (liveMode) {
    liveBtn.textContent = '⏹ Encerrar';
    liveBtn.classList.remove('btn-accent');
    liveBtn.classList.add('btn-live-on');
  } else {
    liveBtn.textContent = '● Iniciar Conversa';
    liveBtn.classList.remove('btn-live-on');
    liveBtn.classList.add('btn-accent');
  }
  livePill.classList.toggle('hidden', !liveMode);
}

function sendWsJson(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function sendGaze(nx, ny, detected, source) {
  const now = performance.now();
  if (now - lastGazeSentMs < 50 && detected) return; // ~20 Hz throttle
  lastGazeSentMs = now;
  gazePill.classList.toggle('hidden', !detected);
  sendWsJson({ type: 'gaze', x: nx, y: ny, detected, source });
}

function stopAudioPlayback() {
  audioQueue.length = 0;
  if (activeAudio) {
    try {
      activeAudio.onended = null;
      activeAudio.pause();
      URL.revokeObjectURL(activeAudio.src);
    } catch {
      // Ignore audio cleanup issues
    }
    activeAudio = null;
  }
}

function finishSpeakingUi() {
  if (activeAudio || audioQueue.length > 0) return;
  speaking = false;
  lastSpeakEndedMs = Date.now();
  interruptBtn.classList.add('hidden');
  if (liveMode && !busy) {
    setUiState('ready', 'Ao vivo');
    scheduleMicRestart(480);
  }
}

function scheduleMicRestart(delayMs = 480) {
  if (micRestartTimer) {
    clearTimeout(micRestartTimer);
    micRestartTimer = null;
  }
  if (!liveMode || micPaused || busy || speaking) return;
  micRestartTimer = setTimeout(() => {
    micRestartTimer = null;
    ensureLiveMicRunning();
  }, delayMs);
}

function startNextQueuedSpeech() {
  if (activeAudio || audioQueue.length === 0) return;
  const next = audioQueue.shift();
  const bin = atob(next.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: next.mime });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.playsInline = true;
  activeAudio = audio;
  speaking = true;
  interruptBtn.classList.remove('hidden');
  setUiState('speaking', 'Falando');
  stopLiveMic();
  audio.onended = () => {
    if (activeAudio === audio) activeAudio = null;
    URL.revokeObjectURL(url);
    if (audioQueue.length > 0) {
      startNextQueuedSpeech();
    } else {
      finishSpeakingUi();
    }
  };
  void audio.play().catch((err) => {
    console.warn('[audio play]', err);
    if (activeAudio === audio) activeAudio = null;
    URL.revokeObjectURL(url);
    if (audioQueue.length > 0) {
      startNextQueuedSpeech();
    } else {
      finishSpeakingUi();
    }
  });
}

function playSpeechBase64(b64, mime = 'audio/mpeg') {
  if (!b64) return;
  audioQueue.push({ b64, mime });
  if (!activeAudio) {
    startNextQueuedSpeech();
  }
}

let frameRendering = false;
let pendingFrameBuffer = null;
async function drawJpegBuffer(arrayBuffer) {
  pendingFrameBuffer = arrayBuffer;
  if (frameRendering) return;
  frameRendering = true;
  try {
    while (pendingFrameBuffer) {
      const nextBuf = pendingFrameBuffer;
      pendingFrameBuffer = null;
      const blob = new Blob([nextBuf], { type: 'image/jpeg' });
      const bmp = await createImageBitmap(blob);
      if (avatarCanvas.width !== bmp.width || avatarCanvas.height !== bmp.height) {
        avatarCanvas.width = bmp.width;
        avatarCanvas.height = bmp.height;
      }
      avatarCtx.drawImage(bmp, 0, 0);
      bmp.close();
      avatarCanvas.classList.add('live');
    }
  } finally {
    frameRendering = false;
  }
}

let cachedWorkerWsUrl = null;
async function resolveAvatarWsUrl() {
  if (!cachedWorkerWsUrl) {
    try {
      const r = await fetch('/api/livez', { cache: 'no-store' });
      if (r.ok) {
        const info = await r.json();
        if (info?.workerWsUrl) cachedWorkerWsUrl = info.workerWsUrl;
      }
    } catch {
      // fall back to local/same-origin proxy below
    }
  }
  const sidParam = `sessionId=${encodeURIComponent(clientSessionId)}`;
  if (
    cachedWorkerWsUrl &&
    (cachedWorkerWsUrl.startsWith('wss://') ||
      (cachedWorkerWsUrl.startsWith('ws://') &&
        ['localhost', '127.0.0.1'].includes(location.hostname)))
  ) {
    const sep = cachedWorkerWsUrl.includes('?') ? '&' : '?';
    return `${cachedWorkerWsUrl}${sep}${sidParam}`;
  }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws/avatar?${sidParam}`;
}

function connectAvatarWs() {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    let settled = false;
    let retryTimer = null;

    const fail = (message) => {
      if (settled) return;
      settled = true;
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      reject(new Error(message));
    };

    const tryConnect = async () => {
      if (settled || !liveMode) {
        fail('A conexão com a Sibem foi encerrada antes de iniciar. Tente novamente.');
        return;
      }
      const wsUrl = await resolveAvatarWsUrl();
      if (settled || !liveMode) {
        fail('A conexão com a Sibem foi encerrada antes de iniciar. Tente novamente.');
        return;
      }

      const socket = new WebSocket(wsUrl);
      socket.binaryType = 'arraybuffer';
      ws = socket;
      let retryScheduled = false;

      const scheduleRetryOrFail = (errMsg) => {
        if (settled || retryScheduled) return;
        if (liveMode && Date.now() - startedAt < 150000) {
          retryScheduled = true;
          setBanner(true, 'Ligando a Sibem... aguarde alguns instantes.');
          retryTimer = setTimeout(() => {
            retryTimer = null;
            void tryConnect();
          }, 2500);
          return;
        }
        fail(errMsg);
      };

      socket.onmessage = (ev) => {
        if (typeof ev.data !== 'string') {
          void drawJpegBuffer(ev.data);
          return;
        }
        const msg = JSON.parse(ev.data);
        if (msg.type === 'ready') {
          if (!settled) {
            settled = true;
            if (retryTimer) {
              clearTimeout(retryTimer);
              retryTimer = null;
            }
            resolve(socket);
          }
          setBanner(false);
          setUiState('ready', 'Ao vivo');
        } else if (msg.type === 'status') {
          if (msg.message) setBanner(true, msg.message);
        } else if (msg.type === 'telemetry') {
          // Telemetry omitted from end-user UI
        } else if (msg.type === 'speech_start') {
          playSpeechBase64(msg.audioBase64, msg.mime || 'audio/mpeg');
        } else if (msg.type === 'speech_end') {
          if (!activeAudio && audioQueue.length === 0) {
            finishSpeakingUi();
          }
        }
      };

      socket.onerror = () => {
        scheduleRetryOrFail('Não foi possível conectar ao atendimento ao vivo. Tente novamente.');
      };

      socket.onclose = () => {
        if (ws === socket) ws = null;
        if (telemetryBadge) telemetryBadge.classList.add('hidden');
        pendingFrameBuffer = null;
        avatarCanvas.classList.remove('live');
        if (!settled) {
          scheduleRetryOrFail('A conexão com a Sibem foi encerrada antes de iniciar. Tente novamente.');
          return;
        }
        if (liveMode) {
          stopLive();
          setUiState('error', 'Desconectado');
          transcriptText.textContent = 'A conexão com a Sibem foi encerrada. Clique em Iniciar Conversa para ligar novamente.';
        }
      };
    };

    void tryConnect();
  });
}

async function startLive({ greet = true } = {}) {
  if (liveMode) return;
  liveMode = true;
  updateLiveButton();
  setUiState('connecting', 'Ligando...');
  setBanner(true, 'Ligando a Sibem... na primeira conexão pode levar cerca de 30 segundos.');

  try {
    await connectAvatarWs();
    ensureLiveMicRunning();
    if (greet) {
      if (!currentVisitor?.name) awaitingVisitorName = true;
      void runTurn({ mode: 'greet' });
    }
  } catch (err) {
    liveMode = false;
    updateLiveButton();
    setBanner(false);
    setUiState('error', 'Indisponível');
    transcriptText.textContent = err.message;
  }
}

function stopLive() {
  const wasActive = liveMode || Boolean(ws);
  liveMode = false;
  updateLiveButton();
  stopLiveMic();
  stopAudioPlayback();
  if (activeAbort) activeAbort.abort();
  if (ws) {
    try {
      ws.close(1000, 'user_stop');
    } catch {
      // ignore
    }
    ws = null;
  }
  if (cameraActive && wasActive) {
    void toggleWebcam();
  }
  saveVisitorProfiles();
  currentVisitor = null;
  pendingFrameBuffer = null;
  avatarCanvas.classList.remove('live');
  if (telemetryBadge) telemetryBadge.classList.add('hidden');
  gazePill.classList.add('hidden');
  visitorGreeted = false;
  stableFaceFrames = 0;
  awaitingNewVisitorGreet = false;
  awaitingVisitorName = false;
  candidateExpr = '';
  exprHoldStartMs = 0;
  setUiState('ready', 'Aguardando');
  statusDot.className = 'dot';
}

const cleanupSessionOnExit = () => {
  if (liveMode || ws || cameraActive) {
    stopLive();
  }
};
window.addEventListener('pagehide', cleanupSessionOnExit);
window.addEventListener('beforeunload', cleanupSessionOnExit);

async function ensureVisionModels() {
  if (faceLandmarker && gestureRecognizer) return;
  const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  const createWithDelegate = async (creator) => {
    try {
      return await creator('GPU');
    } catch {
      return await creator('CPU');
    }
  };
  if (!faceLandmarker) {
    faceLandmarker = await createWithDelegate((delegate) =>
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/mediapipe/face_landmarker.task',
          delegate,
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
      }),
    );
  }
  if (!gestureRecognizer) {
    gestureRecognizer = await createWithDelegate((delegate) =>
      GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/mediapipe/gesture_recognizer.task',
          delegate,
        },
        runningMode: 'VIDEO',
        numHands: 2,
      }),
    );
  }
}

const CHIPS_BY_AUDIENCE = {
  adult: [
    {
      label: '🏛️ Trabalho do MCTI e Lei do Bem',
      q: 'Como funciona o trabalho do MCTI com a Lei do Bem e os números do Guia 2026?',
    },
    {
      label: 'Regra de Ouro PD&I',
      q: 'Qual é a Regra de Ouro da PD&I segundo o Guia MCTI/UFF?',
    },
    {
      label: 'Software e TIC',
      q: 'Projeto de software ERP ou tecnologia madura entra na Lei do Bem?',
    },
    {
      label: 'TRL e FORMP&D',
      q: 'Como funciona o TRL e os limites de caracteres do FORMP&D?',
    },
    {
      label: '✨ Como a Sibem funciona?',
      q: 'Como você foi criada e quais tecnologias de inteligência artificial fazem você funcionar?',
    },
    {
      label: '🎯 Meu projeto entra?',
      q: 'Como sei se o projeto da minha empresa entra na Lei do Bem? Pode fazer uma simulação rápida comigo e dar exemplos?',
    },
  ],
  child: [
    {
      label: '🤖 O que faz o MCTI na feira?',
      q: 'O que o MCTI faz para ajudar cientistas e inventores no Brasil?',
    },
    {
      label: '🚀 Desafio das invenções!',
      q: 'Vamos brincar de desafio de Sim ou Não sobre ciência e invenções?',
    },
    {
      label: '🎮 Robôs e jogos são ciência?',
      q: 'Criar jogos, robôs e aplicativos também é pesquisa e inovação?',
    },
    {
      label: '🦕 Curiosidade de hoje',
      q: 'Pesquise e me conte uma curiosidade super divertida de ciência ou espaço!',
    },
    {
      label: '✨ Como você ganha vida?',
      q: 'Sibem, como você foi criada e como consegue me ver, ouvir e conversar comigo?',
    },
  ],
};

function renderChipsForAudience(nextAudience) {
  const list = CHIPS_BY_AUDIENCE[nextAudience === 'child' ? 'child' : 'adult'];
  chips.innerHTML = '';
  for (const item of list) {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.type = 'button';
    btn.setAttribute('data-q', item.q);
    btn.textContent = item.label;
    chips.appendChild(btn);
  }
}

function updateAudienceUi(nextAudience) {
  audience = nextAudience === 'child' ? 'child' : 'adult';
  audiencePill.classList.remove('hidden');
  audiencePill.classList.toggle('child', audience === 'child');
  audiencePill.textContent = audience === 'child' ? '🧒 Infantil' : '🧑 Adulto';
  renderChipsForAudience(audience);
}

function showGestureUi(label) {
  gesturePill.textContent = label;
  gesturePill.classList.remove('hidden');
  if (gestureHideTimer) clearTimeout(gestureHideTimer);
  gestureHideTimer = setTimeout(() => {
    gesturePill.classList.add('hidden');
  }, 2400);
}

const GESTURE_LABELS = {
  thumbs_up: '👍 Joinha!',
  thumbs_down: '👎 Negativo!',
  head_yes: '✅ Sim (Aceno)!',
  head_no: '🙅 Não (Cabeça)!',
  wave: '👋 Aceno!',
  victory: '✌️ Vitória!',
  smile: '😊 Sorriso!',
  joy: '😄 Alegria / Felicidade!',
  sadness: '🥺 Tristeza',
  surprise: '😲 Surpresa!',
};

const headMotionHistory = [];
let lastHeadDetectCooldownMs = 0;

function detectHeadGestureFromLandmarks(landmarks, tsMs) {
  if (busy || speaking || userSpeaking) {
    headMotionHistory.length = 0;
    return '';
  }

  const noseTip = landmarks[1];
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  if (!noseTip || !leftCheek || !rightCheek || !leftEye || !rightEye) return '';

  const faceW = Math.hypot(rightCheek.x - leftCheek.x, rightCheek.y - leftCheek.y);
  if (faceW < 0.06) return '';

  const midCheekX = (leftCheek.x + rightCheek.x) * 0.5;
  const midEyeY = (leftEye.y + rightEye.y) * 0.5;
  const yawRel = (noseTip.x - midCheekX) / faceW;
  const pitchRel = (noseTip.y - midEyeY) / faceW;

  headMotionHistory.push({ t: tsMs, yawRel, pitchRel });
  while (headMotionHistory.length > 0 && tsMs - headMotionHistory[0].t > 850) {
    headMotionHistory.shift();
  }
  if (headMotionHistory.length < 10 || tsMs - lastHeadDetectCooldownMs < 4500) return '';

  let minY = Infinity, maxY = -Infinity, sumY = 0;
  let minP = Infinity, maxP = -Infinity, sumP = 0;
  for (const s of headMotionHistory) {
    if (s.yawRel < minY) minY = s.yawRel;
    if (s.yawRel > maxY) maxY = s.yawRel;
    sumY += s.yawRel;
    if (s.pitchRel < minP) minP = s.pitchRel;
    if (s.pitchRel > maxP) maxP = s.pitchRel;
    sumP += s.pitchRel;
  }
  const meanY = sumY / headMotionHistory.length;
  const meanP = sumP / headMotionHistory.length;
  const yawRange = maxY - minY;
  const pitchRange = maxP - minP;

  const countCrossings = (key, mean, deadband) => {
    let prevSign = 0;
    let flips = 0;
    for (const s of headMotionHistory) {
      const d = s[key] - mean;
      const sign = d > deadband ? 1 : d < -deadband ? -1 : 0;
      if (sign !== 0 && prevSign !== 0 && sign !== prevSign) flips += 1;
      if (sign !== 0) prevSign = sign;
    }
    return flips;
  };

  const yawFlips = countCrossings('yawRel', meanY, 0.055);
  const pitchFlips = countCrossings('pitchRel', meanP, 0.032);

  if (
    pitchRange >= 0.115 &&
    pitchFlips >= 3 &&
    maxP - meanP > 0.04 &&
    meanP - minP > 0.04 &&
    pitchRange > yawRange * 1.5
  ) {
    headMotionHistory.length = 0;
    lastHeadDetectCooldownMs = tsMs;
    return 'head_yes';
  }
  if (
    yawRange >= 0.24 &&
    yawFlips >= 3 &&
    maxY - meanY > 0.085 &&
    meanY - minY > 0.085 &&
    yawRange > pitchRange * 1.6
  ) {
    headMotionHistory.length = 0;
    lastHeadDetectCooldownMs = tsMs;
    return 'head_no';
  }
  return '';
}

function mapMediaPipeGesture(rawName) {
  if (rawName === 'Thumb_Up') return 'thumbs_up';
  if (rawName === 'Thumb_Down') return 'thumbs_down';
  if (rawName === 'Open_Palm') return 'wave';
  if (rawName === 'Victory') return 'victory';
  return '';
}

function handleConfirmedGesture(kind) {
  if (!kind || userSpeaking) return;
  const now = Date.now();
  const prev = lastGestureTriggerMs[kind] || 0;
  if (now - prev < 9000) return;
  lastGestureTriggerMs[kind] = now;

  showGestureUi(GESTURE_LABELS[kind] || kind);
  // Map expanded facial expressions safely to Modal worker reaction kinds
  const workerReactionKind = kind === 'joy' || kind === 'surprise' ? 'smile' : kind;
  if (kind !== 'sadness') {
    sendWsJson({ type: 'reaction', kind: workerReactionKind, strength: 1.0, duration: 2.1 });
    if (!liveMode || !ws || ws.readyState !== WebSocket.OPEN) {
      void fetch('/api/reaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: clientSessionId, kind: workerReactionKind, strength: 1.0, duration: 2.1 }),
      }).catch(() => {});
    }
  } else {
    sendWsJson({ type: 'smile', level: 0 });
  }

  // Speak back when idle & cooled down
  if (liveMode && !busy && !speaking && !userSpeaking && now - lastVocalReactionMs > 16000) {
    lastVocalReactionMs = now;
    void runTurn({ mode: 'gesture', gesture: kind });
  }
}

const audienceCanvas = document.createElement('canvas');
audienceCanvas.width = 224;
audienceCanvas.height = 224;
const audienceCtx = audienceCanvas.getContext('2d');

async function checkAudienceFromVideo(landmarks, { force = false, timeoutMs = 3500 } = {}) {
  if (audienceManualOverride || !webcamVideo.videoWidth) return audience;
  if (audienceCheckInFlight) return audience;
  const now = Date.now();
  if (!force && now - lastAudienceCheckMs < 24000) return audience;
  lastAudienceCheckMs = now;
  audienceCheckInFlight = true;
  try {
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;
    for (const pt of landmarks) {
      if (pt.x < minX) minX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y > maxY) maxY = pt.y;
    }
    const vw = webcamVideo.videoWidth;
    const vh = webcamVideo.videoHeight;
    const padX = (maxX - minX) * 0.45;
    const padY = (maxY - minY) * 0.45;
    const sx = Math.max(0, Math.floor((minX - padX) * vw));
    const sy = Math.max(0, Math.floor((minY - padY) * vh));
    const sw = Math.min(vw - sx, Math.ceil((maxX - minX + padX * 2) * vw));
    const sh = Math.min(vh - sy, Math.ceil((maxY - minY + padY * 2) * vh));
    if (sw < 32 || sh < 32) return audience;

    audienceCtx.drawImage(webcamVideo, sx, sy, sw, sh, 0, 0, 224, 224);
    const dataUrl = audienceCanvas.toDataURL('image/jpeg', 0.78);
    const b64 = dataUrl.split(',')[1];
    const r = await fetch('/api/audience', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageBase64: b64 }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (r.ok && !audienceManualOverride) {
      const data = await r.json();
      if (data.audience) updateAudienceUi(data.audience);
    }
  } catch {
    // Keep existing audience on network errors or timeout
  } finally {
    audienceCheckInFlight = false;
  }
  return audience;
}

async function greetNewVisitor(landmarks) {
  if (awaitingNewVisitorGreet || visitorGreeted || busy || speaking || userSpeaking) return;
  awaitingNewVisitorGreet = true;
  try {
    const { profile, isReturning } = bindCurrentVisitorFromLandmarks(landmarks);
    if (isReturning && Array.isArray(profile.history)) {
      history = profile.history.slice(-12);
    } else if (hadPreviousVisitor || !isReturning) {
      history = [];
    }
    if (profile.name) {
      awaitingVisitorName = false;
      if (profile.audience && !audienceManualOverride) {
        updateAudienceUi(profile.audience);
      }
      topicTag.textContent = `Olá novamente, ${profile.name}`;
    } else {
      awaitingVisitorName = true;
      topicTag.textContent = 'Boas-vindas • Estande MCTI';
      await checkAudienceFromVideo(landmarks, { force: true, timeoutMs: 2200 });
      profile.audience = audience;
      saveVisitorProfiles();
    }
    if (!cameraActive || !liveMode || ws?.readyState !== WebSocket.OPEN || busy || speaking || userSpeaking) {
      return;
    }
    visitorGreeted = true;
    hadPreviousVisitor = true;
    void runTurn({ mode: 'greet' });
  } finally {
    awaitingNewVisitorGreet = false;
  }
}

audiencePill.addEventListener('click', () => {
  audienceManualOverride = true;
  updateAudienceUi(audience === 'adult' ? 'child' : 'adult');
});

async function toggleWebcam() {
  if (cameraActive) {
    cameraActive = false;
    saveVisitorProfiles();
    currentVisitor = null;
    visitorGreeted = false;
    stableFaceFrames = 0;
    awaitingNewVisitorGreet = false;
    awaitingVisitorName = false;
    candidateExpr = '';
    exprHoldStartMs = 0;
    faceMissingSinceMs = 0;
    cameraBtn.textContent = '📷 Câmera: OFF';
    cameraBtn.classList.remove('active');
    cameraPip.classList.add('hidden');
    if (webcamLoopId) cancelAnimationFrame(webcamLoopId);
    webcamLoopId = null;
    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
      webcamStream = null;
    }
    sendWsJson({ type: 'reset_gaze' });
    gazePill.classList.add('hidden');
    gesturePill.classList.add('hidden');
    return;
  }

  try {
    setBanner(true, 'Ativando câmera...');
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 480 }, height: { ideal: 360 }, facingMode: 'user' },
      audio: false,
    });
    webcamVideo.srcObject = webcamStream;
    await webcamVideo.play();
    await ensureVisionModels();
    setBanner(false);

    if (mouseGazeActive) {
      mouseGazeActive = false;
      if (mouseBtn) mouseBtn.classList.remove('active');
    }

    cameraActive = true;
    visitorGreeted = false;
    stableFaceFrames = 0;
    awaitingNewVisitorGreet = false;
    faceMissingSinceMs = 0;
    cameraBtn.textContent = '📷 Câmera: ON';
    cameraBtn.classList.add('active');
    cameraPip.classList.remove('hidden');
    updateAudienceUi(audience);

    if (!liveMode) {
      void startLive({ greet: false });
    }

    let lastVideoTime = -1;
    let lastFaceRunMs = 0;
    let lastGestureRunMs = 0;
    const detectLoop = () => {
      if (!cameraActive) return;
      const tsMs = performance.now();
      if (webcamVideo.readyState >= 2 && webcamVideo.currentTime !== lastVideoTime && tsMs - lastFaceRunMs >= 70) {
        lastVideoTime = webcamVideo.currentTime;
        lastFaceRunMs = tsMs;
        const faceRes = faceLandmarker.detectForVideo(webcamVideo, tsMs);
        webcamOverlayCtx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);

        const landmarks = faceRes?.faceLandmarks?.[0];
        const blendshapes = faceRes?.faceBlendshapes?.[0]?.categories || [];

        const leftCheek = landmarks?.[234];
        const rightCheek = landmarks?.[454];
        const faceW =
          leftCheek && rightCheek
            ? Math.hypot(rightCheek.x - leftCheek.x, rightCheek.y - leftCheek.y)
            : landmarks?.length
              ? 0.14
              : 0;

        if (landmarks?.length && faceW >= 0.08) {
          if (faceMissingSinceMs && Date.now() - faceMissingSinceMs > 4500) {
            saveVisitorProfiles();
            currentVisitor = null;
            visitorGreeted = false;
            awaitingNewVisitorGreet = false;
            awaitingVisitorName = false;
            lastAudienceCheckMs = 0;
            audienceManualOverride = false;
          }
          faceMissingSinceMs = 0;

          if (!visitorGreeted && !awaitingNewVisitorGreet && !userSpeaking) {
            stableFaceFrames += 1;
            if (
              stableFaceFrames >= 10 &&
              liveMode &&
              ws?.readyState === WebSocket.OPEN &&
              !busy &&
              !speaking &&
              !userSpeaking
            ) {
              void greetNewVisitor(landmarks);
            }
          } else {
            void checkAudienceFromVideo(landmarks);
          }

          const leftEye = landmarks[33] || landmarks[168];
          const rightEye = landmarks[263] || landmarks[168];
          const cx = (leftEye.x + rightEye.x) * 0.5;
          const cy = (leftEye.y + rightEye.y) * 0.5;
          const nx = Math.max(-1, Math.min(1, (0.5 - cx) * 2.2));
          const ny = Math.max(-1, Math.min(1, (cy - 0.5) * 2.2));

          const ox = cx * webcamOverlay.width;
          const oy = cy * webcamOverlay.height;
          webcamOverlayCtx.strokeStyle = '#34d399';
          webcamOverlayCtx.lineWidth = 2;
          webcamOverlayCtx.beginPath();
          webcamOverlayCtx.arc(ox, oy, 9, 0, Math.PI * 2);
          webcamOverlayCtx.stroke();

          // Classify facial expressions (alegria/felicidade, sorriso, tristeza, surpresa)
          const { expression, smileAvg } = classifyFaceExpression(blendshapes);
          if (tsMs - lastSmileSentMs > 100) {
            lastSmileSentMs = tsMs;
            sendWsJson({ type: 'smile', level: smileAvg > 0.35 ? Math.min(1, (smileAvg - 0.3) * 1.6) : 0 });
          }

          if (currentVisitor && !expression && tsMs - lastDescriptorRefreshMs > 2200) {
            const sampleDesc = computeFaceDescriptor(landmarks);
            if (sampleDesc) {
              currentVisitor.descriptor = updateRunningDescriptor(currentVisitor.descriptor, sampleDesc, 0.1);
              lastDescriptorRefreshMs = tsMs;
            }
          }

          if (expression && !busy && !speaking && !userSpeaking) {
            const minHoldMs = expression === 'sadness' ? 1000 : expression === 'smile' ? 900 : 850;
            if (candidateExpr !== expression) {
              candidateExpr = expression;
              exprHoldStartMs = tsMs;
            } else if (exprHoldStartMs && tsMs - exprHoldStartMs >= minHoldMs) {
              exprHoldStartMs = 0;
              candidateExpr = '';
              handleConfirmedGesture(expression);
            }
          } else {
            candidateExpr = '';
            exprHoldStartMs = 0;
          }

          const exprTag =
            expression === 'joy'
              ? '😄 Alegre/Feliz'
              : expression === 'sadness'
                ? '🥺 Triste'
                : expression === 'surprise'
                  ? '😲 Surpreso'
                  : expression === 'smile' || smileAvg > 0.55
                    ? '😊 Sorrindo'
                    : 'Presente';
          const visitorPrefix = currentVisitor?.name ? `${currentVisitor.name} • ` : '';
          pipLabel.textContent = `${audience === 'child' ? '🧒' : '🧑'} ${visitorPrefix}${exprTag}`;
          sendGaze(nx, ny, true, 'webcam');

          const headGest = detectHeadGestureFromLandmarks(landmarks, tsMs);
          if (headGest) {
            handleConfirmedGesture(headGest);
          }
        } else {
          headMotionHistory.length = 0;
          stableFaceFrames = 0;
          candidateExpr = '';
          exprHoldStartMs = 0;
          if (!busy && !speaking && !userSpeaking) {
            if (!faceMissingSinceMs) faceMissingSinceMs = Date.now();
            else if (Date.now() - faceMissingSinceMs > 4500 && (visitorGreeted || currentVisitor)) {
              saveVisitorProfiles();
              currentVisitor = null;
              visitorGreeted = false;
              awaitingNewVisitorGreet = false;
              awaitingVisitorName = false;
              lastAudienceCheckMs = 0;
              audienceManualOverride = false;
            }
          }
          pipLabel.textContent = 'Aguardando visitante...';
          sendGaze(0, 0, false, 'webcam');
        }

        // Run hand gesture recognizer at ~8 Hz
        if (gestureRecognizer && tsMs - lastGestureRunMs > 120) {
          lastGestureRunMs = tsMs;
          const gRes = gestureRecognizer.recognizeForVideo(webcamVideo, tsMs);
          const topGesture = gRes?.gestures?.[0]?.[0];
          const topRaw = topGesture?.categoryName || '';
          const topScore = Number(topGesture?.score || 0);
          const minScore = topRaw === 'Thumb_Down' ? 0.78 : 0.72;
          const mapped = topScore >= minScore ? mapMediaPipeGesture(topRaw) : '';
          if (mapped && !busy && !speaking && !userSpeaking) {
            if (candidateGesture === mapped) {
              candidateGestureFrames += 1;
              if (candidateGestureFrames >= 4) {
                candidateGestureFrames = 0;
                handleConfirmedGesture(mapped);
              }
            } else {
              candidateGesture = mapped;
              candidateGestureFrames = 1;
            }
          } else {
            candidateGesture = '';
            candidateGestureFrames = 0;
          }
        }
      }
      webcamLoopId = requestAnimationFrame(detectLoop);
    };
    webcamLoopId = requestAnimationFrame(detectLoop);
  } catch {
    setBanner(false);
    cameraActive = false;
    cameraBtn.textContent = '📷 Câmera: OFF';
    transcriptText.textContent = 'Não foi possível acessar a câmera. Você pode continuar conversando digitando ou usando o microfone!';
  }
}

if (mouseBtn) {
  stageEl.addEventListener('pointermove', (ev) => {
    if (!mouseGazeActive || !liveMode) return;
    const rect = stageEl.getBoundingClientRect();
    const nx = Math.max(-1, Math.min(1, ((ev.clientX - rect.left) / rect.width) * 2 - 1));
    const ny = Math.max(-1, Math.min(1, ((ev.clientY - rect.top) / rect.height) * 2 - 1));
    sendGaze(nx, ny, true, 'mouse');
  });

  stageEl.addEventListener('pointerleave', () => {
    if (mouseGazeActive && liveMode) {
      sendGaze(0, 0, false, 'mouse');
    }
  });

  mouseBtn.addEventListener('click', () => {
    if (cameraActive) void toggleWebcam();
    mouseGazeActive = !mouseGazeActive;
    mouseBtn.classList.toggle('active', mouseGazeActive);
    if (mouseGazeActive && !liveMode) {
      void startLive({ greet: false });
    } else if (!mouseGazeActive) {
      sendWsJson({ type: 'reset_gaze' });
      gazePill.classList.add('hidden');
    }
  });
}

cameraBtn.addEventListener('click', () => void toggleWebcam());

liveBtn.addEventListener('click', () => {
  if (liveMode) stopLive();
  else void startLive({ greet: true });
});

fullscreenBtn.addEventListener('click', () => {
  stageEl.classList.toggle('expanded');
});

if (portraitInput) {
  portraitInput.addEventListener('change', async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
    const b64 = btoa(bin);
    setBanner(true, 'Atualizando imagem da Sibem...');
    try {
      const r = await fetch('/api/portrait', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64: b64 }),
      });
      if (!r.ok) throw new Error('Não foi possível atualizar a imagem');
      fallbackEl.src = `/api/portrait/current?t=${Date.now()}`;
    } finally {
      setBanner(false);
      portraitInput.value = '';
    }
  });
}

function clearSpeechCommitTimer() {
  if (speechCommitTimer) {
    clearTimeout(speechCommitTimer);
    speechCommitTimer = null;
  }
}

function commitSpokenTurn() {
  clearSpeechCommitTimer();
  const text = speechTurnPendingText.trim();
  speechTurnPendingText = '';
  userSpeaking = false;
  if (!text || busy || speaking) return;
  stopLiveMic();
  promptInput.value = '';
  void runTurn({ userMessage: text, mode: 'chat' });
}

function stopLiveMic() {
  clearSpeechCommitTimer();
  speechTurnPendingText = '';
  userSpeaking = false;
  if (micRestartTimer) {
    clearTimeout(micRestartTimer);
    micRestartTimer = null;
  }
  if (recognition && recognitionActive) {
    try {
      recognition.abort();
    } catch {
      // ignore
    }
  }
  recognitionActive = false;
  micBtn.classList.remove('recording');
  micStateEl.textContent = liveMode && micPaused ? 'Microfone pausado' : '';
}

function ensureLiveMicRunning() {
  if (!SpeechRecognition) {
    micStateEl.textContent = 'Microfone indisponível neste navegador — digite sua pergunta abaixo';
    return;
  }
  if (!liveMode || micPaused || busy || speaking) return;
  if (recognitionActive) return;

  if (!recognition) {
    recognition = new SpeechRecognition();
    recognition.lang = 'pt-BR';
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      recognitionActive = true;
      micBtn.classList.add('recording');
      micStateEl.textContent = '🎤 Pode falar, estou ouvindo...';
    };

    recognition.onresult = (event) => {
      if (busy || speaking || Date.now() - lastSpeakEndedMs < 450) return;

      let fullFinal = '';
      let fullInterim = '';
      for (let i = 0; i < event.results.length; i += 1) {
        const res = event.results[i];
        const txt = res?.[0]?.transcript || '';
        if (res.isFinal) fullFinal += `${txt} `;
        else fullInterim += `${txt} `;
      }
      const combined = `${fullFinal} ${fullInterim}`.replace(/\s+/g, ' ').trim();
      if (!combined) return;

      // Mark userSpeaking immediately so gestures/greets don't talk over the user
      userSpeaking = true;
      speechTurnPendingText = combined;
      micStateEl.textContent = `🎤 Ouvindo: "${combined}"`;

      // Wait 1350ms of silence after last words before submitting so Sibem never cuts mid-sentence
      clearSpeechCommitTimer();
      speechCommitTimer = setTimeout(() => {
        commitSpokenTurn();
      }, 1350);
    };

    recognition.onerror = (ev) => {
      recognitionActive = false;
      clearSpeechCommitTimer();
      userSpeaking = false;
      speechTurnPendingText = '';
      micBtn.classList.remove('recording');
      if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
        micPaused = true;
        micStateEl.textContent = 'Microfone sem permissão (use o texto)';
        return;
      }
      if (liveMode && !micPaused && !busy && !speaking) {
        scheduleMicRestart(450);
      }
    };

    recognition.onend = () => {
      recognitionActive = false;
      micBtn.classList.remove('recording');
      if (speechTurnPendingText.trim() && !busy && !speaking) {
        commitSpokenTurn();
        return;
      }
      clearSpeechCommitTimer();
      userSpeaking = false;
      if (liveMode && !micPaused && !busy && !speaking) {
        scheduleMicRestart(300);
      }
    };
  }

  try {
    recognition.start();
  } catch {
    scheduleMicRestart(350);
  }
}

micBtn.addEventListener('click', () => {
  micPaused = !micPaused;
  if (micPaused) stopLiveMic();
  else if (liveMode) ensureLiveMicRunning();
});

interruptBtn.addEventListener('click', async () => {
  stopAudioPlayback();
  speaking = false;
  lastSpeakEndedMs = Date.now();
  interruptBtn.classList.add('hidden');
  sendWsJson({ type: 'interrupt' });
  await fetch('/api/interrupt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: clientSessionId }),
  }).catch(() => {});
  if (liveMode && !busy) {
    setUiState('ready', 'Ao vivo');
    scheduleMicRestart(480);
  }
});

async function runTurn({ userMessage = '', mode = 'chat', gesture = '' } = {}) {
  if (busy) return;
  if (!liveMode) {
    await startLive({ greet: false });
  }
  busy = true;
  sendBtn.disabled = true;
  stopAudioPlayback();
  sendWsJson({ type: 'interrupt' });
  stopLiveMic();

  if (mode === 'chat' && userMessage) {
    const detectedName = extractVisitorName(userMessage, awaitingVisitorName);
    if (detectedName) {
      rememberVisitorName(detectedName);
    }
  }
  const activeVisitorName = currentVisitor?.name || '';

  setUiState('thinking', 'Pensando...');
  topicTag.textContent =
    mode === 'gesture'
      ? `${GESTURE_LABELS[gesture] || 'Expressão percebida'}${activeVisitorName ? ` • ${activeVisitorName}` : ''}`
      : mode === 'greet'
        ? activeVisitorName
          ? `Bem-vindo de volta, ${activeVisitorName}`
          : audience === 'child'
            ? 'Espaço Ciência • Modo Infantil'
            : 'Estande MCTI • Boas-vindas'
        : `${activeVisitorName ? `${activeVisitorName} • ` : ''}Buscando resposta...`;
  if (mode === 'chat') {
    transcriptText.textContent = 'Preparando resposta...';
  }

  activeAbort = new AbortController();
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        userMessage,
        history,
        mode,
        avatarId: activeAvatarId,
        audience,
        gesture,
        visitorName: activeVisitorName,
        sessionId: clientSessionId,
      }),
      signal: activeAbort.signal,
    });
    if (!res.ok || !res.body) throw new Error('Instabilidade na conexão. Tente novamente.');

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let finalReply = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const blocks = buf.split('\n\n');
      buf = blocks.pop() || '';
      for (const block of blocks) {
        const lines = block.split('\n');
        let evName = 'message';
        let dataLine = '';
        for (const ln of lines) {
          if (ln.startsWith('event:')) evName = ln.slice(6).trim();
          else if (ln.startsWith('data:')) dataLine += ln.slice(5).trim();
        }
        if (!dataLine) continue;
        const payload = JSON.parse(dataLine);
        if (evName === 'knowledge') {
          if (payload.refs?.length && payload.webResults?.length) {
            topicTag.textContent = `Guia Lei do Bem 2026 (p. ${payload.refs[0].pages}) + Pesquisa`;
          } else if (payload.webResults?.length) {
            topicTag.textContent = `🌐 Pesquisa • ${payload.webResults[0].title.slice(0, 42)}`;
          } else if (payload.refs?.length) {
            topicTag.textContent = `Guia MCTI • ${payload.refs[0].title.slice(0, 42)} (p. ${payload.refs[0].pages})`;
          }
        } else if (evName === 'delta') {
          transcriptText.textContent = payload.fullText;
        } else if (evName === 'spoken' || evName === 'spoken_chunk') {
          if (payload?.routed === false && payload?.audioBase64) {
            sendWsJson({ type: 'speak', audioBase64: payload.audioBase64, speechId: payload.speechId });
          }
        } else if (evName === 'reply') {
          finalReply = payload.text;
          transcriptText.textContent = finalReply;
        } else if (evName === 'error') {
          throw new Error(payload.error);
        }
      }
    }

    if (mode === 'chat' && userMessage && finalReply) {
      history.push({ role: 'user', content: userMessage }, { role: 'assistant', content: finalReply });
      if (history.length > 16) history = history.slice(-16);
      if (currentVisitor) {
        currentVisitor.history = history.slice(-12);
        currentVisitor.audience = audience;
        saveVisitorProfiles();
      }
    } else if (mode === 'greet' && finalReply) {
      history.push({ role: 'assistant', content: finalReply });
      if (currentVisitor) {
        currentVisitor.history = history.slice(-12);
        saveVisitorProfiles();
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      transcriptText.textContent = 'Desculpe, não consegui responder agora. Tente perguntar novamente.';
      setUiState('error', 'Instabilidade');
    }
  } finally {
    busy = false;
    sendBtn.disabled = false;
    activeAbort = null;
    if (liveMode && !speaking && !activeAudio && audioQueue.length === 0) {
      setUiState('ready', 'Ao vivo');
      scheduleMicRestart(480);
    }
  }
}

chatForm.addEventListener('submit', (ev) => {
  ev.preventDefault();
  const q = promptInput.value.trim();
  if (!q) return;
  promptInput.value = '';
  void runTurn({ userMessage: q, mode: 'chat' });
});

chips.addEventListener('click', (ev) => {
  const btn = ev.target.closest('.chip');
  if (!btn) return;
  const q = btn.getAttribute('data-q');
  if (q) void runTurn({ userMessage: q, mode: 'chat' });
});

function renderAvatarDock() {
  if (!avatarDock) return;
  avatarDock.innerHTML = '';
  avatarDock.classList.toggle('hidden', avatarList.length <= 1);
  if (avatarList.length <= 1) return;
  for (const av of avatarList) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `avatar-item${av.id === activeAvatarId ? ' active' : ''}`;
    btn.title = `${av.name} — ${av.role}`;
    btn.innerHTML = `
      <img src="${av.image}" alt="${av.name}" />
      <div class="avatar-item-meta">
        <span class="avatar-item-name">${av.name}</span>
        <span class="avatar-item-role">${av.role}</span>
      </div>
    `;
    btn.addEventListener('click', () => void selectAvatar(av.id));
    avatarDock.appendChild(btn);
  }
}

async function selectAvatar(avatarId) {
  if (!avatarId || avatarId === activeAvatarId) return;
  const target = avatarList.find((a) => a.id === avatarId);
  if (!target) return;
  activeAvatarId = target.id;
  brandTitle.textContent = `${target.name} • MCTI`;
  brandSub.textContent = target.role;
  fallbackEl.src = target.image;
  renderAvatarDock();
  try {
    const res = await fetch(`/api/avatar/${encodeURIComponent(avatarId)}`, { method: 'POST' });
    if (!res.ok) throw new Error('Não foi possível selecionar a assistente');
  } catch (err) {
    transcriptText.textContent = err.message;
  }
}

async function loadAvatars() {
  try {
    const res = await fetch('/api/avatars');
    if (!res.ok) return;
    const data = await res.json();
    avatarList = data.avatars || [];
    if (data.activeAvatarId) activeAvatarId = data.activeAvatarId;
    const current = avatarList.find((a) => a.id === activeAvatarId);
    if (current) {
      brandTitle.textContent = `${current.name} • MCTI`;
      brandSub.textContent = current.role;
    }
    renderAvatarDock();
  } catch {
    // Keep default avatar if offline
  }
}

void loadAvatars();
