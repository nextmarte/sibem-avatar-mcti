/**
 * Pure visitor identity, name extraction, and facial expression classifiers
 * used by the Sibem frontend (and testable under Node without DOM/camera).
 */

const NOT_NAMES = new Set([
  'sim', 'nao', 'não', 'oi', 'ola', 'olá', 'tudo', 'bem', 'bom', 'boa', 'dia', 'tarde', 'noite',
  'legal', 'obrigado', 'obrigada', 'sibem', 'mcti', 'uff', 'lei', 'formpd', 'projeto', 'projetos',
  'empresa', 'empresas', 'quero', 'queria', 'gostaria', 'saber', 'como', 'qual', 'quem', 'onde',
  'quando', 'porque', 'por', 'que', 'sobre', 'voce', 'você', 'ela', 'ele', 'isso', 'isto', 'aqui',
  'nada', 'muito', 'pouco', 'mais', 'menos', 'certo', 'ok', 'claro', 'talvez', 'triste', 'feliz',
  'alegre', 'verdade', 'ciencia', 'ciência', 'espaco', 'espaço', 'robo', 'robô', 'planeta',
  'ajudante', 'assistente', 'feira', 'inovacao', 'inovação', 'pesquisa', 'desenvolvimento',
  'brasil', 'governo', 'estande', 'brincar', 'desafio', 'pergunta', 'duvida', 'dúvida',
  'curiosidade', 'tecnologia', 'pessoal', 'gente', 'amigo', 'amiga', 'crianca', 'criança',
  'adulto', 'visitante', 'nome', 'chamo', 'chamar', 'meu', 'minha', 'seu', 'sua',
]);

function normalizeToken(word) {
  return String(word || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
}

function toTitleCaseName(raw) {
  const prepositions = new Set(['da', 'de', 'do', 'das', 'dos', 'e']);
  return raw
    .split(/\s+/)
    .filter(Boolean)
    .map((part, idx) => {
      const lower = part.toLowerCase();
      if (idx > 0 && prepositions.has(lower)) return lower;
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(' ');
}

function validateCandidateName(candidate) {
  const cleaned = String(candidate || '')
    .replace(/[^A-Za-zÀ-ÖØ-öø-ÿ\s'-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned) return '';

  const words = cleaned.split(' ').filter((w) => w.length >= 2 && !NOT_NAMES.has(normalizeToken(w)));
  if (words.length === 0 || words.length > 2) return '';
  if (words.some((w) => w.length > 18)) return '';
  return toTitleCaseName(words.join(' '));
}

/**
 * Extracts a Brazilian Portuguese visitor name from spoken text, either via
 * explicit self-introduction ("Meu nome é João", "Sou a Maria") or from a short
 * direct answer when Sibem has just asked for their name.
 */
export function extractVisitorName(text, awaitingName = false) {
  const raw = String(text || '').trim();
  if (!raw || raw.length > 140) return '';

  const introPatterns = [
    /(?:o\s+meu\s+nome\s+[eé]|meu\s+nome\s+[eé]|eu\s+me\s+chamo|me\s+chamo|pode\s+me\s+chamar\s+de)\s+([A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'-]+)?)/i,
    /(?:^|[,.!?\s])(?:oi|olá|ola|prazer)?[,\s]*(?:eu\s+)?sou\s+(?:o|a)?\s*([A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'-]+)?)(?=[,.!?\s]|$)/i,
    /(?:^|[,.!?\s])aqui\s+[eé]\s+(?:o|a)?\s*([A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'-]+)?)(?=[,.!?\s]|$)/i,
  ];

  for (const re of introPatterns) {
    const match = raw.match(re);
    if (match?.[1]) {
      // Stop before trailing clauses like "João e quero saber..."
      const trimmedClause = match[1].split(/\s+(?:e|que|quero|queria|gostaria|vim|para|pra)\b/i)[0];
      const valid = validateCandidateName(trimmedClause);
      if (valid) return valid;
    }
  }

  if (awaitingName && !raw.includes('?')) {
    const firstChunk = raw
      .replace(/^(?:oi|olá|ola|e\s+a[ií]|tudo\s+bem|prazer|bom\s+dia|boa\s+tarde|boa\s+noite)[,!.\s]*/i, '')
      .split(/[,.;!?]|\s+(?:e\s+quero|quero|queria|gostaria|como|qual|quem|onde|sobre|me\s+fala|me\s+explica)\b/i)[0]
      .trim();
    const tokens = firstChunk.split(/\s+/).filter(Boolean);
    if (tokens.length >= 1 && tokens.length <= 2) {
      const valid = validateCandidateName(firstChunk);
      if (valid) return valid;
    }
  }

  return '';
}

function getPt(landmarks, idx) {
  const pt = landmarks?.[idx];
  return pt && Number.isFinite(pt.x) && Number.isFinite(pt.y) ? pt : null;
}

/**
 * Computes a scale- and roll-invariant 2D geometric signature from rigid facial
 * landmarks so returning fair visitors can be recognized locally in the browser.
 */
export function computeFaceDescriptor(landmarks) {
  const leOuter = getPt(landmarks, 33);
  const reOuter = getPt(landmarks, 263);
  const leInner = getPt(landmarks, 133);
  const reInner = getPt(landmarks, 362);
  const noseBridge = getPt(landmarks, 168);
  const noseTip = getPt(landmarks, 1);
  const subnasal = getPt(landmarks, 2);
  const leftCheek = getPt(landmarks, 234);
  const rightCheek = getPt(landmarks, 454);
  const chin = getPt(landmarks, 152);
  const forehead = getPt(landmarks, 10);
  const leftNose = getPt(landmarks, 98);
  const rightNose = getPt(landmarks, 327);

  if (
    !leOuter || !reOuter || !leInner || !reInner || !noseBridge || !noseTip ||
    !subnasal || !leftCheek || !rightCheek || !chin || !forehead || !leftNose || !rightNose
  ) {
    return null;
  }

  const dxEye = reOuter.x - leOuter.x;
  const dyEye = reOuter.y - leOuter.y;
  const eyeSpan = Math.hypot(dxEye, dyEye);
  if (eyeSpan < 0.025) return null;

  const midX = (leOuter.x + reOuter.x) * 0.5;
  const midY = (leOuter.y + reOuter.y) * 0.5;
  const cosA = dxEye / eyeSpan;
  const sinA = dyEye / eyeSpan;

  const project = (pt) => {
    const vx = (pt.x - midX) / eyeSpan;
    const vy = (pt.y - midY) / eyeSpan;
    // Rotate into eye-aligned frame
    return {
      x: vx * cosA + vy * sinA,
      y: -vx * sinA + vy * cosA,
    };
  };

  const pLeIn = project(leInner);
  const pReIn = project(reInner);
  const pNoseB = project(noseBridge);
  const pNoseT = project(noseTip);
  const pSub = project(subnasal);
  const pLc = project(leftCheek);
  const pRc = project(rightCheek);
  const pChin = project(chin);
  const pFore = project(forehead);
  const pLn = project(leftNose);
  const pRn = project(rightNose);

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const faceWidth = Math.max(0.5, dist(pLc, pRc));
  const faceHeight = Math.max(0.5, dist(pFore, pChin));

  return [
    dist(pLeIn, pReIn), // inner-canthal to outer-eye ratio
    dist(pNoseB, pNoseT), // nose bridge length
    dist(pNoseB, pSub), // midface nose height
    dist(pLn, pRn), // alar nose width
    faceWidth, // bizygomatic cheek width ratio
    faceHeight, // total vertical face ratio
    pChin.y, // lower face projection
    -pFore.y, // forehead height
    pLc.y - pRc.y, // slight asymmetry / cheek offset
    dist(pSub, pChin), // lower third height
    faceWidth / faceHeight, // facial aspect ratio
  ];
}

export function descriptorDistance(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
    return Number.POSITIVE_INFINITY;
  }
  let sq = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sq += d * d;
  }
  return Math.sqrt(sq / a.length);
}

export function matchVisitorProfile(profiles, descriptor, maxDist = 0.22) {
  if (!descriptor || !Array.isArray(profiles) || profiles.length === 0) return null;
  let best = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const profile of profiles) {
    if (!profile?.descriptor) continue;
    const d = descriptorDistance(descriptor, profile.descriptor);
    if (d < bestDist) {
      bestDist = d;
      best = profile;
    }
  }
  if (best && bestDist <= maxDist) {
    return { profile: best, distance: bestDist };
  }
  return null;
}

export function updateRunningDescriptor(existing, sample, alpha = 0.2) {
  if (!Array.isArray(sample)) return existing;
  if (!Array.isArray(existing) || existing.length !== sample.length) return [...sample];
  return existing.map((v, i) => v * (1 - alpha) + sample[i] * alpha);
}

/**
 * Classifies sustained facial expressions from MediaPipe FaceLandmarker blendshapes
 * covering alegria/felicidade (joy), sorriso (smile), tristeza (sadness), and surpresa (surprise).
 */
export function classifyFaceExpression(blendshapes = []) {
  const scoreMap = new Map();
  for (const cat of blendshapes) {
    if (cat?.categoryName) scoreMap.set(cat.categoryName, Number(cat.score || 0));
  }
  const get = (name) => scoreMap.get(name) || 0;

  const smileAvg = (get('mouthSmileLeft') + get('mouthSmileRight')) * 0.5;
  const cheekSq = (get('cheekSquintLeft') + get('cheekSquintRight')) * 0.5;
  const eyeSq = (get('eyeSquintLeft') + get('eyeSquintRight')) * 0.5;
  const frownAvg = (get('mouthFrownLeft') + get('mouthFrownRight')) * 0.5;
  const browInnerUp = get('browInnerUp');
  const browOuterUp = (get('browOuterUpLeft') + get('browOuterUpRight')) * 0.5;
  const eyeWide = (get('eyeWideLeft') + get('eyeWideRight')) * 0.5;
  const jawOpen = get('jawOpen');

  const joyScore = smileAvg * 0.6 + ((cheekSq + eyeSq) * 0.5) * 0.4;
  const sadnessScore = frownAvg * 0.65 + browInnerUp * 0.35;
  const surpriseScore = Math.max(browOuterUp, eyeWide) * 0.55 + jawOpen * 0.45;

  let expression = '';
  if (smileAvg >= 0.7 && (cheekSq + eyeSq) * 0.5 >= 0.24 && frownAvg < 0.25) {
    expression = 'joy';
  } else if (frownAvg >= 0.38 && browInnerUp >= 0.22 && smileAvg < 0.18) {
    expression = 'sadness';
  } else if ((browOuterUp >= 0.48 || eyeWide >= 0.44) && jawOpen >= 0.3 && smileAvg < 0.35 && frownAvg < 0.35) {
    expression = 'surprise';
  } else if (smileAvg >= 0.64 && frownAvg < 0.25) {
    expression = 'smile';
  }

  return {
    expression,
    smileAvg,
    joyScore,
    sadnessScore,
    surpriseScore,
  };
}
