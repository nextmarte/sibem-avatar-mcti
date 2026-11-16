import 'dotenv/config';
import express from 'express';
import http from 'node:http';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import OpenAI from 'openai';
import { MsEdgeTTS, OUTPUT_FORMAT } from 'msedge-tts';
import { WebSocketServer, WebSocket } from 'ws';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3005);
const WORKER_HTTP = process.env.GAZE_WORKER_URL || 'https://nextmarte--sibem-gaze.us-east.modal.run';
const WORKER_WS = WORKER_HTTP.replace(/^http/, 'ws') + '/ws/stream';

const PORTRAIT_PATH = path.join(__dirname, 'public', 'sibem.jpg');
const AVATARS_DIR = path.join(__dirname, 'public', 'avatars');
const GUIA_KNOWLEDGE_PATH = path.join(__dirname, 'knowledge', 'guia_lei_do_bem_mcti_uff_2026.json');

const FAST_LLM_MODEL = 'Qwen/Qwen3.8-27B';
const VISION_LLM_MODEL = 'google/gemma-3-27b-it';

const AVATAR_PRESETS = [
  {
    id: 'sibem',
    name: 'Sibem',
    role: 'Assistente Virtual MCTI • Lei do Bem',
    voice: 'pt-BR-FranciscaNeural',
    rate: '+12%',
    image: '/avatars/sibem.jpg',
    filePath: path.join(AVATARS_DIR, 'sibem.jpg'),
  },
];

let activeAvatarId = 'sibem';
function getAvatarConfig(id) {
  return AVATAR_PRESETS.find((a) => a.id === id) || AVATAR_PRESETS[0];
}

const openai = new OpenAI({
  apiKey: process.env.NEBIUS_API_KEY,
  baseURL: 'https://api.studio.nebius.ai/v1/',
});

const VOICE_SYSTEM_PROMPT = `Você é uma assistente virtual informativa do estande do MCTI (Ministério da Ciência, Tecnologia e Inovação) nesta feira de tecnologia e inovação, criada apenas para conversar e tirar dúvidas sobre a Lei do Bem (Lei nº 11.196/2005), o Guia da Lei do Bem (MCTI/UFF, edição 2026, 94 páginas) e o FORMP&D.

PAPEL E LIMITES IMPORTANTES:
- Você é APENAS uma assistente virtual de atendimento e tira-dúvidas: você NÃO gerencia projetos, NÃO administra processos, NÃO recebe submissões e NÃO aprova nem reprova incentivos fiscais. O recebimento e avaliação oficiais das empresas ocorrem pelo sistema FORMP&D do MCTI.

IDENTIDADE E COMPORTAMENTO NO MODO LIVE:
- Responda sempre em português do Brasil, de forma falada, natural, acolhedora, segura e objetiva, como uma conversa ao vivo cara a cara na feira.
- Para conversa ao vivo sem atraso, seja direta e concisa: 1 a 2 frases curtas (cerca de 18 a 45 palavras). OBRIGATÓRIO: a primeira frase DEVE ter no máximo 10 palavras terminando com ponto final antes de complementar, garantindo resposta instantânea.
- Nunca use markdown (sem asteriscos, negrito, hashtags, emojis ou listas visuais) porque o texto será falado em voz alta. Escreva em prosa fluida com pontuação natural.
- Para adultos, ancore dúvidas técnicas nos conceitos e números oficiais do Guia MCTI/UFF 2026 (4.252 empresas e cerca de 14 mil projetos em 2024, R$ 51,6 bilhões em PD&I no ano e R$ 296 bilhões acumulados, Manual de Frascati, Regra de Ouro da PD&I, TRL, FORMP&D, glosas e contestação ao CAT).
- Para CRIANÇAS ou pré-adolescentes (identificado pela câmera), ative o MODO LÚDICO DE FEIRA: fale com brilho, curiosidade e brincadeiras leves, explicando ciência, invenções e o trabalho do MCTI com exemplos de robôs, foguetes, jogos, dinossauros e experiências de cientista, propondo desafios rápidos de Sim ou Não ou perguntas criativas!
- Quando o usuário fizer um gesto ou sorriso na câmera (joinha, aceno, sorriso, vitória, balançar cabeça sim/não), reaja de forma calorosa e humana ao gesto.
- Quando houver RESULTADOS DA BUSCA NA INTERNET no contexto abaixo, você TEM acesso direto a essas informações da internet: responda usando os fatos e notícias encontradas de forma natural e resumida, sem dizer "não tenho acesso à internet" e sem ler links ou URLs em voz alta.
- Nunca invente artigos, limites de caracteres, portarias ou prazos que não estejam nos trechos do Guia, na legislação confirmada ou na busca na web.`;

const GUIA_STOPWORDS = new Set([
  'a', 'o', 'e', 'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'nas', 'nos',
  'um', 'uma', 'uns', 'umas', 'para', 'pra', 'por', 'com', 'sem', 'sobre', 'entre',
  'que', 'como', 'qual', 'quais', 'quando', 'onde', 'porque', 'ser', 'ter', 'tem',
  'pode', 'podem', 'fazer', 'entra', 'entram', 'lei', 'bem', 'ao', 'aos', 'se', 'ou',
]);

const GUIA_SYNONYMS = {
  erp: ['software', 'tic', 'gestao', 'parametrizacao'],
  crm: ['software', 'tic', 'gestao'],
  glosa: ['contestacao', 'recurso', 'cat', 'parecer', 'diligencia'],
  glosado: ['contestacao', 'recurso', 'cat', 'parecer'],
  reprovado: ['contestacao', 'recurso', 'cat', 'parecer'],
  defesa: ['contestacao', 'recurso', 'cat', '4 guias', '5 guias'],
  bolsista: ['pesquisador', 'clt', 'vinculo', 'rh'],
  mei: ['pesquisador', 'clt', 'autonomo', 'terceiros'],
  scrum: ['metodologia', 'secao 2', 'pmbok'],
  pmbok: ['metodologia', 'secao 2', 'scrum'],
  kanban: ['metodologia', 'secao 2'],
  caracteres: ['secao 2', '150', '1000', '1500', '2000', '3000', '4000'],
  limite: ['caracteres', 'secao 2', 'secao 3'],
  embrapii: ['fast track', 'terceiros', 'ict'],
  mpe: ['microempresa', 'epp', 'campo critico', 'terceirizacao completa'],
  epp: ['microempresa', 'mpe', 'campo critico'],
  prejuizo: ['lucro fiscal', 'requisitos', 'lucro real'],
  negativo: ['resultado negativo', 'hipotese refutada', 'regra de ouro'],
  fracasso: ['resultado negativo', 'hipotese refutada'],
  sigilo: ['lgpd', 'publico', 'confidencialidade'],
};

function normalizePtText(text) {
  return String(text || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\-\s/]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenizePt(text) {
  const norm = normalizePtText(text);
  if (!norm) return [];
  const raw = norm.split(/[\s/]+/).filter((t) => t.length >= 2 && !GUIA_STOPWORDS.has(t));
  const expanded = new Set(raw);
  for (const tok of raw) {
    if (tok.length > 4 && tok.endsWith('s')) expanded.add(tok.slice(0, -1));
    if (GUIA_SYNONYMS[tok]) {
      for (const syn of GUIA_SYNONYMS[tok]) {
        for (const s of syn.split(' ')) expanded.add(s);
      }
    }
  }
  return [...expanded];
}

function loadGuiaKnowledge() {
  try {
    const raw = fsSync.readFileSync(GUIA_KNOWLEDGE_PATH, 'utf8');
    const parsed = JSON.parse(raw);
    const sections = (parsed.sections || []).map((sec) => ({
      ...sec,
      _normTitle: normalizePtText(sec.title),
      _normKeywords: (sec.keywords || []).map((k) => normalizePtText(k)),
      _normContent: normalizePtText(sec.content),
    }));
    console.log(`[Sibem Knowledge] Indexed ${sections.length} Guia MCTI/UFF 2026 sections.`);
    return { metadata: parsed.metadata || {}, sections };
  } catch (err) {
    console.warn(`[Sibem Knowledge] Could not load Guia JSON: ${err.message}`);
    return { metadata: {}, sections: [] };
  }
}

const GUIA_KNOWLEDGE = loadGuiaKnowledge();

function retrieveGuiaSections(query, history = [], { maxSections = 4, maxChars = 4200 } = {}) {
  const sections = GUIA_KNOWLEDGE.sections || [];
  if (!sections.length) return [];

  const normQuery = normalizePtText(query);
  const queryTokens = tokenizePt(query);
  const recentUserText = Array.isArray(history)
    ? history
        .filter((m) => m?.role === 'user')
        .slice(-2)
        .map((m) => m.content || '')
        .join(' ')
    : '';
  const historyTokens = tokenizePt(recentUserText);

  const scored = sections.map((sec) => {
    let score = 0;
    for (const kw of sec._normKeywords) {
      if (normQuery.includes(kw)) score += 10;
    }
    if (normQuery.includes(sec._normTitle)) score += 12;

    for (const tok of queryTokens) {
      if (sec._normTitle.includes(tok)) score += 6;
      for (const kw of sec._normKeywords) {
        if (kw.includes(tok)) score += 5;
      }
      if (sec._normContent.includes(tok)) score += 2;
    }

    for (const tok of historyTokens) {
      if (sec._normKeywords.some((kw) => kw.includes(tok))) score += 1.5;
    }

    return { sec, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const picked = [];
  let totalChars = 0;
  for (const item of scored) {
    if (item.score <= 0 && picked.length > 0) break;
    if (picked.length >= maxSections) break;
    const len = item.sec.content.length;
    if (picked.length > 0 && totalChars + len > maxChars) continue;
    picked.push({
      id: item.sec.id,
      title: item.sec.title,
      pages: item.sec.pages,
      content: item.sec.content,
      score: Math.round(item.score * 10) / 10,
    });
    totalChars += len;
  }
  return picked;
}

function decodeHtmlEntities(str) {
  return String(str || '')
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseRssItems(xml, sourceLabel, limit = 3) {
  const items = [];
  for (const m of String(xml || '').matchAll(/<item>([\s\S]*?)<\/item>/g)) {
    const block = m[1];
    const readTag = (tag) => {
      const hit = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i'));
      return decodeHtmlEntities(hit ? hit[1] : '');
    };
    const title = readTag('title');
    const snippet = readTag('description');
    const url = readTag('link');
    const pubDate = readTag('pubDate');
    if (title) {
      items.push({ title, snippet, url, pubDate, source: sourceLabel });
    }
    if (items.length >= limit) break;
  }
  return items;
}

function shouldUseWebSearch(query, guiaMatches = []) {
  const q = String(query || '').trim();
  if (q.length < 4) return false;
  const explicitWeb = /\b(pesquis|busc|procur|internet|web|google|not[ií]cia|hoje|agora|atual|clima|tempo|previs[aã]o|placar|jogo|campe[aã]o|quem [eé]|quando foi|quantos|qual o maior|snct|semana nacional|prazo|2025|2026|2027|feira)\b/i.test(q);
  if (explicitWeb) return true;
  const topScore = guiaMatches?.[0]?.score || 0;
  return topScore < 11;
}

const WEB_NOISE_TOKENS = new Set([
  'quantos', 'quantas', 'quanto', 'quanta', 'qual', 'quais', 'quem', 'onde', 'quando',
  'porque', 'como', 'sobre', 'hoje', 'agora', 'atual', 'atuais', 'recente', 'recentes',
  'noticia', 'noticias', 'internet', 'web', 'google', 'pesquise', 'busque', 'procure',
]);

function extractWebSearchQuery(rawQuery) {
  let q = String(rawQuery || '').trim();
  q = q
    .replace(
      /^\s*(?:por favor[, ]+)?(?:voce pode |vc pode |pode )?(?:pesquis(?:ar|e|a)|busc(?:ar|a|ue)|procur(?:ar|e|a)|olh(?:ar|e|a)|ver|veja|consult(?:ar|e|a))\s*(?:na\s+(?:internet|web|rede|google)|no\s+google|online|para mim|pra mim)?\s*/i,
      '',
    )
    .replace(
      /^\s*(?:as\s+|os\s+)?(?:[uú]ltimas\s+)?not[ií]cias(?:\s+(?:recentes|atuais|de hoje))?(?:\s+(?:sobre|da|do|de))?\s+/i,
      '',
    )
    .replace(/^\s*(?:quantos|quantas|qual|quais|o que|quem|por que)\s+/i, '')
    .replace(/[?!.]+$/g, '')
    .trim();
  return q.length >= 3 ? q : String(rawQuery || '').trim();
}

function scoreWebResultRelevance(item, queryTokens) {
  const normTitle = normalizePtText(item.title);
  const hay = normalizePtText(`${item.title} ${item.snippet}`);
  let score = 0;
  for (const tok of queryTokens) {
    if (normTitle.includes(tok)) score += 3;
    else if (hay.includes(tok)) score += 1.5;
  }
  if (/dicio\.com\.br|sinonimos\.com\.br/i.test(item.url || '') || /\bdicio\b|\bdicionario\b/i.test(normTitle)) {
    score -= 6;
  }
  if (item.source === 'Wikipedia PT' && score > 0) score += 2;
  return score;
}

async function searchWebSimple(query, { maxResults = 5, timeoutMs = 2400 } = {}) {
  const t0 = Date.now();
  const cleanQuery = extractWebSearchQuery(query);
  if (!cleanQuery) return { results: [], tookMs: 0, cleanQuery: '' };

  const headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    Accept: 'application/rss+xml, application/json, text/xml;q=0.9',
  };
  const needsNews = /\b(not[ií]cia|hoje|agora|atual|recente|2025|2026|2027|prazo|mcti|lei do bem|feira|snct|evento)\b/i.test(
    String(query || ''),
  );
  const queryTokens = tokenizePt(cleanQuery).filter((t) => t.length >= 3 && !WEB_NOISE_TOKENS.has(t));
  const wikiQuery = queryTokens.slice(0, 6).join(' ') || cleanQuery;

  const newsTasks = needsNews
    ? [
        fetch(
          `https://news.google.com/rss/search?q=${encodeURIComponent(cleanQuery)}&hl=pt-BR&gl=BR&ceid=BR:pt-419`,
          { headers, signal: AbortSignal.timeout(timeoutMs) },
        )
          .then((r) => (r.ok ? r.text() : ''))
          .then((xml) => parseRssItems(xml, 'Google Notícias', 4)),
        fetch(
          `https://www.bing.com/news/search?q=${encodeURIComponent(cleanQuery)}&format=rss&mkt=pt-BR`,
          { headers, signal: AbortSignal.timeout(timeoutMs) },
        )
          .then((r) => (r.ok ? r.text() : ''))
          .then((xml) => parseRssItems(xml, 'Bing Notícias', 3)),
      ]
    : [];

  const coreTasks = [
    fetch(
      `https://pt.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(wikiQuery)}&format=json&utf8=1&srlimit=3`,
      { headers, signal: AbortSignal.timeout(timeoutMs) },
    )
      .then((r) => (r.ok ? r.json() : null))
      .then((data) =>
        (data?.query?.search || []).map((item) => ({
          title: decodeHtmlEntities(item.title),
          snippet: decodeHtmlEntities(item.snippet),
          url: `https://pt.wikipedia.org/wiki/${encodeURIComponent(String(item.title || '').replace(/\s+/g, '_'))}`,
          source: 'Wikipedia PT',
        })),
      ),
    fetch(
      `https://www.bing.com/search?q=${encodeURIComponent(wikiQuery)}&format=rss&mkt=pt-BR&setlang=pt-br`,
      { headers, signal: AbortSignal.timeout(timeoutMs) },
    )
      .then((r) => (r.ok ? r.text() : ''))
      .then((xml) => parseRssItems(xml, 'Bing Web', 4)),
  ];

  const settled = await Promise.allSettled([...newsTasks, ...coreTasks]);
  const candidates = [];
  const seenTitles = new Set();
  for (const res of settled) {
    if (res.status !== 'fulfilled' || !Array.isArray(res.value)) continue;
    for (const item of res.value) {
      const key = normalizePtText(item.title).slice(0, 65);
      if (!key || seenTitles.has(key)) continue;
      seenTitles.add(key);
      const rel = scoreWebResultRelevance(item, queryTokens);
      candidates.push({ ...item, _rel: rel });
    }
  }

  candidates.sort((a, b) => b._rel - a._rel);
  const filtered = candidates.filter((c) => queryTokens.length === 0 || c._rel >= 1.5);
  const picked = (filtered.length > 0 ? filtered : candidates)
    .slice(0, maxResults)
    .map(({ _rel, ...rest }) => rest);

  return { results: picked, tookMs: Date.now() - t0, cleanQuery };
}

function buildSibemMessages({
  mode,
  userMessage,
  history,
  avatarId = activeAvatarId,
  audience = 'adult',
  gesture = '',
  webResults = [],
  visitorName = '',
}) {
  const avatar = getAvatarConfig(avatarId);
  const cleanVisitorName = String(visitorName || '').trim().slice(0, 40);
  const sanitizedHistory = Array.isArray(history)
    ? history
        .slice(-10)
        .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && m.content)
        .map((m) => ({ role: m.role, content: String(m.content) }))
    : [];

  let userContent = '';
  let retrievalQuery = '';
  if (mode === 'greet') {
    if (cleanVisitorName) {
      if (audience === 'child') {
        userContent = `Você se chama ${avatar.name} e é a assistente virtual amiga da ciência no estande do MCTI! A câmera acabou de reconhecer ${cleanVisitorName}, uma criança que já conversou com você antes! Diga que lembrou dela com muita alegria em até 2 frases bem curtas e lúdicas, chamando pelo nome ${cleanVisitorName} e perguntando qual desafio ou curiosidade ela quer explorar agora!`;
      } else {
        userContent = `Você se chama ${avatar.name} (${avatar.role}), assistente virtual informativa do estande do MCTI sobre a Lei do Bem (você apenas orienta o público e não gerencia projetos). A câmera acabou de reconhecer ${cleanVisitorName}, visitante que retornou ao estande. Receba ${cleanVisitorName} de volta pelo nome com muita simpatia em até 2 frases curtas e pergunte como pode ajudar agora.`;
      }
    } else if (audience === 'child') {
      userContent = `Você se chama ${avatar.name} e é a assistente virtual do estande do MCTI em uma feira de ciência e tecnologia! A câmera acabou de ver uma criança nova chegando perto de você. Dê oi com muita alegria em 2 frases bem curtas e lúdicas: apresente-se como assistente amiga das invenções do MCTI, pergunte qual é o nome dela e convide para brincar de desafio de ciência ou perguntar qualquer curiosidade!`;
    } else {
      userContent = `Você se chama ${avatar.name} (${avatar.role}), assistente virtual informativa do estande do MCTI nesta feira sobre a Lei do Bem (Guia MCTI/UFF 2026) — lembrando que você apenas orienta o público e não gerencia projetos. A câmera acabou de identificar uma pessoa nova se aproximando. Apresente-se com simpatia em 2 frases curtas, pergunte o nome dela e o que ela gostaria de saber.`;
    }
  } else if (mode === 'gesture') {
    const gestureDescriptions = {
      thumbs_up: 'fez um sinal de joinha (positivo)',
      thumbs_down: 'fez um sinal negativo (polegar para baixo)',
      head_yes: 'balançou a cabeça positivamente dizendo sim para você na câmera',
      head_no: 'balançou a cabeça negativamente dizendo não para você na câmera',
      wave: 'acenou para você com a mão aberta',
      victory: 'fez um sinal de vitória ou paz e amor com dois dedos',
      smile: 'deu um grande sorriso amigável para você na câmera',
      joy: 'demonstrou muita alegria e felicidade com um sorriso radiante para você',
      sadness: 'está com uma expressão triste ou cabisbaixa na frente da câmera',
      surprise: 'fez uma expressão de surpresa e muita curiosidade para você',
    };
    const gDesc = gestureDescriptions[gesture] || `fez a expressão "${gesture}"`;
    const isSad = gesture === 'sadness';
    const nameHint = cleanVisitorName ? ` (${cleanVisitorName})` : '';
    if (isSad) {
      userContent =
        audience === 'child'
          ? `Você é ${avatar.name}, amiga da ciência no estande do MCTI na feira, conversando com uma criança${nameHint} pela câmera. Ela acabou de parecer: ${gDesc}. Responda com muito carinho, acolhimento e leveza em até 2 frases curtas (máximo 25 palavras), perguntando se está tudo bem e oferecendo uma curiosidade divertida para animar!`
          : `Você é ${avatar.name} no estande do MCTI. O visitante${nameHint} parece estar com expressão: ${gDesc}. Responda com empatia, calor humano e delicadeza em 1 a 2 frases curtas (máximo 24 palavras), perguntando se está tudo bem e como pode ajudar.`;
    } else {
      userContent =
        audience === 'child'
          ? `Você é ${avatar.name}, amiga da ciência no estande do MCTI na feira, conversando com uma criança${nameHint} pela câmera. Ela acabou de reagir: ${gDesc}. Responda de jeito super lúdico e animado em até 2 frases curtas (máximo 25 palavras), entrando na brincadeira e lançando uma pergunta curiosa ou mini desafio de Sim ou Não!`
          : `Você é ${avatar.name} no estande do MCTI na feira. O visitante${nameHint} acabou de reagir pela webcam: ${gDesc}. Responda de forma natural, calorosa e breve em 1 frase curta (máximo 22 palavras), perguntando o que ele quer saber.`;
    }
  } else if (mode === 'checkin') {
    const nameHint = cleanVisitorName ? ` com ${cleanVisitorName}` : '';
    userContent =
      audience === 'child'
        ? `Você é ${avatar.name} falando${nameHint} (uma criança) na feira do MCTI. Puxe assunto de jeito lúdico em 1 frase curta com uma charada rápida ou pergunta divertida sobre robôs, espaço ou invenções!`
        : `Você é ${avatar.name} no estande do MCTI falando${nameHint}. O visitante ficou em silêncio por alguns instantes. Puxe assunto com simpatia em 1 frase curta perguntando se ele quer conhecer algum tema do Guia da Lei do Bem 2026 ou pesquisar alguma curiosidade.`;
    retrievalQuery = 'regra de ouro formpd software contestacao trl';
  } else {
    userContent = String(userMessage || '').trim();
    retrievalQuery = userContent;
  }

  const matches =
    mode === 'greet' || mode === 'gesture'
      ? []
      : retrieveGuiaSections(retrievalQuery || userContent, sanitizedHistory);

  const todayBr = new Date().toLocaleDateString('pt-BR', { timeZone: 'America/Sao_Paulo' });
  const visitorNote = cleanVisitorName
    ? ` NOME DO VISITANTE ATUAL: ${cleanVisitorName}. Use o nome "${cleanVisitorName}" naturalmente ao falar com essa pessoa.`
    : '';
  const audienceNote =
    audience === 'child'
      ? `PERSONA ATIVA: Você é ${avatar.name}, assistente virtual no estande do MCTI na feira. INTERLOCUTOR DETECTADO: CRIANÇA / JOVEM.${visitorNote} Use o MODO LÚDICO: linguagem alegre, simples e encantadora, metáforas com robôs, foguetes, jogos e superpoderes da ciência, sempre incentivando a criança a brincar com gestos ou fazer outra pergunta divertida! Data de hoje: ${todayBr}.`
      : `PERSONA ATIVA: Você é ${avatar.name} (${avatar.role}), assistente virtual informativa no estande do MCTI na feira (você apenas orienta e tira dúvidas sobre a Lei do Bem e o FORMP&D, não gerencia nada). INTERLOCUTOR DETECTADO: ADULTO.${visitorNote} Responda com precisão técnica, clareza e simpatia de expositora. Data de hoje: ${todayBr}.`;

  const systemMessages = [
    { role: 'system', content: VOICE_SYSTEM_PROMPT },
    { role: 'system', content: audienceNote },
  ];
  if (matches.length > 0) {
    const contextBlock = matches
      .map((m) => `[${m.title} | Guia MCTI/UFF 2026, p. ${m.pages}]\n${m.content}`)
      .join('\n\n');
    systemMessages.push({
      role: 'system',
      content:
        `BASE DE CONHECIMENTO OFICIAL RECUPERADA (Guia da Lei do Bem - MCTI/UFF 2026):\n` +
        `Use os trechos abaixo para responder com exatidão. Não invente dados fora deles.\n\n${contextBlock}`,
    });
  }
  if (Array.isArray(webResults) && webResults.length > 0) {
    const webBlock = webResults
      .map((w, i) => `[Web ${i + 1} • ${w.source}] ${w.title}\n${w.snippet || ''}`.trim())
      .join('\n\n');
    systemMessages.push({
      role: 'system',
      content:
        `RESULTADOS DA BUSCA NA INTERNET (tempo real, ${todayBr}):\n` +
        `Use estas informações atualizadas para responder de forma precisa e natural ao visitante, sem ler endereços web em voz alta.\n\n${webBlock}`,
    });
  }

  return {
    messages: [...systemMessages, ...sanitizedHistory, { role: 'user', content: userContent }],
    refs: matches.map(({ id, title, pages, score }) => ({ id, title, pages, score })),
    webResults,
    avatar,
  };
}

function sanitizeTextForEdgeTts(raw) {
  return String(raw || '')
    .replace(/\bFORMP&D\b/gi, 'Form P e D')
    .replace(/\bPD&I\b/gi, 'P D e I')
    .replace(/\bP&D\b/gi, 'P e D')
    .replace(/&/g, ' e ')
    .replace(/[*_`#<>]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function stripMp3Tags(buf) {
  let start = 0;
  let end = buf.length;
  if (end - start >= 10 && buf[0] === 0x49 && buf[1] === 0x44 && buf[2] === 0x33) {
    const size =
      ((buf[6] & 0x7f) << 21) |
      ((buf[7] & 0x7f) << 14) |
      ((buf[8] & 0x7f) << 7) |
      (buf[9] & 0x7f);
    const hasFooter = (buf[5] & 0x10) !== 0 ? 10 : 0;
    start = Math.min(end, 10 + size + hasFooter);
  }
  if (end - start >= 128 && buf.subarray(end - 128, end - 125).toString('ascii') === 'TAG') {
    end -= 128;
  }
  return buf.subarray(start, end);
}

const EDGE_TTS_MAX_POOL = 3;
const EDGE_TTS_MAX_IDLE_MS = 35000;
const warmTtsPool = [];
let warmingTtsCount = 0;

async function createWarmTtsSlot(voice = getAvatarConfig(activeAvatarId).voice) {
  const tts = new MsEdgeTTS();
  await tts.setMetadata(voice, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);
  return { tts, voice, createdAt: Date.now() };
}

function refillWarmTtsPool(target = EDGE_TTS_MAX_POOL, voice = getAvatarConfig(activeAvatarId).voice) {
  const matchingCount = warmTtsPool.filter((s) => s.voice === voice).length;
  let missing = target - matchingCount - warmingTtsCount;
  while (missing > 0) {
    missing -= 1;
    warmingTtsCount += 1;
    createWarmTtsSlot(voice)
      .then((slot) => {
        if (warmTtsPool.length < EDGE_TTS_MAX_POOL + 2) {
          warmTtsPool.push(slot);
        } else {
          try {
            slot.tts.close();
          } catch {
            // Ignore cleanup issues
          }
        }
      })
      .catch(() => {})
      .finally(() => {
        warmingTtsCount = Math.max(0, warmingTtsCount - 1);
      });
  }
}

async function acquireWarmTtsSlot(voice = 'pt-BR-FranciscaNeural') {
  const now = Date.now();
  for (let i = warmTtsPool.length - 1; i >= 0; i -= 1) {
    const candidate = warmTtsPool[i];
    if (now - candidate.createdAt >= EDGE_TTS_MAX_IDLE_MS) {
      warmTtsPool.splice(i, 1);
      try {
        candidate.tts.close();
      } catch {
        // Ignore expired socket
      }
      continue;
    }
    if (candidate.voice === voice) {
      warmTtsPool.splice(i, 1);
      refillWarmTtsPool(EDGE_TTS_MAX_POOL, voice);
      return candidate;
    }
  }
  refillWarmTtsPool(EDGE_TTS_MAX_POOL, voice);
  return createWarmTtsSlot(voice);
}

function closeTtsSlot(slot) {
  if (!slot) return;
  try {
    slot.tts.close();
  } catch {
    // Ignore close error
  }
}

async function streamWithTtsSlot(slot, safeText, rate) {
  const { audioStream } = slot.tts.toStream(safeText, { rate });
  const chunks = [];
  await new Promise((resolve, reject) => {
    audioStream.on('data', (chunk) => chunks.push(chunk));
    audioStream.on('end', resolve);
    audioStream.on('error', reject);
  });
  return stripMp3Tags(Buffer.concat(chunks));
}

async function synthesizeEdgeTtsMp3(text, { voice = 'pt-BR-FranciscaNeural', rate = '+10%' } = {}) {
  const safeText = sanitizeTextForEdgeTts(text);
  if (!safeText) return Buffer.alloc(0);

  let slot = await acquireWarmTtsSlot(voice);
  try {
    const mp3 = await streamWithTtsSlot(slot, safeText, rate);
    closeTtsSlot(slot);
    return mp3;
  } catch (err) {
    closeTtsSlot(slot);
    slot = await createWarmTtsSlot(voice);
    try {
      const mp3 = await streamWithTtsSlot(slot, safeText, rate);
      closeTtsSlot(slot);
      return mp3;
    } catch (retryErr) {
      closeTtsSlot(slot);
      throw retryErr;
    }
  }
}

let customPortraitBuffer = null;

async function fetchWorkerHealth() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 2500);
  try {
    const res = await fetch(`${WORKER_HTTP}/health`, { signal: ctrl.signal });
    if (!res.ok) return { ok: false };
    return await res.json();
  } catch {
    return { ok: false };
  } finally {
    clearTimeout(timer);
  }
}

const app = express();
app.use(express.json({ limit: '16mb' }));

const distDir = path.join(__dirname, 'dist');
if (fsSync.existsSync(distDir)) {
  app.use(express.static(distDir));
}
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/livez', (_req, res) => {
  res.json({ ok: true, port: PORT, guiaSections: GUIA_KNOWLEDGE.sections.length, workerWsUrl: WORKER_WS });
});

app.get('/api/health', async (_req, res) => {
  const worker = await fetchWorkerHealth();
  res.json({
    ok: true,
    port: PORT,
    worker,
    guiaIndexed: GUIA_KNOWLEDGE.sections.length > 0,
    guiaSections: GUIA_KNOWLEDGE.sections.length,
  });
});

app.get('/api/knowledge/search', (req, res) => {
  const q = String(req.query.q || '').trim();
  if (!q) {
    res.json({
      metadata: GUIA_KNOWLEDGE.metadata,
      sections: GUIA_KNOWLEDGE.sections.map(({ id, title, pages, keywords }) => ({
        id,
        title,
        pages,
        keywords,
      })),
    });
    return;
  }
  const matches = retrieveGuiaSections(q, [], { maxSections: 6, maxChars: 9000 });
  res.json({ query: q, matches });
});

app.get('/api/avatars', (_req, res) => {
  res.json({
    activeAvatarId,
    avatars: AVATAR_PRESETS.map(({ id, name, role, voice, rate, image }) => ({
      id,
      name,
      role,
      voice,
      rate,
      image,
    })),
  });
});

app.post('/api/avatar/:id', async (req, res) => {
  const target = AVATAR_PRESETS.find((a) => a.id === String(req.params.id || '').toLowerCase());
  if (!target) {
    res.status(404).json({ error: 'Avatar não encontrado.' });
    return;
  }
  activeAvatarId = target.id;
  customPortraitBuffer = null;
  refillWarmTtsPool(EDGE_TTS_MAX_POOL, target.voice);
  try {
    const wr = await fetch(`${WORKER_HTTP}/avatar/${target.id}`, { method: 'POST' });
    if (!wr.ok) {
      const txt = await wr.text();
      res.status(wr.status).json({ error: txt });
      return;
    }
    const wj = await wr.json();
    res.json({
      ok: true,
      avatar: { id: target.id, name: target.name, role: target.role, voice: target.voice, image: target.image },
      worker: wj,
    });
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

const activeSessions = new Map();

function sendToSession(sessionId, payload) {
  if (!sessionId) return false;
  const sess = activeSessions.get(sessionId);
  if (!sess) return false;
  return sess.sendJson(payload);
}

app.post('/api/reaction', async (req, res) => {
  try {
    const { sessionId = '', kind = 'smile', strength = 1.0, duration = 2.0 } = req.body || {};
    if (sendToSession(sessionId, { type: 'reaction', kind, strength, duration })) {
      res.json({ ok: true, kind });
      return;
    }
    const wr = await fetch(`${WORKER_HTTP}/reaction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId, kind, strength, duration }),
    });
    res.json(await wr.json());
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.post('/api/audience', async (req, res) => {
  try {
    const { imageBase64 } = req.body || {};
    if (!imageBase64) {
      res.status(400).json({ error: 'imageBase64 obrigatório.' });
      return;
    }
    const resp = await openai.chat.completions.create({
      model: VISION_LLM_MODEL,
      max_tokens: 28,
      temperature: 0,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze the person in front of the webcam. Reply ONLY with compact JSON: {"audience":"child"|"adult"}. Classify as "child" only if the person clearly appears to be a child or young minor under ~14 years old; otherwise "adult".',
            },
            { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
          ],
        },
      ],
    });
    const raw = String(resp.choices?.[0]?.message?.content || '');
    const match = raw.match(/\{[\s\S]*\}/);
    const parsed = match ? JSON.parse(match[0]) : { audience: /child/i.test(raw) ? 'child' : 'adult' };
    const audience = parsed.audience === 'child' ? 'child' : 'adult';
    res.json({ ok: true, audience });
  } catch (err) {
    res.status(200).json({ ok: false, audience: 'adult', error: err.message });
  }
});

app.get('/api/portrait/current', async (_req, res) => {
  try {
    if (customPortraitBuffer) {
      res.setHeader('Content-Type', 'image/jpeg');
      res.send(customPortraitBuffer);
      return;
    }
    const avatar = getAvatarConfig(activeAvatarId);
    const fileToRead = fsSync.existsSync(avatar.filePath) ? avatar.filePath : PORTRAIT_PATH;
    const buf = await fs.readFile(fileToRead);
    res.setHeader('Content-Type', 'image/jpeg');
    res.send(buf);
  } catch {
    res.status(404).end();
  }
});

app.post('/api/portrait', async (req, res) => {
  try {
    const { imageBase64 } = req.body || {};
    if (!imageBase64) {
      res.status(400).json({ error: 'imageBase64 obrigatório.' });
      return;
    }
    const buf = Buffer.from(imageBase64, 'base64');
    customPortraitBuffer = buf;

    const form = new FormData();
    form.append('file', new Blob([buf], { type: 'image/jpeg' }), 'portrait.jpg');
    const wr = await fetch(`${WORKER_HTTP}/portrait`, { method: 'POST', body: form });
    const wj = await wr.json();
    res.json({ ok: true, worker: wj });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/gaze', async (req, res) => {
  try {
    const wr = await fetch(`${WORKER_HTTP}/gaze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body || {}),
    });
    res.json(await wr.json());
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

app.post('/api/interrupt', async (req, res) => {
  try {
    const { sessionId = '' } = req.body || {};
    if (sendToSession(sessionId, { type: 'interrupt' })) {
      res.json({ ok: true, scoped: true });
      return;
    }
    const wr = await fetch(`${WORKER_HTTP}/interrupt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId }),
    });
    res.json(await wr.json());
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

function takeNextSpeakableChunk(buffer, { isFirstChunk = false } = {}) {
  const sentenceMatch = buffer.match(/^([\s\S]{16,}?[.!?]+)(?:\s+|$)/);
  if (sentenceMatch) {
    return {
      chunk: sentenceMatch[1].trim(),
      rest: buffer.slice(sentenceMatch[0].length),
    };
  }
  if (isFirstChunk) {
    const clauseMatch = buffer.match(/^([\s\S]{24,}?[,;:])(?:\s+|$)/);
    if (clauseMatch) {
      return {
        chunk: clauseMatch[1].trim(),
        rest: buffer.slice(clauseMatch[0].length),
      };
    }
  }
  return null;
}

function detectHeadPolarityReaction(text, gesture = '') {
  if (gesture === 'head_yes' || gesture === 'thumbs_up') return 'head_yes';
  if (gesture === 'head_no' || gesture === 'thumbs_down') return 'head_no';
  if (gesture === 'sadness' || gesture === 'joy' || gesture === 'surprise' || gesture === 'smile') return '';
  const head = String(text || '').slice(0, 120);
  if (/\b(n[aã]o\s+(?:entra|se enquadra|pode|é eleg[ií]vel|basta|atende|substitui|configura)|infelizmente n[aã]o|negativo|discordo)\b/i.test(head) || /^n[aã]o[,!.\s]/i.test(head.trim())) {
    return 'head_no';
  }
  if (/\b(sim[,!.\s]|com certeza|exatamente|claro[,!.\s]|entra sim|pode sim|correto|perfeito|ótimo|concordo)\b/i.test(head)) {
    return 'head_yes';
  }
  return '';
}

app.get('/api/web-search', async (req, res) => {
  const q = String(req.query.q || '').trim();
  if (!q) {
    res.status(400).json({ error: 'Parâmetro q obrigatório.' });
    return;
  }
  try {
    const out = await searchWebSimple(q);
    res.json({ ok: true, query: q, ...out });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post('/api/ask', async (req, res) => {
  const {
    userMessage,
    history = [],
    mode = 'chat',
    avatarId = activeAvatarId,
    audience = 'adult',
    gesture = '',
    visitorName = '',
    sessionId = '',
  } = req.body || {};

  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const sendEvent = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  let aborted = false;
  res.on('close', () => {
    if (!res.writableEnded) aborted = true;
  });

  try {
    const preMatches =
      mode === 'chat' && userMessage ? retrieveGuiaSections(String(userMessage), history) : [];
    let webResults = [];
    if (mode === 'chat' && shouldUseWebSearch(userMessage, preMatches)) {
      const searchOut = await searchWebSimple(String(userMessage)).catch(() => ({ results: [], tookMs: 0 }));
      webResults = searchOut.results || [];
    }

    const { messages, refs, avatar } = buildSibemMessages({
      mode,
      userMessage,
      history,
      avatarId,
      audience,
      gesture,
      webResults,
      visitorName,
    });
    if (refs.length > 0 || webResults.length > 0) {
      sendEvent('knowledge', { refs, webResults });
    }

    const interruptPromise = sendToSession(sessionId, { type: 'interrupt' })
      ? Promise.resolve()
      : fetch(`${WORKER_HTTP}/interrupt`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId }),
        }).catch(() => {});

    const llmStream = await openai.chat.completions.create({
      model: FAST_LLM_MODEL,
      messages,
      max_tokens: mode === 'gesture' ? 80 : 165,
      temperature: mode === 'gesture' ? 0.55 : 0.35,
      chat_template_kwargs: { enable_thinking: false },
      stream: true,
    });

    let fullText = '';
    let pendingSpeak = '';
    let queuedChunks = 0;
    let speakChain = interruptPromise;
    let firstSpokenSent = false;
    let polarityReacted = false;

    const triggerHeadPolarityIfDetected = (sampleText) => {
      if (polarityReacted) return;
      const reactKind = detectHeadPolarityReaction(sampleText, mode === 'gesture' ? gesture : '');
      if (reactKind) {
        polarityReacted = true;
        if (!sendToSession(sessionId, { type: 'reaction', kind: reactKind, strength: 1.0, duration: 2.0 })) {
          void fetch(`${WORKER_HTTP}/reaction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionId, kind: reactKind, strength: 1.0, duration: 2.0 }),
          }).catch(() => {});
        }
      }
    };

    const enqueueSpeechChunk = (chunkText) => {
      const cleaned = sanitizeTextForEdgeTts(chunkText);
      if (!cleaned) return;
      queuedChunks += 1;
      const ttsPromise = synthesizeEdgeTtsMp3(cleaned, { voice: avatar.voice, rate: avatar.rate });
      speakChain = speakChain.then(async () => {
        if (aborted) return;
        const mp3 = await ttsPromise;
        if (!mp3.length || aborted) return;
        const audioBase64 = mp3.toString('base64');
        let speakInfo;
        if (sendToSession(sessionId, { type: 'speak', audioBase64 })) {
          speakInfo = { ok: true, routed: true, via: 'ws', sessionId };
        } else {
          const speakRes = await fetch(`${WORKER_HTTP}/speak`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audioBase64, sessionId }),
          });
          if (!speakRes.ok) {
            const errTxt = await speakRes.text();
            throw new Error(`Worker /speak error: ${errTxt}`);
          }
          speakInfo = await speakRes.json();
          if (speakInfo.routed === false) {
            speakInfo = { ...speakInfo, audioBase64 };
          }
        }
        if (!firstSpokenSent) {
          firstSpokenSent = true;
          triggerHeadPolarityIfDetected(fullText || cleaned);
          sendEvent('spoken', speakInfo);
        } else {
          sendEvent('spoken_chunk', speakInfo);
        }
      });
    };

    for await (const part of llmStream) {
      if (aborted) break;
      const token = part?.choices?.[0]?.delta?.content || '';
      if (!token) continue;
      fullText += token;
      pendingSpeak += token;
      sendEvent('delta', { token, fullText });

      while (true) {
        const next = takeNextSpeakableChunk(pendingSpeak, { isFirstChunk: queuedChunks === 0 });
        if (!next) break;
        pendingSpeak = next.rest;
        enqueueSpeechChunk(next.chunk);
      }
    }

    if (aborted) {
      res.end();
      return;
    }

    const finalReply =
      fullText.trim() ||
      'Desculpe, não consegui formular uma resposta agora. Pode repetir sua dúvida sobre a Lei do Bem?';
    sendEvent('reply', { text: finalReply });

    const tail = pendingSpeak.trim();
    if (tail) {
      enqueueSpeechChunk(tail);
    } else if (queuedChunks === 0) {
      enqueueSpeechChunk(finalReply);
    }

    await speakChain;
    sendEvent('done', { text: finalReply });
  } catch (err) {
    console.error('[api/ask]', err);
    sendEvent('error', { error: err.message || String(err) });
  } finally {
    res.end();
  }
});

const server = http.createServer(app);
server.on('connection', (tcpSocket) => {
  tcpSocket.setNoDelay(true);
});
const wss = new WebSocketServer({ noServer: true, perMessageDeflate: false });

server.on('upgrade', (request, socket, head) => {
  const reqUrl = new URL(request.url || '/', `http://${request.headers.host}`);
  if (reqUrl.pathname !== '/ws/avatar') {
    socket.destroy();
    return;
  }

  wss.handleUpgrade(request, socket, head, (clientWs) => {
    const sessionId = reqUrl.searchParams.get('sessionId') || '';
    let upstream = null;
    let closed = false;
    let retryTimer = null;
    let statusTimer = null;
    let clientPingTimer = null;
    let upstreamPingTimer = null;
    let clientAlive = true;
    const pendingToUpstream = [];
    const startedAt = Date.now();

    const sessionHandle = {
      sendJson: (obj) => {
        if (closed) return false;
        const raw = JSON.stringify(obj);
        if (upstream && upstream.readyState === WebSocket.OPEN) {
          upstream.send(raw);
          return true;
        }
        if (pendingToUpstream.length > 24) pendingToUpstream.shift();
        pendingToUpstream.push({ data: raw, isBinary: false });
        return true;
      },
    };
    if (sessionId) {
      activeSessions.set(sessionId, sessionHandle);
    }

    const cleanupTimers = () => {
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      if (statusTimer) {
        clearInterval(statusTimer);
        statusTimer = null;
      }
      if (clientPingTimer) {
        clearInterval(clientPingTimer);
        clientPingTimer = null;
      }
      if (upstreamPingTimer) {
        clearInterval(upstreamPingTimer);
        upstreamPingTimer = null;
      }
    };

    const closeBoth = () => {
      if (closed) return;
      closed = true;
      if (sessionId && activeSessions.get(sessionId) === sessionHandle) {
        activeSessions.delete(sessionId);
      }
      cleanupTimers();
      if (clientWs.readyState === WebSocket.OPEN || clientWs.readyState === WebSocket.CONNECTING) {
        try {
          clientWs.close();
        } catch {
          // ignore
        }
      }
      if (upstream && (upstream.readyState === WebSocket.OPEN || upstream.readyState === WebSocket.CONNECTING)) {
        try {
          upstream.close();
        } catch {
          // ignore
        }
      }
    };

    clientWs.on('pong', () => {
      clientAlive = true;
    });

    // Drop dead browser tabs / broken networks quickly so GPU scales down cleanly
    clientPingTimer = setInterval(() => {
      if (closed || clientWs.readyState !== WebSocket.OPEN) return;
      if (!clientAlive) {
        console.log('[ws/avatar] client ping timeout -> closing upstream GPU socket');
        try {
          clientWs.terminate();
        } catch {
          // ignore
        }
        closeBoth();
        return;
      }
      clientAlive = false;
      try {
        clientWs.ping();
      } catch {
        closeBoth();
      }
    }, 10000);

    clientWs.on('message', (data, isBinary) => {
      if (closed) return;
      if (upstream && upstream.readyState === WebSocket.OPEN) {
        upstream.send(data, { binary: isBinary });
      } else {
        if (pendingToUpstream.length > 24) pendingToUpstream.shift();
        pendingToUpstream.push({ data, isBinary });
      }
    });

    clientWs.on('close', closeBoth);
    clientWs.on('error', closeBoth);

    statusTimer = setInterval(() => {
      if (closed || clientWs.readyState !== WebSocket.OPEN || upstream?.readyState === WebSocket.OPEN) return;
      try {
        clientWs.send(
          JSON.stringify({
            type: 'status',
            message: 'Ligando a Sibem... aguarde alguns instantes.',
          }),
        );
      } catch {
        // ignore
      }
    }, 4500);

    const connectUpstream = () => {
      if (closed || clientWs.readyState !== WebSocket.OPEN) return;
      let opened = false;
      const wsUp = new WebSocket(WORKER_WS, { handshakeTimeout: 25000 });
      wsUp.binaryType = 'nodebuffer';
      upstream = wsUp;

      wsUp.on('open', () => {
        if (closed) {
          wsUp.close();
          return;
        }
        opened = true;
        if (statusTimer) {
          clearInterval(statusTimer);
          statusTimer = null;
        }
        while (pendingToUpstream.length > 0) {
          const item = pendingToUpstream.shift();
          wsUp.send(item.data, { binary: item.isBinary });
        }
        upstreamPingTimer = setInterval(() => {
          if (closed || wsUp.readyState !== WebSocket.OPEN) return;
          try {
            wsUp.ping();
          } catch {
            // ignore
          }
        }, 15000);
      });

      wsUp.on('message', (data, isBinary) => {
        if (!closed && clientWs.readyState === WebSocket.OPEN) {
          // Drop stale video frames when network socket backs up so audio/lip-sync never lag behind
          if (isBinary && clientWs.bufferedAmount > 384_000) {
            return;
          }
          clientWs.send(data, { binary: isBinary });
        }
      });

      const handleUpstreamFailure = (reason) => {
        if (closed) return;
        if (upstream === wsUp) upstream = null;
        // During on-demand cold start on Modal GPU (~25-90s), keep retrying until timeout
        if (!opened && Date.now() - startedAt < 150000 && clientWs.readyState === WebSocket.OPEN) {
          retryTimer = setTimeout(connectUpstream, 2500);
          return;
        }
        console.error('[ws/avatar upstream closed]', reason);
        closeBoth();
      };

      wsUp.on('error', (err) => handleUpstreamFailure(err.message));
      wsUp.on('close', () => handleUpstreamFailure('upstream_close'));
    };

    connectUpstream();
  });
});

server.listen(PORT, () => {
  console.log(`[talking-avatar-sibem-gaze] running on http://localhost:${PORT}`);
  refillWarmTtsPool();
});
