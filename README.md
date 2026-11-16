# Sibem — Assistente Virtual Interativa da Lei do Bem (MCTI)

Assistente virtual conversacional em tempo real criada para atendimento ao público sobre a **Lei do Bem (Lei nº 11.196/2005)**, o **Guia da Lei do Bem (MCTI/UFF 2026)** e o **FORMP&D**, com animação facial neural sincronizada por áudio, rastreamento de olhar/gestos/expressões pela câmera e execução escalável com GPU sob demanda na **Modal**.

- **Aplicação publicada na Modal:** `https://nextmarte--sibem.us-east.modal.run/`

---

## Arquitetura

- **Front-end (`index.html`, `src/`)**: Interface Vite + MediaPipe Tasks Vision (`public/mediapipe/`) para detecção local de rosto, olhar, sorriso, emoção e gestos (`joinha`, `aceno`, `vitória`, aceno de cabeça), captura de microfone com VAD sem interrupção da fala do usuário e renderizador `<canvas>` para o stream WebP em tempo real.
- **Servidor Web & Orquestração (`server.js`)**: Express + `ws` que serve o bundle `dist/`, consulta a base indexada do Guia (`knowledge/guia_lei_do_bem_mcti_uff_2026.json`), gera respostas via **Nebius Token Factory** (`Qwen/Qwen3.8-27B` e `google/gemma-3-27b-it`), sintetiza voz em PT-BR com `msedge-tts` (`pt-BR-FranciscaNeural`) e faz proxy WebSocket com isolamento por `sessionId`.
- **Worker Neural GPU (`gaze_worker.py` + `ditto_engine/`)**: FastAPI + Uvicorn + PyTorch rodando o pipeline Ditto (`digital-avatar/ditto-talkinghead`) com isolamento de estado por sessão (`SessionState`), redimensionamento direto na GPU e envio em WebP.
- **Deploy Serverless (`modal_gaze.py`)**:
  - `web`: contêiner CPU econômico (até 100 conexões concorrentes por réplica).
  - `serve`: worker GPU **NVIDIA H100** sob demanda (`min_containers=0`, `max_containers=4`, `scaledown_window=35` segundos), ligado apenas enquanto houver conversa ativa.

---

## Estrutura do Projeto

```text
.
├── ditto_engine/           # Motor neural de animação facial e áudio-para-movimento
├── knowledge/              # Base estruturada do Guia da Lei do Bem (MCTI/UFF 2026)
├── public/                 # Retrato da Sibem, fundo e modelos/WASM do MediaPipe
├── src/                    # Código-fonte do front-end (main.js, visitorMemory.js, style.css)
├── gaze_worker.py          # Servidor FastAPI/WebSocket do avatar em GPU
├── modal_gaze.py           # Infraestrutura e deploy serverless na Modal (web + H100 sob demanda)
├── server.js               # Servidor Node.js (API SSE, TTS, RAG, proxy WebSocket multiusuário)
├── index.html              # Página principal
├── package.json            # Dependências e scripts Node.js
├── requirements.txt        # Dependências Python para deploy na Modal / worker GPU local
└── .env.example            # Modelo de variáveis de ambiente
```

---

## Pré-requisitos

- **Node.js 20+** (ou 22+)
- **Conta Nebius Token Factory** (`NEBIUS_API_KEY`)
- **Python 3.11+** com `modal` CLI (necessário apenas se quiser fazer deploy na Modal ou rodar o worker GPU localmente)

---

## Como Executar a partir de Qualquer Lugar

### Opção 1 — Rodar o app localmente usando a GPU H100 sob demanda da Modal (não precisa de GPU local)

1. Clone o repositório e instale as dependências Node:
   ```bash
   git clone https://github.com/nextmarte/sibem-avatar-mcti.git
   cd sibem-avatar-mcti
   npm install
   ```

2. Configure o arquivo `.env` a partir do modelo:
   ```bash
   cp .env.example .env
   ```
   Preencha `NEBIUS_API_KEY` com sua chave e mantenha `GAZE_WORKER_URL=https://nextmarte--sibem-gaze.us-east.modal.run`.

3. Gere o build do front-end e inicie o servidor:
   ```bash
   npm run build
   npm start
   ```
   Acesse `http://localhost:3005` no navegador.

---

### Opção 2 — Subir sua própria instância completa na Modal

1. Instale as dependências e gere o build do front-end:
   ```bash
   npm install
   npm run build
   ```

2. Instale e autentique a CLI da Modal (requer Python 3.11+):
   ```bash
   pip install modal
   modal setup
   ```

3. Cadastre o segredo com a chave da Nebius na Modal:
   ```bash
   modal secret create sibem-nebius NEBIUS_API_KEY=sua_chave_nebius_aqui
   ```

4. Publique a aplicação e o worker H100 sob demanda:
   ```bash
   modal deploy modal_gaze.py
   ```
   *(Na primeira execução em GPU, os checkpoints de `digital-avatar/ditto-talkinghead` são baixados automaticamente para o volume persistente `sibem-gaze-models` na Modal.)*

---

### Opção 3 — Rodar o worker GPU localmente (máquina com GPU NVIDIA + CUDA + ffmpeg)

1. Crie um ambiente Python 3.11 e instale as dependências:
   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Baixe os checkpoints de `digital-avatar/ditto-talkinghead` para `./checkpoints` (`ditto_cfg/*` e `ditto_pytorch/*`) ou defina `CFG_PKL` e `DATA_ROOT`.

3. Suba o worker GPU e o servidor Node:
   ```bash
   python gaze_worker.py
   ```
   Em outro terminal, ajuste no `.env` para `GAZE_WORKER_URL=http://127.0.0.1:8766`, rode `npm run build && npm start` e acesse `http://localhost:3005`.
