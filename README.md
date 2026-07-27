# LLM Fine-tune Classification

Классификация тональности отзывов Kinopoisk (**Bad / Good / Neutral**) на базе `HuggingFaceTB/SmolLM2-135M-Instruct`.

Обучение: **full SFT** или **PEFT (LoRA)**. Запуск: локально или в Docker.

---

## 1. Установка зависимостей

### 1.1. Клонирование и `.env`

```bash
cd llm-finetune-classification

# создай .env в корне:
```

```env
HF_TOKEN=hf_xxxxxxxx
BASE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
SFT_OUTPUT=./sft_output
SFT_OUTPUT_PEFT=./sft_output_peft
```

Токен: https://huggingface.co/settings/tokens

### 1.2. Python-окружение

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Mac / CPU:**
```bash
pip install -r requirements.txt
```

**Linux + NVIDIA GPU:**
```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 1.3. В IDE (PyCharm)

1. Open project → папка `llm-finetune-classification`.
2. Interpreter → `.venv`.
3. ПКМ по `src` → **Mark Directory as → Sources Root**  
   (или в Run Configuration: `PYTHONPATH=src`, Working directory = корень проекта).
4. Зависимости: Terminal → команды из п. 1.2.

---

## 2. Обучение модели

Всегда из корня проекта, с `PYTHONPATH=src`.

### Full SFT (все веса)

```bash
# скрипт: если нет папки SFT_OUTPUT — сам обучит, потом predict на test
PYTHONPATH=src python src/main.py
```

Или только обучение через код/API (см. ниже):
```bash
PYTHONPATH=src python -c "from training import run_full_training; print(run_full_training())"
```

Веса: `./sft_output` (`SFT_OUTPUT` в `.env`).

### PEFT / LoRA

```bash
PYTHONPATH=src python -c "from training import run_peft_training; print(run_peft_training())"
```

Веса: `./sft_output_peft` (`SFT_OUTPUT_PEFT` в `.env`).

### Через API (после запуска сервера, п. 4)

```bash
# LoRA
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"mode": "peft", "max_steps": 50}'

# Full
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"mode": "full", "max_steps": 50}'

# статус
curl http://localhost:8000/train/status
```

Параметры JSON (все необязательные): `mode`, `max_steps`, `per_device_train_batch_size`, `learning_rate`, `output_dir`.

**Данные:** train 70% / val 15% / test 15% (`load_splits`, seed=42).  
Val — loss во время train; test — финальная оценка.

---

## 3. Инференс

### 3.1. Один отзыв (API)

Сначала подними API (п. 4), затем:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "movie_name": "Блеф (1976)",
    "review": "Отличный фильм, очень смешно!",
    "model_path": "./sft_output_peft"
  }' | python3 -m json.tool
```

Или в браузере: **http://localhost:8000/docs** → `POST /predict` → Try it out.

Без `model_path` по умолчанию берётся `SFT_OUTPUT_PEFT`.

### 3.2. Оценка на test

```bash
curl -s -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./sft_output_peft",
    "limit": 50
  }' | python3 -m json.tool
```

Ответ: `accuracy`, `f1_macro`, `split: test`.

### 3.3. Full-модель на test (скрипт)

```bash
PYTHONPATH=src python src/main.py
```

Печатает `true` / `pred` по test (после train, если модели не было).

### 3.4. После train через API

```bash
curl -X POST http://localhost:8000/model/reload
# затем снова /predict
```

---

## 4. Docker: поднять образ и работать с ним

### 4.1. Требования

- [Docker Desktop](https://www.docker.com/products/docker-desktop) запущен (`docker info` без ошибок).
- Файл `.env` в корне проекта (п. 1.1).

### 4.2. Сборка и запуск

```bash
cd llm-finetune-classification

# CPU (Mac, сервер без GPU)
docker compose --profile cpu up --build -d

# GPU (RTX 4090 + NVIDIA Container Toolkit)
docker compose --profile gpu up --build -d
```

Первая сборка долгая (скачивается torch и зависимости).

Логи:
```bash
docker compose --profile cpu logs -f
# GPU:
docker compose --profile gpu logs -f
```

### 4.3. Проверка, что контейнер жив

```bash
docker compose ps
curl -s http://localhost:8000/health | python3 -m json.tool
```

- API: http://localhost:8000  
- Swagger: http://localhost:8000/docs  

Ожидаешь в `/health`: `"status": "ok"`.

GPU:
```bash
docker compose --profile gpu exec api-gpu \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.4. Работа с API внутри Docker

Те же запросы, что в п. 2–3, на `http://localhost:8000`:

```bash
# health
curl -s http://localhost:8000/health

# обучение peft
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"mode": "peft", "max_steps": 50}'

curl -s http://localhost:8000/train/status

# инференс
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"movie_name":"Блеф (1976)","review":"Отличный фильм!"}'

# оценка на test
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_path":"./sft_output_peft","limit":20}'
```

Или UI: http://localhost:8000/docs

Веса пишутся на хост в `./sft_output` и `./sft_output_peft` (volumes).

### 4.5. Остановка

```bash
docker compose --profile cpu down
# или
docker compose --profile gpu down
```

### 4.6. Типичные проблемы

| Проблема | Решение |
|----------|---------|
| `Cannot connect to Docker daemon` | запустить Docker Desktop |
| `unable to get image` | daemon не запущен / нет сети |
| порт 8000 занят | `API_PORT=8001 docker compose --profile cpu up -d` |
| CUDA build на Mac | только `--profile cpu` |
| predict падает, нет модели | сначала `/train` или локально `main.py` |

---

## 5. API (справка)

| Метод | Путь | Назначение |
|-------|------|------------|
| GET | `/health` | статус, device |
| POST | `/predict` | один отзыв |
| POST | `/evaluate` | accuracy/F1 на **test** |
| POST | `/train` | peft / full (фон) |
| GET | `/train/status` | idle / running / completed / failed |
| POST | `/model/reload` | перезагрузить веса |

---

## 6. Структура кода

```
src/
  api.py              # FastAPI
  training.py         # run_peft_training, run_full_training
  inference.py        # load_model, predict
  prepare_datasets.py # load_splits() → train, val, test
  formatting_func.py  # промпт
  config.py           # .env
  main.py             # full train (если нет папки) + predict на test
  model_service.py    # кэш модели для API
  get_device.py       # cuda / mps / cpu
```

## 7. Зависимости

См. `requirements.txt`. В Docker torch CPU/CUDA задаётся через `TORCH_INDEX_URL` в `Dockerfile` / профилях compose.
