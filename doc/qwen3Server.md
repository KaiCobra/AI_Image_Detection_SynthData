# Qwen3 Object Extraction Server

基於 **Qwen3-4B-Instruct-2507** 的 REST API 服務，用於從圖片描述 (caption) 中提取可移除的前景物件。

## 環境資訊

| 項目 | 路徑 |
|------|------|
| Conda 環境 | `/media/avlab/afd90e24-420c-46a0-95e0-c017b48d0db0/conda_envs/qwen3` |
| 模型權重 | `/media/avlab/afd90e24-420c-46a0-95e0-c017b48d0db0/Qwen3Weights` |
| Python 版本 | 3.11 |
| 主要套件 | `torch 2.5.1+cu121`, `transformers>=4.51.0`, `flask`, `accelerate` |

## 啟動 Server

### 方法一：使用啟動腳本（推薦）

```bash
cd ~/Documents/AI_Image_Detection_SynthData
bash qwen3_server/start_server.sh
```

可自訂 port：

```bash
bash qwen3_server/start_server.sh --port 6060
```

### 方法二：手動啟動

```bash
# ⚠️ 如果目前有 .venv 處於啟用狀態，必須先 deactivate
deactivate 2>/dev/null; conda activate /media/avlab/afd90e24-420c-46a0-95e0-c017b48d0db0/conda_envs/qwen3
export TMPDIR=/media/avlab/afd90e24-420c-46a0-95e0-c017b48d0db0/tmp
cd ~/Documents/AI_Image_Detection_SynthData
python qwen3_server/server.py --port 5050 --host 0.0.0.0
```

啟動後模型約需 2–3 秒載入至 GPU，出現 `Running on http://0.0.0.0:5050` 即代表就緒。

> **常見錯誤：`ModuleNotFoundError: No module named 'torch'`**
>
> 這代表 workspace 中的 `.venv` 仍處於啟用狀態（VS Code 會自動啟用），其 PATH 優先級高於 conda。
> 請先執行 `deactivate` 再 `conda activate`，或直接使用方法一的啟動腳本（已自動處理）。

## API 端點

預設位址：`http://localhost:5050`

### `POST /extract` — 單筆提取

**Request:**

```json
{
  "caption": "a bicycle leaning against an old brick wall, in the style of nostalgic rural life"
}
```

**Response:**

```json
{
  "object": "bicycle"
}
```

### `POST /batch_extract` — 批次提取

**Request:**

```json
{
  "captions": [
    "a bicycle leaning against an old brick wall",
    "there is a rider statue above the street, in the style of french cityscape",
    "There is a large building that is in the middle of a field, inside the Roman Colosseum"
  ]
}
```

**Response:**

```json
{
  "objects": ["bicycle", "statue", "</nothing>"]
}
```

### `GET /health` — 健康檢查

**Response:**

```json
{
  "status": "ok",
  "model": "Qwen3-4B-Instruct-2507"
}
```

## 回傳邏輯

- 若 caption 中存在可移除的前景物件（如 bicycle、statue、car、dog），回傳該 **單字**（小寫）。
- 若 caption 僅描述場景/風景/建築，無明確可移除的前景物件，回傳 `</nothing>`。

## 從 sam3 環境呼叫（Client 使用方式）

Client 只依賴 `requests`，可在任何 Python 環境中使用，不需要 GPU 或 transformers。

### 單筆

```python
from qwen3_server.client import extract_object

obj = extract_object("a bicycle leaning against an old brick wall")
print(obj)  # "bicycle"
```

### 批次

```python
from qwen3_server.client import batch_extract_objects

results = batch_extract_objects([
    "a bicycle leaning against an old brick wall",
    "There is a large building in the middle of a field",
])
print(results)  # ["bicycle", "</nothing>"]
```

### 確認 Server 是否在線

```python
from qwen3_server.client import is_server_running

if is_server_running():
    print("Server is up")
```

### 自訂 Server 位址

```python
obj = extract_object("...", server_url="http://192.168.50.203:5050")
```

## 用 curl 測試

```bash
# 健康檢查
curl http://localhost:5050/health

# 單筆提取
curl -X POST http://localhost:5050/extract \
  -H "Content-Type: application/json" \
  -d '{"caption": "a bicycle leaning against an old brick wall"}'

# 批次提取
curl -X POST http://localhost:5050/batch_extract \
  -H "Content-Type: application/json" \
  -d '{"captions": ["a bicycle leaning against an old brick wall", "a sunset over the ocean"]}'
```

## 檔案結構

```
qwen3_server/
├── __init__.py
├── server.py          # Flask server，載入模型並提供 REST API
├── client.py          # 輕量 client，可從 sam3 等其他環境呼叫
└── start_server.sh    # 一鍵啟動腳本（自動 activate conda env）
```

## 注意事項

- 根目錄磁碟空間不足，conda 環境與模型權重皆存放於外接 NVMe (`/media/avlab/afd90e24-420c-46a0-95e0-c017b48d0db0`)。
- Server 使用 `threaded=False`（單 thread），同一時間僅處理一個 request。若需並行請用 batch endpoint。
- GPU 記憶體占用約 8–9 GB (bfloat16)。如果 sam3 的工作也需要 GPU，請注意 VRAM 分配。
