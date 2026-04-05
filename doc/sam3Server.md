# SAM3 Segmentation Server

基於 **SAM3 (Segment Anything Model 3)** 的 REST API 服務，用於以文字 prompt 從圖片中分割物件，回傳帶透明背景的 RGBA PNG 切割圖，可直接用於圖片合成。

## 環境資訊

| 項目 | 路徑 |
|------|------|
| Conda 環境 | `sam3` |
| 模型權重 | `sam3/sam3_hf/sam3.pt`（symlink → `/media/avlab/8TB/sam3_weight/sam3_hf/`） |
| Python 版本 | 3.12 |
| 主要套件 | `torch 2.7.0+cu126`, `sam3 0.1.0`, `pillow`, `numpy` |

## 啟動 Server

### 方法一：使用啟動腳本（推薦）

```bash
cd ~/Documents/AI_Image_Detection_SynthData
bash sam3_server/start_server.sh
```

可自訂 port：

```bash
bash sam3_server/start_server.sh --port 5051
```

### 方法二：手動啟動

```bash
conda activate sam3
cd ~/Documents/AI_Image_Detection_SynthData
python sam3_server/server.py --port 5051 --host 0.0.0.0
```

啟動後模型約需 10–15 秒載入至 GPU，出現以下訊息即代表就緒：

```
INFO:__main__:SAM3 server listening on 0.0.0.0:5051
```

## API 端點

預設位址：`http://localhost:5051`

### `POST /segment` — 物件分割

**Request:**

```json
{
  "prompt": "truck",
  "image_path": "/path/to/photo.jpg",
  "confidence": 0.3
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `prompt` | string | 是 | 要分割的物件名稱，例如 `"truck"`、`"dog"` |
| `image_path` | string | 擇一 | 圖片的本機路徑（同機器時用這個，速度較快） |
| `image` | string | 擇一 | 圖片的 base64 編碼 |
| `confidence` | float | 否 | 信心閾值，預設 `0.3` |

**Response（偵測到物件）:**

```json
{
  "cutout": "<base64 RGBA PNG>",
  "mask": "<base64 grayscale PNG>",
  "bbox": [85, 282, 1711, 850],
  "score": 0.863
}
```

| 欄位 | 類型 | 說明 |
|------|------|------|
| `cutout` | string | RGBA PNG（base64），物件以外區域為透明 |
| `mask` | string | 灰階 PNG（base64），白色=物件，黑色=背景 |
| `bbox` | list | 邊界框 `[x1, y1, x2, y2]`，像素座標 |
| `score` | float | 偵測信心分數 |

**Response（未偵測到物件）:**

```json
{"error": "No object found", "n_objects": 0}
```

### `GET /health` — 健康檢查

**Response:**

```json
{
  "status": "ok",
  "model": "SAM3"
}
```

## 回傳邏輯

- 若多個物件符合 prompt，自動回傳 **信心分數最高** 的那一個。
- `image_path` 與 `image`（base64）二擇一；同機器建議用 `image_path` 避免 base64 傳輸開銷。

## 從其他環境呼叫（Client 使用方式）

Client 只依賴 `requests` + `Pillow`，可在任何 Python 環境中使用，不需要 GPU 或 sam3 套件。

### 基本用法

```python
from sam3_server.client import segment_object

# 用本機路徑（推薦）
result = segment_object("truck", image_path="/path/to/photo.jpg")
result["cutout"].save("truck_cutout.png")
```

### 用 PIL Image

```python
from PIL import Image
from sam3_server.client import segment_object

img = Image.open("photo.jpg")
result = segment_object("dog", image=img)
```

### 回傳值

```python
result["cutout"]  # PIL RGBA Image — 帶透明背景的物件切割圖
result["mask"]    # PIL L-mode Image — 二值遮罩
result["bbox"]    # [x1, y1, x2, y2]
result["score"]   # 信心分數 (float)
```

未偵測到物件時回傳 `None`。

### 確認 Server 是否在線

```python
from sam3_server.client import is_server_running

if is_server_running():
    print("Server is up")
```

### 自訂參數

```python
# 調低 confidence 門檻
result = segment_object("small bird", image_path="photo.jpg", confidence=0.2)

# 指定不同 server 位址
result = segment_object("car", image_path="photo.jpg", server_url="http://192.168.1.100:5051")
```

## 用 curl 測試

```bash
# 健康檢查
curl http://localhost:5051/health

# 物件分割（本機路徑）
curl -X POST http://localhost:5051/segment \
  -H "Content-Type: application/json" \
  -d '{"prompt": "truck", "image_path": "/path/to/photo.jpg"}'
```

## 檔案結構

```
sam3_server/
├── __init__.py
├── server.py          # HTTP server（Python stdlib，無需 Flask）
├── client.py          # 輕量 client，可從任何環境呼叫
└── start_server.sh    # 一鍵啟動腳本（自動 activate conda env）
```

## 注意事項

- Server 啟動時會載入 SAM3 模型至 GPU，需要約 7 GB VRAM。
- Server 使用 single-threaded（同一時間僅處理一個 request）。若需批次處理請用迴圈逐筆呼叫。
- 若 GPU 同時跑 Qwen3 server（~8–9 GB），請確認 VRAM 足夠（建議 ≥ 24 GB）。
