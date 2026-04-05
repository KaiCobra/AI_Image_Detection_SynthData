# AI Detection Dataset — 合成 Pipeline

將 AI 生成圖片的物件切割後疊合到真實圖片上，產生帶遮罩的訓練資料。

## 架構總覽

```
prompts.csv (filename, caption)
       │
       ▼
┌──────────────┐     caption      ┌──────────────┐
│  Synthbuster │ ──────────────▶  │ Qwen3 Server │  (port 5050)
│  AI 圖片庫   │                  │ 提取物件單字  │
└──────┬───────┘                  └──────┬───────┘
       │ image                           │ object word
       │                                 │
       │         ┌───────────────────────┘
       │         │
       ▼         ▼
┌──────────────────┐
│   SAM3 Server    │  (port 5051)
│ 文字 prompt 分割 │
└──────┬───────────┘
       │ RGBA cutout (or full image if </nothing>)
       │
       ▼
┌──────────────────┐     real image     ┌───────────┐
│    Compositor    │ ◀───────────────── │ data/real/ │
│  疊合 + 遮罩生成 │                    └───────────┘
└──────┬───────────┘
       │
       ▼
  batch_NNNN/
    ├── composite.png   — AI 物件疊合在真實圖片上
    ├── mask.png        — 二值遮罩（255=AI, 0=real）
    └── metadata.json   — 來源、座標、信心分數
```

## 資料流程

### Step 1：讀取 prompts.csv

```
data/ai/synthbuster/synthbuster/prompts.csv
```

| 欄位 | 說明 |
|------|------|
| 第 1 欄 `image name` | 檔案名稱（不含副檔名），對應每個生成器子目錄下的 `.png` |
| 第 2 欄 `Prompt` | 該 AI 圖片的生成 caption |

每個檔名在 9 個生成器子目錄中都存在（dalle2、dalle3、firefly、glide、midjourney-v5、SD 1.3/1.4/2/XL），pipeline 會隨機選擇一個生成器。

### Step 2：Qwen3 提取物件

將 caption 送至 Qwen3 server，取得可移除的前景物件單字：

```python
from qwen3_server.client import extract_object

obj = extract_object("a bicycle leaning against an old brick wall")
# → "bicycle"
```

- 若回傳 `</nothing>`（場景無明確前景物件），跳過 SAM3，直接用整張 AI 圖片疊圖。

### Step 3：SAM3 分割物件

將物件單字 + AI 圖片送至 SAM3 server，取得 RGBA 切割圖：

```python
from sam3_server.client import segment_object

result = segment_object("bicycle", image_path="data/ai/synthbuster/synthbuster/dalle3/r0b0e2ba5t.png")
cutout = result["cutout"]  # PIL RGBA — 只有 bicycle 區域有像素，其餘透明
```

- 若 SAM3 也找不到物件（`result is None`），同樣退回用整張 AI 圖片。

### Step 4：疊合

| 情境 | Base（底圖） | Overlay（疊上去的） |
|------|-------------|-------------------|
| SAM3 成功 | real image | SAM3 RGBA cutout |
| `</nothing>` 或 SAM3 失敗 | real image | 整張 AI image（隨機位置 + 大小） |

- AI 永遠是疊在上面的圖層，real 永遠是底圖。
- 遮罩 (mask.png)：AI 像素所在的區域標記為 255，其餘為 0。

## 前置需求

啟動兩個 server（各自獨立的 conda 環境）：

```bash
# Terminal 1 — Qwen3 server
bash qwen3_server/start_server.sh          # port 5050

# Terminal 2 — SAM3 server
bash sam3_server/start_server.sh            # port 5051
```

確認兩個 server 都就緒後再執行 pipeline：

```bash
curl http://localhost:5050/health
curl http://localhost:5051/health
```

## 執行

```bash
python generate_dataset.py --batches 100 --output output/batches
```

## 輸出結構

```
output/batches/
├── batch_0000/
│   ├── composite.png
│   ├── mask.png
│   └── metadata.json
├── batch_0001/
│   ...
```

### metadata.json 範例

```json
{
  "ai_source": "synthbuster/dalle3",
  "ai_filename": "r0b0e2ba5t",
  "ai_caption": "branches of trees that surround a lake, ...",
  "ai_object": "bicycle",
  "sam3_score": 0.863,
  "sam3_bbox": [85, 282, 1711, 850],
  "real_source": "disk",
  "overlay_position": [50, 30],
  "canvas_size": [512, 512]
}
```

## Server 資訊快速對照

| Server | Port | Conda 環境 | 啟動指令 | 文件 |
|--------|------|-----------|---------|------|
| Qwen3 | 5050 | `/media/.../conda_envs/qwen3` | `bash qwen3_server/start_server.sh` | [doc/qwen3Server.md](qwen3Server.md) |
| SAM3 | 5051 | `sam3` | `bash sam3_server/start_server.sh` | [doc/sam3Server.md](sam3Server.md) |

## 注意事項

- 兩個 server 合計需要約 16 GB VRAM（Qwen3 ~9 GB + SAM3 ~7 GB），建議使用 24 GB 以上的 GPU。
- Qwen3 和 SAM3 server 使用不同的 conda 環境，避免套件衝突。
- Server 皆為 single-threaded，pipeline 逐筆循序呼叫。
- 若 `prompts.csv` 用完（999 筆 + header），pipeline 會循環使用。
