# DR3 user-file insight extractor

Pipeline cục bộ này tạo lại **candidate I_UF insights** từ `query.txt` và các file
chính thức nằm trực tiếp trong `data/<task>/user_files/`. Nó không clone DR3-Eval,
không dùng sandbox corpus, không sinh I_SC và không tìm file bên ngoài `user_files/`.

> `gold_insights` chỉ là tên key để tương thích evaluator. Nội dung được sinh bởi
> máy và **phải được kiểm tra thủ công** trước khi được xem là final gold ground truth
> hoặc human-verified data.

## Cài đặt

Yêu cầu Python 3.10+:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền `OPENROUTER_API_KEY` trong `.env`. Không đưa `.env` vào source control. Model
mặc định là `qwen/qwen3.7-flash` qua OpenRouter OpenAI-compatible API; có thể đổi
model và timeout qua các biến trong `.env`.

## Chạy

Kiểm tra parser mà không gọi API và không ghi output:

```powershell
python extract_insights.py --task 008 --parse-only
python extract_insights.py --all --parse-only
```

Sinh insight cho một task hoặc toàn bộ task:

```powershell
python extract_insights.py --task 008
python extract_insights.py --all
```

Các path đều có CLI option, ví dụ:

```powershell
python extract_insights.py --task 008 `
  --data-dir data `
  --output-dir output `
  --prompt prompts/user_file_insight_prompt.txt
```

## Cách xử lý source

- TXT, CSV, XLSX và DOCX được parse deterministic; DOCX giữ thứ tự paragraph/table,
  CSV giữ columns/rows, XLSX giữ sheet names, formulas, columns và tất cả row.
- JPG/JPEG/PNG/WEBP không qua OCR; byte ảnh gốc được gửi trực tiếp bằng multimodal
  `image_url` data URL.
- PDF dùng PyMuPDF theo từng page. Page có text dùng text; page gần như không có text
  hoặc image-heavy được render sang PNG và gửi vision. Boundary `[Page N]` luôn được giữ.
- Mỗi original file luôn là một final source. Source dài được chia theo page/heading/
  sheet/row section, trích candidate theo chunk, rồi consolidation về tối đa 1-2 insight
  cho filename gốc. Không có row/page/chunk nào bị silently truncated.

Ngưỡng chunk có thể cấu hình bằng `MAX_SOURCE_CHARS`, `CHUNK_TARGET_CHARS` và
`MAX_IMAGES_PER_REQUEST`.

## Output và lỗi

Mỗi task tạo:

```text
output/<task>/
├── raw/<source>.json
├── generated_insights.json
└── run_metadata.json
```

`raw/` lưu nguyên response từng lần gọi (direct/chunk/consolidation và JSON repair).
`run_metadata.json` lưu model, temperature, hash prompt, timestamp, source type,
text/vision/hybrid path, chunking và trạng thái từng source. JSON được validate; nếu
response lỗi format/schema, pipeline chỉ yêu cầu model repair đúng một lần và không tự
bịa insight local. Lỗi một source được log và các source/task còn lại vẫn tiếp tục.

## Sinh checklist Instruction Following

Checklist pipeline độc lập chỉ đọc `data/<task>/query.txt`; nó không đọc
`user_files/`, generated insights hay nguồn bên ngoài. Kiểm tra prompt/query mà không
gọi API hoặc ghi output:

```powershell
python generate_checklists.py --task 008 --validate-only
python generate_checklists.py --all --validate-only
```

Sinh checklist cho một task hoặc toàn bộ task:

```powershell
python generate_checklists.py --task 008
python generate_checklists.py --all
```

Mỗi task ghi `checklist.json`, `checklist_metadata.json` và raw response tại
`raw/checklist_raw.txt`. Nếu JSON cần model repair, response repair được lưu riêng tại
`raw/checklist_repair_raw.txt`. `checklist.json` là machine-generated candidate và phải
được human review trước khi dùng làm checklist IF chính thức.
