# Vietnamese Emotion Classification Demo

Demo này cho phép user nhập câu tiếng Việt trên giao diện Streamlit, sau đó hệ thống gọi model PhoBERT ONNX đang được serve bằng NVIDIA Triton Inference Server để dự đoán cảm xúc.

## Kiến Trúc

```text
User nhập text tiếng Việt
        |
        v
Streamlit GUI
        |
        v
PhoBERT tokenizer trong Python app
        |
        v
input_ids + attention_mask
        |
        v
Triton Inference Server chạy trong Docker
        |
        v
model.onnx
        |
        v
logits -> softmax -> emotion label
        |
        v
Hiển thị kết quả trên GUI
```

Triton chỉ nhận tensor, không nhận raw text. Vì vậy phần Streamlit app sẽ tokenize câu tiếng Việt trước, rồi gửi `input_ids` và `attention_mask` sang Triton.

## Cấu Trúc Thư Mục

```text
.
├── app/
│   ├── __init__.py
│   ├── requirements.txt
│   ├── streamlit_app.py
│   ├── tokenizer.py
│   └── triton_client.py
│
├── model_repository/
│   └── emotion_phobert/
│       ├── config.pbtxt
│       ├── labels.txt
│       └── 1/
│           └── model.onnx
│
├── onnx_emotion_phobert_v2/
│   ├── added_tokens.json
│   ├── bpe.codes
│   ├── config.json
│   ├── model.onnx
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── docker-compose.yml
└── README.md
```

## Model Triton

Model Triton nằm ở:

```text
model_repository/emotion_phobert/
```

Model ONNX nằm trong version folder:

```text
model_repository/emotion_phobert/1/model.onnx
```

File `config.pbtxt` khai báo model có:

```text
input_ids: INT64, shape [batch, 256]
attention_mask: INT64, shape [batch, 256]
logits: FP32, shape [batch, 7]
```

Streamlit app cũng tokenize với `max_length=256`, nên shape gửi sang Triton khớp với `config.pbtxt`.

## Yêu Cầu

Cần có:

- Docker Desktop
- Python 3.10 trở lên
- Internet trong lần đầu pull Docker image và install Python packages

Nếu dùng Windows PowerShell, nên chạy các lệnh bên dưới từ thư mục project này.

## Bước 1: Chạy Triton Server Bằng Docker

Cách khuyên dùng là Docker Compose:

```powershell
docker compose up triton
```

Lệnh này sẽ:

- Pull image `nvcr.io/nvidia/tritonserver:25.12-py3` nếu máy chưa có.
- Mount thư mục `model_repository` vào container tại `/models`.
- Chạy Triton với `--model-repository=/models`.
- Mở các port:
  - `8000`: HTTP endpoint
  - `8001`: gRPC endpoint
  - `8002`: metrics endpoint

Nếu muốn dùng image Triton khác, set biến môi trường trước khi chạy:

```powershell
$env:TRITON_IMAGE="nvcr.io/nvidia/tritonserver:26.03-py3"
docker compose up triton
```

Bạn cũng có thể chạy trực tiếp bằng `docker run`:

```powershell
docker run --rm `
  --name emotion-phobert-triton `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}\model_repository:/models" `
  nvcr.io/nvidia/tritonserver:25.12-py3 `
  tritonserver --model-repository=/models
```

Giữ terminal Triton đang chạy, mở terminal mới để chạy Streamlit.

## Bước 2: Kiểm Tra Triton Đã Ready

Mở terminal PowerShell khác và chạy:

```powershell
Invoke-RestMethod http://localhost:8000/v2/health/ready
```

Nếu không báo lỗi nghĩa là Triton server đã sẵn sàng.

Kiểm tra model:

```powershell
Invoke-RestMethod http://localhost:8000/v2/models/emotion_phobert
```

Khi model load thành công, response sẽ có tên model `emotion_phobert`, input `input_ids`, `attention_mask`, và output `logits`.

## Bước 3: Cài Dependencies Cho Streamlit App

Tạo virtual environment:

```powershell
python -m venv .venv
```

Kích hoạt virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài package:

```powershell
python -m pip install --upgrade pip
python -m pip install -r app\requirements.txt
```

## Bước 4: Chạy GUI Streamlit

Khi Triton đang chạy ở `localhost:8000`, chạy:

```powershell
streamlit run app\streamlit_app.py
```

Streamlit sẽ mở một web UI trong trình duyệt. Bạn nhập một câu tiếng Việt, bấm `Predict`, app sẽ trả về nhãn cảm xúc và độ tin cậy.

Nếu Triton chạy ở host hoặc port khác, set biến môi trường:

```powershell
$env:TRITON_URL="localhost:8000"
$env:TRITON_MODEL_NAME="emotion_phobert"
streamlit run app\streamlit_app.py
```

## Nhãn Cảm Xúc

Model trả về 7 nhãn:

```text
0: Enjoyment
1: Sadness
2: Anger
3: Fear
4: Disgust
5: Surprise
6: Other
```

## Cách Hoạt Động Của App

Khi user bấm `Predict`:

1. Streamlit đọc text từ ô nhập.
2. `app/tokenizer.py` load tokenizer từ `onnx_emotion_phobert_v2`.
3. Text được tokenize thành `input_ids` và `attention_mask` với shape `[1, 256]`.
4. `app/triton_client.py` gửi 2 tensor này sang Triton qua HTTP.
5. Triton chạy `model_repository/emotion_phobert/1/model.onnx`.
6. Triton trả `logits` shape `[1, 7]`.
7. App chạy softmax, chọn nhãn có xác suất cao nhất và hiển thị trên GUI.

## Troubleshooting

### Streamlit báo không kết nối được Triton

Kiểm tra Triton container còn chạy không:

```powershell
docker ps
```

Kiểm tra endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/v2/health/ready
```

### Triton báo lỗi shape input

Đảm bảo `app/streamlit_app.py` dùng:

```text
MAX_LENGTH=256
```

và `model_repository/emotion_phobert/config.pbtxt` dùng:

```text
dims: [ 256 ]
```

Hai giá trị này phải khớp nhau.

### Docker báo port đã được sử dụng

Có thể port `8000`, `8001`, hoặc `8002` đang bị process khác dùng. Dừng container cũ:

```powershell
docker stop emotion-phobert-triton
```

rồi chạy lại:

```powershell
docker compose up triton
```

### Docker không pull được image Triton

Thử đổi image tag:

```powershell
$env:TRITON_IMAGE="nvcr.io/nvidia/tritonserver:25.12-py3"
docker compose up triton
```

Nếu Docker yêu cầu đăng nhập NVIDIA NGC, đăng nhập `nvcr.io` theo hướng dẫn trên NGC rồi chạy lại lệnh.

## Ghi Chú

- Demo hiện tại ép Triton chạy CPU bằng `instance_group: KIND_CPU`, nên dễ chạy trên laptop hơn.
- Nếu muốn dùng GPU, cần sửa `config.pbtxt` sang `KIND_GPU` và chạy Docker với `--gpus all`.
