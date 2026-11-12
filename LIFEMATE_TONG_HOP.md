# Zalo AI Challenge 2026 — LifeMate: Tổng hợp toàn bộ thông tin

Nguồn: https://challenge.zalo.ai/portal/lifemate (các tab Description / Evaluation / Rule / Submission Guidance, Leaderboard), trang chủ challenge.zalo.ai (Timeline, Prize, FAQ), `zalo-ai-challenge-regulations-vi.pdf` (14 trang), và gói dữ liệu `train+public_test` (README, VERSION, catalog, zaic_tools). Đọc ngày 01/11/2026.

> **Cập nhật 09/11/2026**: đọc lại portal, trang chủ và `Docker_Guide_ZAC2026_LifeMate.pdf` (mới, ngày 05/11). Các điểm mới được đánh dấu **🆕**. Không đổi: Description, công thức chấm điểm, bộ dữ liệu (vẫn v3.2.0, md5 các file jsonl/catalog không đổi).

> Tab **Data** trên portal yêu cầu đăng nhập nên không đọc được. Nội dung tương đương có trong README của gói dữ liệu (mục 4).

---

## 1. Đề bài (LifeMate — Agentic AI That Turns Your Voice Into Action)

Xây dựng AI agent nhận **yêu cầu bằng giọng nói tiếng Việt** (có thể lẫn từ tiếng Anh), rồi **gọi tool của các app** để thực hiện. Chấm theo việc agent **làm gì**, không theo việc agent **nói gì**. Giọng nói gồm giọng tổng hợp và giọng thật, ba miền Bắc, Trung, Nam.

**Input mỗi mẫu**
- Audio yêu cầu cuối (`audio/<id>/<id>.wav`), đây là phần được chấm.
- Context: `current_time`, `timezone`, `location` (lat/lng), `user_profile` (tên, địa chỉ nhà, nơi làm việc, thành phố), `installed_apps` (chỉ được gọi các tool này).
- History: các lượt trước. Lượt user chỉ có audio (`<id>_u1.wav`, ...). Lượt assistant có text, tool_calls, tool_results; các thao tác này **đã được áp vào state**.
- Môi trường mô phỏng (state.json): chỉ đọc/ghi qua tool.

**Output**: danh sách tool call có thứ tự, kèm `state_hash`. Có 3 kiểu hành vi:
1. **Thực thi**: gọi các tool cần thiết theo trình tự để đạt state cuối đúng.
2. **Thiếu thông tin**: gọi `sys.ask_missing_info` với `{"tool_name": "app.tool", "params": [...]}`. Có thể gọi sau khi đã chạy các tool khác (ví dụ: tắt đèn → `contacts.search` → hỏi `contact_id` của `sms.send`).
3. **Ngoài phạm vi**: trả `[]` nếu không app nào đã cài làm được.

---

## 2. Cách chấm điểm (Evaluation)

Mỗi mẫu được chấm trên hai tiêu chí: state cuối có đúng không, và tool call khớp với gold đến mức nào.

- **State match**: `state_hash` nộp phải bằng gold, nếu không thì mẫu đó được **0 điểm**. Trường free-text (fuzzy) không ảnh hưởng hash. Hash phải tính bằng `env.get_state_hash()`. BTC sẽ replay tool_calls bằng zaic_tools; nếu kết quả khác thì lấy kết quả của BTC.
- **Tool-call match**: một call khớp gold khi trùng tên tool, mọi tham số non-fuzzy giống hệt, và mọi tham số fuzzy có **token-F1 > 0.6** (sau khi lowercase và Unicode NFC). Mỗi call khớp tối đa 1 call gold. **Không xét thứ tự.**
- **Out-of-scope**: nếu gold rỗng, mẫu được 1 điểm chỉ khi bài nộp cũng rỗng; ngược lại 0.
- `sys.ask_missing_info` được chấm như mọi tool khác.

**Công thức** (khi gold không rỗng):

```
Score = 0                      nếu hash_sys ≠ hash_gold hoặc M = 0
      = 2·P·R / (P + R)        ngược lại
P = M / (M + U)      R = M / |G|
Final score = 100/N · Σ Score_i
```
- M = số call khớp, |G| = số call gold, U = số call nộp không khớp (**kể cả call bị từ chối do sai schema**), N = số mẫu.
- ⇒ Gọi thừa tool **read** vẫn làm giảm Precision. Call sai schema cũng bị tính là U.

**Phân định hòa**: chấm lại F1 tool_calls và bỏ qua hash. Nếu vẫn hòa, đội có thời gian xử lý trung bình mỗi mẫu thấp hơn xếp trên. (Theo Thể lệ chung, trên public leaderboard nếu bằng điểm thì bài nộp sớm hơn xếp trên.)

---

## 3. Ràng buộc kỹ thuật (Rule)

| Hạng mục | Yêu cầu |
|---|---|
| Tổng tham số hệ thống (ASR + LLM + mọi thành phần) | **< 6B** |
| LLM | **< 5B** tham số, context **≤ 16K (16,384) tokens** |
| Thời gian | **≤ 30 giây/mẫu**; quá giờ hoặc lỗi thì mẫu đó 0 điểm |
| Phần cứng chấm | 1× **NVIDIA A30** (24GB), CPU 16 nhân Xeon Gold 6442Y, **64GB RAM**. 🆕 ⚠️ Docker Guide trang 4 lại ghi RTX 5090 32GB / Xeon Gold 6459C, nhưng ảnh `nvidia-smi` trong chính guide là A30 24GB ⇒ **lên kế hoạch theo 24GB** |
| Mạng | Chạy **hoàn toàn offline**, không gọi API ngoài khi inference |
| Tái lập | Cố định mọi seed (Python, NumPy, PyTorch...), decode deterministic (greedy / temperature=0). Nếu chạy lại ra kết quả khác, BTC chạy Docker 5 lần và lấy trung bình |
| Truy cập dữ liệu | **Không được đọc/copy/parse** bất kỳ file nào trong `<split>_environment/` (kể cả state.json). Chỉ dùng zaic_tools, **đúng một ToolEnv/mẫu**, mọi call đã gọi phải có trong bài nộp. 🆕 **Khi inference chỉ được dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`**; cấm `get_state`, `states`, `Executor`, `Simulator`, clone env (`get_state` chỉ dùng khi train) |
| Dữ liệu/mô hình | Dữ liệu công khai và model mã nguồn mở **phát hành trước 26/10/2026**, giấy phép cho phép. Được sinh dữ liệu tổng hợp bằng dịch vụ/model khác (LLM, TTS...) trong lúc phát triển, nhưng **không dùng khi inference** |
| Cấm | Gán nhãn public_test/private_test (thủ công hoặc bằng model/dịch vụ) để train/tune |

Quy định thêm trong README của zaic_tools:
- Không được tạo lại ToolEnv để "thử rồi làm lại".
- Call chỉ bị từ chối khi sai schema: tool không tồn tại hoặc chưa cài, thiếu `required`, sai type/enum/min/max/format/items, thiếu cả nhóm `x-one-of`.
- ID không tồn tại, không đủ số dư, không tìm thấy... **không phải lỗi**: tool vẫn chạy, không đổi state, trả `{"ok": true, "done": false, ...}` hoặc danh sách rỗng.
- App thông tin (weather, gold_price, fuel_price...) chỉ đọc và không thuộc phần state được so.

---

## 4. Vòng thi, mốc thời gian, cách nộp (GMT+7)

| Ngày | Sự kiện |
|---|---|
| 26/10/2026 | Bắt đầu; phát hành dev kit, train, public test, phần mềm đánh giá |
| 26/10 – 16/11 | **Vòng 1 – Public Test**: nộp `submission.jsonl` trên portal, **tối đa 5 lần/ngày** (reset cuối ngày) |
| 16/11/2026, 23:59 | **Nộp final solution**: Docker (train code, inference code, data) qua form "Solution Submission"; gửi **link Google Drive public + checksum**; tối đa 2 lần, BTC chấm bản cuối |
| 🆕 16/11/2026, 00:00 | Khóa sửa thông tin đội, thêm/xóa thành viên, xóa đội (FAQ trang chủ; bản cũ ghi 17/11) |
| 20/11/2026, 🆕 11:00 | Phát hành **private test** (trang chủ ghi 11:00; tab Submission Guidance vẫn ghi 9:00) |
| 23/11/2026, 23:59 | Hạn nộp kết quả private test (chạy bằng solution đã đóng băng) |
| 02/12/2026 | Công bố top 2 mỗi đề |
| 12/12/2026 | Công bố kết quả chung cuộc |

**Docker**: BTC mount thư mục private test (cấu trúc giống public_test) vào `/data`, chạy script inference, lấy kết quả ở `result/submission.jsonl`. Container phải chạy offline. Vòng 2: đội tự chạy solution đã đóng băng trên private test rồi nộp; BTC chạy lại Docker để kiểm chứng.

🆕 **Docker Guide** (`Docker_Guide_ZAC2026_LifeMate.pdf`, link: https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf):
- Base image **CUDA ≤ 13.2** (driver BTC 595.58.03 chạy được CUDA 12.x và 13.x ≤ 13.2), ví dụ `nvidia/cuda:13.2.0-devel-ubuntu22.04`.
- Script `predict.sh`: đọc `/data/private_test/` (`private_test.jsonl`, `private_test_tools_catalog.json`, `private_test_environment/`, `audio/<id>/<id>.wav` + `<id>_u1.wav`…, `zaic_tools/`), ghi `/result/submission.jsonl`; in thời gian load model và thời gian predict ra stdout.
- **Import `zaic_tools` từ `/data/<split>`** (`sys.path.insert(0, "/data/private_test")`), KHÔNG dùng bản đóng trong image.
- **Mỗi dòng nộp có thêm trường `"time"`** = ms xử lý mẫu, đo từ trước khi đọc audio đến sau `get_state_hash()`.
- Mọi model (ASR, LLM, tokenizer…) nằm sẵn trong image, không tải HuggingFace lúc chạy. `--network host` chỉ dùng lúc dựng image; lúc chấm không có mạng.
- Cấu trúc gợi ý: `predict.py`, `preprocessing.py`, `saved_models/`, `requirements.txt`, `predict.sh`, `start_jupyter.sh`, `predict_notebook.ipynb` (đo thời gian), `training_code/` (+ `README.md`).
- ⚠️ PDF chỉ có 8 trang, **bị cắt giữa mục 5**; các mục 6 (bổ sung bắt buộc về training code & tài liệu), 7 (cấu trúc notebook đo thời gian), 8 (upload docker) có trong mục lục nhưng không có nội dung ⇒ cần hỏi BTC (aichallenge@zalo.me) hoặc tải lại sau.

**Định dạng nộp** (`submission.jsonl`, UTF-8, mỗi mẫu đúng 1 dòng; 🆕 bản Docker thêm `"time"`):
```json
{"id": "public_0039", "tool_calls": [{"name": "alarm.list", "arguments": {}}, {"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}], "state_hash": "461fcd40..."}
{"id": "public_0003", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "music.add_to_playlist", "params": ["playlist_id", "track_id"]}}], "state_hash": "23f1c7d2..."}
{"id": "public_0002", "tool_calls": [], "state_hash": "0c9a2ec0..."}
```
Kiểm tra schema file nộp: `python -m zaic_tools --split public_test --root . --validate submission.jsonl` (chạy với `PYTHONPATH=public_test`).

---

## 5. Giải thưởng

- Tổng giải thưởng toàn cuộc thi: **USD 12,000** tiền mặt, cộng quà hiện vật.
- Mỗi đề: **Giải Nhất 90.412.000 VND (~3.500 USD)**, **Giải Nhì 64.580.000 VND (~2.500 USD)** (tỷ giá MSB ngày 23/10/2026). Top 5 mỗi đề nhận gói quà ZAIC.
- Thuế được khấu trừ trước khi trao giải.
- Thủ tục nhận giải: phản hồi trong 1 tuần kể từ lần liên hệ đầu; nộp hồ sơ trong 2 tuần kể từ thông báo; tiền được trao trong 30 ngày sau khi đủ hồ sơ. Phải nộp mã nguồn cuối cùng, tài liệu, hướng dẫn chạy và cấp phép sử dụng (Mục 8). Không được chuyển nhượng giải.

---

## 6. Thể lệ chung (PDF) — các điểm quan trọng

- **Điều kiện**: quốc tịch Việt Nam, có tài khoản Zalo hợp lệ, từ 18 tuổi. Nhân viên/CTV/người có hợp đồng với VNG và công ty thuộc tập đoàn **không được tham gia**.
- **Đội**: tối đa **5 người**. Mỗi người chỉ dùng 1 tài khoản Zalo và chỉ ở 1 đội (hoặc thi cá nhân). Cấm dùng nhiều tài khoản hoặc tạo nhiều đội. Bài của thành viên đã bị xóa khỏi đội vẫn tính cho đội.
- **Gộp đội**: được phép, do đội trưởng thực hiện. Điều kiện: tổng lượt nộp sau gộp ≤ (lượt/ngày × số ngày đã diễn ra).
- **Nộp bài**: tối đa **5 bài/ngày**, chọn 1 bài cuối để chấm chính thức. Bài nộp trễ không được tính.
- **Đóng băng mô hình** trước khi công bố private test: nộp mã huấn luyện, mã kiểm thử, mô hình, tài liệu mô tả phương pháp và hướng dẫn chạy. BTC so kết quả private với bản đóng băng và có thể yêu cầu mã nguồn.
- **Dữ liệu**: chỉ dùng phi thương mại hoặc học thuật, trích dẫn "challenge.zalo.ai". **Không chia sẻ ra ngoài đội**; xóa dữ liệu cá nhân khi cuộc thi kết thúc. Được thêm sub-label cho train; **cấm** dùng nhãn thủ công hoặc dự đoán của con người trên tập test.
- **Dữ liệu ngoài**: phải hợp pháp, tuân thủ Luật BVDLCN 2025 và NĐ 356/2025; phải chứng minh được khi BTC yêu cầu, nếu không bài bị coi là không hợp lệ.
- **Mã nguồn mở**: chỉ dùng thư viện/mã có giấy phép OSI, **không giới hạn dùng thương mại**.
- **Chia sẻ code**: trong thời gian thi, không chia sẻ riêng ngoài đội. Chỉ được công khai cho tất cả thí sinh trên diễn đàn chính thức, và khi đó code mặc nhiên theo giấy phép OSI.
- **Sở hữu trí tuệ**: tác giả giữ quyền; BTC được quyền sử dụng không độc quyền, vĩnh viễn, miễn phí cho mục đích nghiên cứu, quảng bá, giáo dục.
- **Vi phạm**: gian lận có thể bị loại ngay; vi phạm khắc phục được có 1 tuần để sửa.
- **Khiếu nại**: trong **7 ngày làm việc** kể từ khi công bố kết quả chính thức. Quyết định của BTC là cuối cùng.
- **Liên hệ**: **aichallenge@zalo.me**

---

## 7. Bộ dữ liệu đã tải (`E:\zalo\train+public_test\train+public_test`)

- Phiên bản: `release v3.2.0`, build 2026-10-24. **Train 1545 mẫu / Public test 634 mẫu.** Private test phát hành sau.
- Audio: WAV **16 kHz, mono, 16-bit**, dài khoảng 1,3–12,4 giây (trung bình ~4,7 giây). Tổng 6969 file wav (gồm cả audio các lượt history).
- Train có `audio_text` (transcript) cho lượt cuối và các lượt user trong history, cùng `label` = {`tool_calls`, `tool_results`, `final_state`}. Public test **không có** transcript.
- Phân bố nhãn train: thực thi 1217, ask_missing_info 217, ngoài phạm vi (`[]`) 111. Số call/mẫu: 0–6 (thường 1–3).
- History: khoảng 34% mẫu không có history, dài nhất 24 lượt (≈12 cặp user/assistant).
- Tool hay gặp: `contacts.search`, `sys.ask_missing_info`, `smart_home.light_set`, `reminders.create`, `bill_payment.list_bills/pay`, `smart_home.get_status`, `calendar.list_events`, `chat_ott.*`, `banking.transfer`...

### ⚠️ Tập app của train và public test khác nhau (cần khả năng tổng quát hóa)
- **Chỉ có trong public test (11 app)**: ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator.
- **Chỉ có trong train (21 app)**: appliances, bike_rental, calculator, door_lock, exchange_rate, fitness, home_automation, home_energy, home_security, household_chores, learning, library, lottery, loyalty, medication, news, package_tracking, podcast_radio, robot_vacuum, subscriptions, video_meeting.
- ⇒ Private test rất có thể còn có app mới. Agent phải đọc JSON schema của tool lúc chạy, không được học thuộc tên tool.

### Schema tool đáng chú ý
- `kind`: `read` / `write` / `meta`. `x-one-of`: bắt buộc một trong các nhóm (ví dụ `contact_id` hoặc `phone_number`). `x-ref`: id lấy từ dữ liệu app khác (ví dụ `contacts.contacts`). `x-match: fuzzy`: văn bản tự do, chấm token-F1 > 0.6.
- `sys.ask_missing_info`: `{"tool_name": "<app.tool>", "params": ["<tên tham số đúng schema>"]}`.

### API zaic_tools (Python 3.10+, chỉ dùng thư viện chuẩn)
```python
env = ToolEnv("public_test", "public_0001", root=".")
env.call(tool, args)      # chạy tool, state đổi trong bộ nhớ
env.validate(tool, args)  # chỉ kiểm schema; đã xác minh trong executor.py: KHÔNG ghi vào tool_calls
env.get_tool_calls(); env.get_state_hash(); env.get_state()
```
🆕 `env.get_state()` **chỉ dùng khi train**; lúc inference chỉ được `validate` / `call` / `get_tool_calls` / `get_state_hash`.
State hash = sha256 của `canon(state)`, bỏ `_meta`, app thông tin và các trường text tự do.

---

## 8. Public leaderboard

🆕 **Từ khoảng 09/11, leaderboard chia 3 hạng mục**: **SLM** (model < 6B), **LLM** (open-source ≥ 6B), **APIs/Others** (Claude, Gemini, OpenAI…). **Chỉ xếp hạng theo SLM**; LLM và APIs chỉ để tham khảo.

**09/11/2026 (SLM):** Uống nào 99.22 · Caro 99.21 · TrustedAI 96.66 · BXT 96.58 · Tấn Cường - Tèo 96.25 · 67Labs 95.74 · Pawgrammers 95.72 · DamTinhMienTrung 95.27 · Piece Of Cake 95.12 · Akatsuki 95.00. Top 20 ≥ 91.67. ⇒ Hai đội đầu bỏ xa nhóm sau ~2,5 điểm.

*01/11/2026 (bảng cũ, chưa chia hạng mục):* Xcen 99.37 · PVC 98.95 · Hasagi 98.66; top 20 ≥ 95.88. Các đội này không còn ở top SLM ⇒ nhiều khả năng điểm cũ dùng API.

---

## 9. Hàm ý chiến lược (tóm tắt)

1. Pipeline: **ASR tiếng Việt** (3 miền, có lẫn tiếng Anh) → **LLM < 5B** làm agent (vòng lặp gọi tool) → zaic_tools. Tổng < 6B tham số, ≤ 30 giây/mẫu trên A30, context 16K.
2. State hash sai thì mẫu đó 0 điểm, nên các tham số non-fuzzy (id, thời gian ISO, số tiền, enum) phải chính xác tuyệt đối.
3. Mỗi call thừa (kể cả read, kể cả call sai schema) làm giảm P. Nên dùng `env.validate` trước `env.call` (validate không bị tính vào bài nộp). Bám sát phong cách gold: khi nào gold gọi `get_status` / `contacts.search` trước.
4. History cũng là audio, nên phải ASR cả các lượt user trước để hiểu ngữ cảnh (đại từ, "cái đó", "người lúc nãy").
5. Thời gian tương đối ("mai", "tối nay") tính theo `context.current_time` + timezone.
6. Phải tổng quát hóa sang app chưa thấy: đưa schema vào prompt, sinh thêm dữ liệu tổng hợp (LLM/TTS ngoài chỉ dùng khi phát triển).
7. 🆕 Guard / vòng ReAct lúc inference **không được đọc state** (`get_state` bị cấm); mọi suy luận phải dựa trên kết quả trả về của `env.call`.
8. 🆕 Docker: base CUDA ≤ 13.2, `zaic_tools` lấy từ `/data/<split>`, ghi thêm `"time"` mỗi dòng; tính bộ nhớ theo A30 24GB.
