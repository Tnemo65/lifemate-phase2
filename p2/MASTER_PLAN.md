# LifeMate Phase 2 — Phân tích tổng thể và phương án (12/11/2026)

Phạm vi: golden transcript → tool call, model < 5B (Qwen3-4B-Instruct-2507), chấm trên mẫu gold chưa từng thấy.
Quy ước trong tài liệu: **[đo]** = số tự đo trong dự án; **[paper]** = số đọc trực tiếp từ paper; **[giả thuyết]** = chưa có số đo.

---

## 1. Hiện trạng: ta đang có gì

| Thành phần | Hiện có |
|---|---|
| Dữ liệu gold | 1.545 mẫu có nhãn. Chia: TRAIN 1.127 / DEV 209 / FINAL_TEST 209 (khoá). DEV giữ riêng 7 app làm "app lạ" |
| Hai họ dữ liệu | Template (câu mẫu nhiều ý, 587 mẫu TRAIN) và hội thoại do LLM sinh (history dài, phương ngữ, 391 mẫu TRAIN); 149 mẫu không có ID |
| Vòng lặp agent | ReAct: mỗi bước model sinh 1 tool call → simulator chạy → model xem kết quả → bước tiếp; dừng khi trả lời chữ, khi hỏi lại, hoặc quá 10 bước. Không có guard |
| Context | System prompt (7 quy tắc) + hồ sơ người dùng + bảng mốc ngày + schema JSON thô của mọi tool đã cài + history. Dài nhất 9,6K / 16K token |
| Train | LoRA r32 trên Modal; loss chỉ tính ở các bước assistant của lượt cuối; 30% mẫu bị đổi tên app / tool / tham số (Hammer masking) |
| Chấm | Scorer tự viết đúng công thức BTC (oracle = 100); vLLM cấu hình đã kiểm (tắt prefix cache, 8 chuỗi) |
| Bộ sinh dữ liệu | `synth_env.py`: người viết + người giải (đều là Claude Code) + kiểm chéo + lọc luật; đã có 221 mẫu |
| Công cụ chẩn đoán | `errors.py` (lỗi theo chỗ lệch đầu tiên), `phenomena.py` (điểm theo hiện tượng trong câu), `cmp_runs.py` (so 2 model, có khoảng tin cậy) |

Kết quả đã đo **[đo]**:

| Model | Dữ liệu train | Epoch | DEV (209) | Ghi chú |
|---|---|---|---|---|
| Zero-shot | — | — | 27,2 | |
| v1 | 1.127 | 2 | 51,9 | |
| v2 | 1.127 + masking | 2 | 54,1 | template 72,7 / hội thoại 40,2; trên chính mẫu đã học chỉ 78,4 |
| lc50 (mốc thí nghiệm) | 564 | 4 | 49,2 | mẫu đã học 90,0; 563 mẫu gold chưa học, app quen: **67,3** |
| v2e4 | 1.127 + masking | 4 | chưa chấm | Modal bị khoá |
| lc50 + 221 mẫu sinh | 785 | 4 | chưa có | bị ngắt ở 61% |
| Thầy (Claude + quy ước + ví dụ truy hồi) | — | — | 89,3 trên 24 mẫu | hơi lạc quan: quy ước được chỉnh trên chính 24 mẫu này |

Đường cong học **[đo]**: gấp đôi dữ liệu gold (564 → 1.127) cho khoảng +4,9 điểm DEV (lẫn một phần yếu tố số epoch).

---

## 2. Khó khăn của phase 2

### 2.1 Luật chấm khắt khe
- Trạng thái cuối sai thì mẫu 0 điểm. **[đo]** Trên 563 mẫu chưa học, 22,4 điểm mất vì lý do này (126 mẫu).
- Trạng thái đúng vẫn bị trừ nếu danh sách call lệch gold, kể cả call đọc: mất thêm 8,7 điểm (88 mẫu).
- Tham số không phải câu chữ tự do phải giống hệt (ID, giờ ISO, số tiền, enum, từ khoá tìm kiếm). Thừa một tham số tuỳ chọn là call đó không khớp.
- Mẫu đáng rỗng: gọi bất kỳ tool nào là 0 điểm.

### 2.2 Dữ liệu nhỏ so với độ đa dạng
- Họ hội thoại: 391 mẫu trải trên 295 tổ hợp tool khác nhau, 253 tổ hợp chỉ có đúng 1 mẫu **[đo]**.
- Model học thuộc thay vì học kỹ năng **[đo]**: câu hội thoại đã học 83,6 → câu chưa học (app quen) 50,3 → câu chưa học (đa số app lạ) 35,0.
- Hai nút thắt chồng nhau: câu mới (−33) và app mới (−15).
- Các hệ thống cùng cỡ model trong paper dùng 15–50K mẫu **[paper]**; ta có 1,1K.

### 2.3 Bẫy trong câu nói **[đo]** (mốc chung 67,3 trên 563 mẫu chưa học)
| Bẫy | Mẫu TRAIN | Điểm | Ví dụ |
|---|---|---|---|
| Đọc chính tả email / số điện thoại | 23 | 27,5 | "family a còng example chấm test" |
| Phải hỏi lại ngay vì thiếu thông tin | 39 | 35,1 | "tạo lịch Kế hoạch ra mắt thứ bảy này" → model tự bịa 10:00 |
| Tự sửa lời giữa câu | 46 | 38,2 | "xăng E5, à không, xăng 95" |
| Có lời huỷ / sửa nhưng vẫn phải làm một phần | 81 | 50,6 | "xoá hết hẹn giờ, à khoan, chỉ cái buổi chiều" |
| Phương ngữ | 175 | 52,5 | răng, rứa, mô, chừ, nghen |
| ≥ 3 thao tác ghi / ≥ 3 app | 49 / 151 | 54,7 / 57,0 | đặt cháo + trả tiền nước + ghi chú |
| Điều kiện có so sánh số | 65 | 57,0 | "ví mà không đủ thì trả ngân hàng" → chọn sai nhánh |
| Tham chiếu history | 156 | 57,5 | "cái đó", "hồi nãy" |
| Tính toán | 35 | 71,9 | "tiền mười lăm lít xăng" → model ghi 15.000 |
| Huỷ hẳn (đáng rỗng) | 20 | 75,0 | "à thôi khỏi" — model làm được |

Ba dòng đầu chỉ có 13–24 mẫu đo nên con số chỉ mang tính chỉ hướng.

### 2.4 Phân bố lỗi theo chỗ lệch đầu tiên **[đo]** (`errors.py`, 563 mẫu chưa học)
| Họ hội thoại (50,3 điểm) | % phần mất | Họ template (72,8 điểm) | % phần mất |
|---|---|---|---|
| Đọc thừa / đọc sai tool | 15 | Câu chữ tự do lệch gold | 20 |
| Tự đoán thay vì hỏi lại | 13 | Sai chuỗi / enum (vd chọn sai nhánh bank / ví) | 15 |
| Chọn sai ID trong kết quả đọc | 11 | Thiếu tham số | 9 |
| Thiếu tham số | 10 | Thừa tham số tuỳ chọn | 8 |
| Thừa tham số tuỳ chọn | 9 | Đọc thừa / đọc sai tool | 8 |
| Ghi sai tool; đáng rỗng mà gọi | 6; 6 | Hỏi thừa; ghi sai tool; dừng sớm | 5; 5; 5 |

### 2.5 Quy ước ngầm của bộ dữ liệu
Gold tuân theo các thói quen không ghi trong đề; model phải tái tạo đúng:
- ID mới thì đọc trước (99,7% TRAIN, 100% DEV); ID đã có trong history thì không đọc lại **[đo]**.
- Tool đọc kiểu liệt kê gọi `{}` (727 lần), không truyền bộ lọc **[đo]**.
- Trước chuyển khoản: xem số dư + danh sách người nhận (96–97%); trước đặt món: xem menu (86%) **[đo]**.
- Hỏi lại ID chỉ khi lệnh đọc trả ≥ 2 kết quả (65/65); 1 kết quả thì không bao giờ hỏi (255/255) **[đo]**.
- Một số quy ước không nhất quán ngay trong gold (tạo lịch có `end` 13/20 lần; `get_route` có `mode` 6/15 lần) → trần điểm thực tế dưới 100 cho mọi model.

### 2.6 App lạ
- DEV: 70% mẫu dùng app không có trong TRAIN (do ta cố ý giữ lại). Private test chắc chắn có app mới.
- Tham số tuỳ chọn của app lạ không học được từ train (vd `loyalty.get_points.program`).

### 2.7 Ràng buộc thi và nguồn lực
- Model < 5B, context ≤ 16K, ≤ 30 giây / mẫu, offline, kết quả phải tất định.
- Hạn nộp Docker 16/11. Modal đang khoá vì vượt hạn mức chi tiêu.
- Không có API LLM lớn; chỉ có Claude Code (≈ 19K token cho mỗi mẫu sinh giữ lại).

### 2.8 Khó khăn về đo lường
- DEV chỉ 209 mẫu: chênh lệch dưới ~4 điểm giữa hai model nằm trong nhiễu.
- DEV lệch so với public (59% hội thoại so với 44%).
- Điểm public trên bảng xếp hạng không dùng để đánh giá năng lực model.

---

## 3. Xiaomi MiCU làm gì ở từng bước **[paper]** (arXiv 2606.01099)

Bài toán: câu lệnh nhà thông minh → 1 thiết bị + 1 hành động. Một lượt, không gọi tool, không history.

| Bước | Việc làm | Chi tiết |
|---|---|---|
| 1. Dữ liệu dễ | Điền template bằng thông số thiết bị | 30K mẫu; mỗi hành động hợp lệ 1 mẫu; nhãn đúng từ cách xây dựng |
| 2. Dữ liệu khó | Lấy câu thật phức tạp từ log, gán nhãn bằng LLM | 20K mẫu; ~1K mẫu người gán tay kèm lời giải làm kho → DeepSeek-R1 gán phần còn lại, mỗi mẫu được kèm các mẫu tay giống nhất → tự soát lại → sinh nhiều lần rồi bỏ phiếu → người duyệt |
| 3. Tập đánh giá | 9K câu thật, người gán nhãn | Không bao giờ đánh giá trên dữ liệu sinh |
| 4. Train tiếp tổng quát | Văn bản tiếng Trung + văn bản dài | 1 epoch, lr 5e-5 |
| 5. SFT theo thứ tự | Dễ 1 epoch → khó 2 epoch | lr 5e-6, train toàn bộ tham số, 8 GPU H20 |
| 6. Dạy suy luận | Thầy viết lời giải 3 bước (kiểm ý định → lọc theo tên, phòng → hướng dẫn riêng từng loại), được cho biết đáp án; model học sinh `<think>` trước đáp án | 25K lời giải |
| 7. RL (DAPO) | Mỗi câu sinh nhiều đáp án; thưởng đúng thẻ + đúng JSON + đúng nội dung | lr 1e-5 |
| 8. Nén token | Mô tả mỗi loại thiết bị → 1 token đặc biệt | −31,9% token; −0,6 điểm |
| 9. Triển khai | 4 × RTX 4090, lượng tử hoá FP8, giải mã suy đoán | 98 yêu cầu / giây, 173 ms |

Kết quả tách từng phần (hai bảng riêng trong paper, không cộng chéo được):
| Bảng 3 — đổi cách dùng dữ liệu | Điểm | Bảng 4 — thêm dần từng bước | Điểm |
|---|---|---|---|
| Chỉ dữ liệu dễ | 39,3 | Base | 34,1 |
| Chỉ dữ liệu khó | 92,8 | + SFT dễ → khó | 93,7 |
| Trộn dễ + khó | 93,0 | + dạy suy luận | 94,0 |
| Khó → dễ | 66,4 | + RL | 94,6 |
| Dễ → khó | 94,6 | | |

Bài học: (a) dữ liệu sinh từ template gần như vô dụng khi đứng một mình; giá trị nằm ở câu thật + nhãn kỹ; (b) đừng kết thúc train bằng dữ liệu dễ; (c) suy luận và RL chỉ thêm ~1 điểm khi dữ liệu đã tốt; (d) 4B là đủ, 8B / 14B không hơn; dữ liệu bão hoà ở 50–75% của 50K.

---

## 4. Midea HomeFlow làm gì ở từng bước **[paper]** (arXiv 2606.01230)

Bài toán: trợ lý nhà thông minh nhiều lượt, có gọi tool, có hỏi lại, chấm bằng trạng thái nhà. Gần bài của ta nhất.

| Bước | Việc làm | Chi tiết |
|---|---|---|
| 1. Simulator (HomeEnv) | Thiết bị mô tả bằng YAML (thuộc tính, miền giá trị, dịch vụ); sandbox chạy code; bộ chặn lệnh sai / vượt giới hạn; bộ kiểm điều kiện | Agent chỉ thấy kết quả lệnh nó chạy |
| 2. Sinh môi trường | Nhà ngẫu nhiên: thời tiết → bố cục phòng → thiết bị theo phòng → trạng thái ban đầu | "Train trên một môi trường cố định chắc chắn overfit" |
| 3. Hồ sơ người dùng | 20K hồ sơ: tuổi, sức khoẻ, thói quen, sở thích | Có ràng buộc nhất quán |
| 4. Blueprint | GPT-5 viết cảnh sinh hoạt → tách ý định đơn → mỗi ý kèm điều kiện kiểm được trên state | Lọc 3 lớp: chạy được, trong giới hạn schema, LLM kiểm khớp ý định |
| 5. Điều khiển độ khó | Trộn các trục: rõ / mơ hồ / thiếu thông tin; 1 / nhiều / hàng loạt thiết bị; phụ thuộc / độc lập | Loại câu do bộ sinh quyết định |
| 6. Sinh hội thoại | GPT-5 đóng người dùng (bám blueprint) và trợ lý; chạy thật trong sandbox | Chỉ giữ hội thoại đạt mọi điều kiện, không lỗi thực thi |
| 7. Tìm kiếm theo cây | Mỗi quyết định của trợ lý thử 3 hướng; giữ nhánh đạt; dùng lại phần đầu đã đúng | Tỉ lệ sinh thành công 86% → 92% → 96% |
| 8. SFT | 15K hội thoại đã kiểm | 30,9 → 81,5 |
| 9. RL trong simulator | DeepSeek đóng người dùng; thưởng từng bước theo tỉ lệ điều kiện đạt; phạt lệnh bị chặn, code dài, quá 3 lượt; ràng buộc KL về bản SFT | +3,1; 96 GPU × 95 giờ |

Flow lúc chạy:
- **Context:** danh sách thiết bị theo phòng, bộ nhớ thói quen, bối cảnh, history.
- **Tool:** `pyexec` (model viết Python điều khiển thiết bị), đọc trạng thái, đọc thông số, máy tính, thời tiết; trả lời / hỏi lại bằng lời.
- **Quy trình trong system prompt:** nhận dạng ý định → đọc thông số và trạng thái trước khi làm → chỉ hỏi lại khi có nhiều cách hiểu hợp lệ → thực thi → kiểm kết quả → trả lời ngắn.
- **Vòng lặp:** một agent, tối đa 3 lượt, trung bình ~2 lần gọi tool.

Kết quả 4B: role-play không kiểm 75,6 → có blueprint 80,2 → thêm tìm kiếm cây 81,5 → RL 84,6. Nhóm kết hợp / có điều kiện vẫn thấp nhất: 55,6 → 64,7 → 72,2.

Bài học: (a) nhãn phải được chạy thật và kiểm bằng state (+5,9 so với role-play); (b) môi trường phải đa dạng; (c) loại câu và độ khó do bộ sinh điều khiển; (d) RL đắt mà lợi nhỏ; (e) câu có điều kiện vẫn khó ngay cả khi đủ dữ liệu.

---

## 5. Phương án kết hợp cho ta, theo từng tầng

Nguyên tắc chung: mỗi thay đổi đi qua một phép đo trên mẫu gold chưa dùng; không thay hai thứ trong một lần train.

### 5.1 Dữ liệu gold
- Dùng hết 1.545 mẫu cho bản cuối (TRAIN + DEV), sau khi công thức đã chốt bằng thí nghiệm trên phần giữ lại.
- Không dùng public test, không dùng `best_score.jsonl`.

### 5.2 Dữ liệu sinh (trọng tâm) — ghép Midea (kiểm bằng simulator) với Xiaomi (kỷ luật gán nhãn)
| Điểm thiết kế | Học từ | Hiện có | Cần làm tiếp |
|---|---|---|---|
| Môi trường thật, nhãn chạy thật | Midea 1, 6 | Sinh trên state của mẫu TRAIN, nhãn = kết quả chạy trong `zaic_tools` | — |
| Hai lần suy ra đáp án độc lập | Xiaomi bỏ phiếu; Midea blueprint | Người viết (thấy state) và người giải (không thấy) phải trùng state + trùng call; giữ 88% | — |
| Lọc bằng luật rút từ gold | Midea bộ chặn | Loại call bị từ chối, tool chưa cài, ID không có nguồn, tool liệt kê có tham số, sai loại, sai số thao tác ghi | Thêm luật khi phát hiện lệch quy ước mới |
| Loại câu do code gán | Midea 5 | Tỉ lệ làm / hỏi / rỗng và số thao tác ghi theo gold | **Thêm loại theo bẫy đã đo:** tự sửa lời, đọc chính tả, tính toán, ≥ 3 app, ≥ 3 thao tác ghi, câu dài |
| Tỉ lệ loại câu | — | Theo gold | Giữ theo gold; chỉ tăng tối đa 2× cho bẫy có điểm < 50, và theo dõi lỗi ngược (vd hỏi thừa khi thêm mẫu hỏi lại) |
| Kho ví dụ cho thầy | Xiaomi ~1K mẫu tay | 3 trajectory gold gần nhất + sổ quy ước | — |
| Kiểm định thầy trên gold | Xiaomi tập đánh giá thật | 89,3 trên 24 mẫu DEV (lạc quan) | Chấm thầy trên 60 mẫu mới từ nửa TRAIN chưa dùng, theo loại; loại nào < 85 thì chưa sinh loại đó |
| Người duyệt | Xiaomi bước cuối | Tôi đọc tay 8 mẫu | Bạn đọc 30–50 mẫu mỗi lô (~15 phút); đây là kiểm tra độc lập duy nhất về quy ước |
| Đa dạng môi trường | Midea 2 | 1.127 state TRAIN + đổi tên app | Bản cuối sinh trên cả state DEV; app mới: xem 5.8 |
| Quy mô | Xiaomi 20K khó; Midea 15K | 221 | Tuỳ kết quả thí nghiệm; mỗi 1.000 mẫu ≈ 19 triệu token Claude |

Phân bố lỗi dùng thế nào: `errors.py` và `phenomena.py` chạy trên mẫu gold chưa dùng để sinh → cho biết loại nào thiếu → thêm loại đó vào bộ sinh với tỉ lệ có trần → sinh → qua cổng → train → đo lại. Tập dùng để chẩn đoán và tập dùng để xác nhận là hai tập khác nhau (xem 5.9).

### 5.3 Train / fine-tune
| Quyết định | Căn cứ | Trạng thái |
|---|---|---|
| 4 epoch thay vì 2 | **[đo]** điểm trên mẫu đã học 76,7 → 90,0 | Chờ điểm DEV của v2e4 |
| Đổi tên (masking) lấy mẫu lại mỗi epoch | **[đo]** 30% mẫu hiện chưa từng được học ở tên gốc | Cần sửa `sft_data.py` |
| Thứ tự: mẫu sinh trước, gold sau | **[paper]** Xiaomi: khó → dễ 66,4; dễ → khó 94,6 so với trộn 93,0 | Làm khi có đủ mẫu sinh; không kết thúc bằng mẫu sinh |
| Giữ LoRA r32 | **[paper]** các hệ LoRA đạt kết quả tốt; Xiaomi train toàn bộ | A/B sau nếu còn ngân sách |
| Không RL | **[paper]** Xiaomi +0,66; Midea +3,1 với 9.000 giờ GPU; AWPO: GRPO sau SFT làm giảm multi-turn | Bỏ |
| Không dạy suy luận dài | **[paper]** Xiaomi +0,27; vượt ngân sách 30 giây | Bỏ |
| Bản cuối | TRAIN + DEV + mẫu sinh, công thức đã chốt | Chạy FINAL_TEST đúng 1 lần |

### 5.4 Context
| Thay đổi | Căn cứ | Chi phí | Cách kiểm |
|---|---|---|---|
| Thêm `kind` (đọc / ghi) và `returns` của mỗi tool vào schema | Catalog có sẵn nhưng prompt đang bỏ; luật "ID mới thì đọc tool sinh ra nó" đúng 100% kể cả app lạ **[đo]** | Train lại | A/B trên nửa TRAIN |
| Đưa sổ quy ước (mục 2.5) vào system prompt | Thầy tăng 77 → 89 khi thêm 3 quy ước **[đo]**; với model 4B là **[giả thuyết]** | Train lại, +300 token | A/B |
| Đưa 2–3 trajectory gold gần nhất vào prompt, cả lúc train lẫn lúc chạy | Thầy và đội thắng AFAC dùng; chỉ thêm lúc chạy thì hại model đã SFT **[paper]** | Train lại, +600–900 token, thêm bước truy hồi | A/B; kiểm 16K và thời gian |
| Nén token kiểu Xiaomi | Prompt mới 9,6K / 16K | — | Không làm |

### 5.5 Tool
- Bộ tool do BTC cố định; ta không thêm được tool.
- `validate` trước `call`: v2 chỉ có 10 / 446 call bị từ chối **[đo]** → lợi ích tối đa ~0,5 điểm; để sau.
- Hành động viết bằng code kiểu Midea hợp với lỗi điều kiện / lặp / tính toán, nhưng phải viết lại toàn bộ dữ liệu và vòng lặp, bằng chứng ở 4B còn trái chiều, và chính Midea vẫn chỉ 72 ở nhóm này → không làm trong 4 ngày còn lại.

### 5.6 Vòng lặp (flow)
- Giữ ReAct 1 call / bước (giống hình dạng của Midea).
- Model đã tuân thủ luật "ID phải có nguồn" (2 vi phạm / 143 lần ghi **[đo]**) → guard cho luật này không có lợi.
- Guard chỉ thêm khi luật đúng ~100% trên gold và có số đo cho thấy model đang vi phạm. Ứng viên cần đo: chặn gọi lại cùng một lệnh đọc với cùng tham số; giới hạn 8 bước.

### 5.7 Harness
- Chấm: vLLM cấu hình chuẩn; mỗi so sánh dùng `cmp_runs.py` (hiệu ± khoảng tin cậy 95%, số mẫu thắng / thua).
- Sinh dữ liệu: `synth_env.py` + `teacher_env.py`; mỗi lô ghi tỉ lệ giữ, lý do loại, so thống kê với gold.
- Chưa đo: thời gian mỗi mẫu khi chạy từng mẫu một (giới hạn 30 giây), độ ổn định giữa các lần chạy.
- GPU: Modal đang khoá.

### 5.8 App lạ
- Đã có: đổi tên 30% mẫu; quy ước đọc-trước-ghi suy được từ schema.
- Sẽ thử: `kind` + `returns` trong schema (5.4).
- Vùng xám, cần bạn quyết: bộ `zaic_tools` của public có code và catalog của 11 app không có trong train. Sinh dữ liệu trên các app đó không dùng mẫu public nào, nhưng vẫn là dùng tài nguyên của tập public. Mặc định: không làm.

### 5.9 Đo lường và chống overfit
| Tập | Vai trò | Quy tắc |
|---|---|---|
| Nửa TRAIN "q50" (564) | Nguồn sinh dữ liệu và ví dụ cho thầy trong thí nghiệm | — |
| Nửa TRAIN còn lại (563) | Chẩn đoán lỗi + đo hiệu quả mẫu sinh | Chỉ hợp lệ với model train trên q50 |
| DEV sạch (185) | Xác nhận; đo app lạ | Không dùng để sinh, không dùng để chỉnh quy ước |
| FINAL_TEST (209) | Điểm cuối | Chạy 1 lần |

- Mẫu sinh không dùng 7 app giữ lại cho tới bản cuối.
- Tỉ lệ loại câu bám gold; không bám danh sách lỗi DEV.
- Mỗi lô sinh qua 3 cổng: thầy đúng trên gold (nhãn) → thống kê khớp gold (đầu vào) → model train thêm lô đó tăng điểm trên mẫu gold chưa dùng (hiệu quả).

---

## 6. Lộ trình (hạn Docker 16/11)

| Bước | Việc | Điều kiện | Chi phí |
|---|---|---|---|
| 0 | Mở lại Modal | Bạn tăng hạn mức | — |
| 1 | Chấm v2e4 trên DEV; train lại và chấm "nửa TRAIN + 221 mẫu sinh" | — | ~$7 |
| 2a | Nếu mẫu sinh có lợi: thêm loại câu theo bẫy; sinh trên toàn TRAIN (~1.000–2.000 mẫu); bạn duyệt 30–50 mẫu | Bước 1 dương | 19–38 triệu token Claude |
| 2b | Nếu không có lợi: dừng sinh; bản cuối = gold, 4 epoch | Bước 1 âm | — |
| 3 | Một A/B context: `kind` + `returns` + sổ quy ước | Còn thời gian | ~$8 |
| 4 | Train bản cuối trên TRAIN + DEV (+ mẫu sinh); chạy FINAL_TEST 1 lần | Công thức đã chốt | ~$15–25 |

Kỳ vọng thực tế: DEV khoảng 60–70 nếu mẫu sinh tốt gần bằng gold; chưa có căn cứ nào cho mức 90+.

## 7. Điều chưa biết
- Điểm DEV của v2e4 (train kỹ hơn giúp bao nhiêu).
- Mẫu sinh có giúp không, và mỗi mẫu sinh đáng bao nhiêu mẫu gold.
- Thầy đúng bao nhiêu trên mẫu gold mới (ngoài 24 mẫu đã dùng để chỉnh quy ước).
- Trần điểm do gold tự không nhất quán.
- App của private test khác train đến đâu.
- Thời gian mỗi mẫu trên phần cứng của BTC.
