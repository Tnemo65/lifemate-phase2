# Research → Giả thuyết nâng cấp (10/11/2026)

## 1. Hiện trạng (SFT v2, DEV 209 mẫu) — lỗi nằm ở đâu
| Nhóm | n | Điểm |
|---|---|---|
| Họ **template** (ID ngắn, câu mẫu) | 74 | 72.7 (seen 71 / unseen 74) |
| Họ **hội thoại LLM sinh** (history dài, phương ngữ) | 123 | **40.2** (seen 40 / unseen 40) |
| ask_missing_info | 27 | 16.7 |

- **App lạ không phải nút thắt chính:** trong cùng một họ, điểm app đã gặp và app lạ gần bằng nhau. Nút thắt là họ LLM: lọc theo điều kiện trên kết quả tool, làm lặp cho nhiều thực thể ("cả hai", "mấy bài Nhi gửi"), tính toán, tham chiếu history.
- Public cũng có khoảng 45% mẫu dùng app lạ. Ước lượng v2 trên public khoảng 58 điểm, trong khi top SLM đạt 95–99.
- Đã loại: SFT làm model kém hỏi lại. Zero-shot chỉ hỏi được 6.2, SFT được 16.7.

## 2. Bằng chứng (đã đọc bảng kết quả; chi tiết trong báo cáo các agent)
| Hướng | Paper | Số đo |
|---|---|---|
| Suy luận ngắn trước call | IFFC 2608.22472 | Qwen3-4B cùng trọng số: think so với no-think, BFCL Live 74.6 → 86.7 |
| | Qualgo SIGIR'26 (tiếng Việt, 1B) | Trường quyết định có cấu trúc: irrelevance 33 → 53, AST giữ nguyên, ít token hơn CoT 1.8 lần |
| | DiaFORGE 2507.03336 | Bỏ trace (27B): độ chính xác 0.89 → 0.77; Gemma-3-4B học hỏi lại: 0.24 → 0.81 |
| | Reasoning Trap 2510.22977 | ⚠ Bật think làm tăng gọi nhầm tool khi có tool nhiễu (8B: 36 → 57%) |
| Dữ liệu tổng hợp kiểm bằng simulator | HomeFlow 2606.01230 | Qwen3-4B: SFT role-play 75.6 → SFT đã kiểm state 81.5 → RL 84.6 |
| | AgentScaler 2509.13311 | Qwen3-4B: ACEBench 49.5 → 65.9; bản zh 43.9 → 65.6. Lọc bằng khớp state DB + khớp call (gần giống metric của ta) |
| | AskToAct 2503.01940, When2Call 2504.18851 | Dữ liệu ask sinh bằng cách xoá tham số → khái quát được sang API lạ |
| Preference (DPO) sau SFT | When2Call (4B) | Tỉ lệ gọi tool ảo: 16% → 4.3% (SFT) → 1.9% (RPO) |
| | Relign 2412.04141 | Gọi tool ảo 49 → 26 (SFT) → 18 (SFT + DPO) |
| | ETO 2403.02502 | DPO trên cặp thành công/thất bại tốt hơn RFT (67.4 so với 63.6) |
| Validate + retry | Tool-DC 2603.11495 | Bỏ bước Check: −12.5 điểm (3B) |

**Đã loại (có bằng chứng):**
- **Few-shot chỉ thêm lúc inference cho model đã SFT:** DAIL-SQL ghi nhận giảm 4–5 điểm.
- **Constrained decoding:** Qwen3-4B vốn đã hợp lệ 100% format, không có gì để sửa.
- **GRPO thuần sau SFT:**
  - AWPO: Qwen3-4B-2507 multi-turn tụt 48 → 40.
  - Nghiên cứu đối chứng 2609.17848: khi chuyển sang tập khác, GRPO hơn SFT trung bình dưới 1 điểm.
  - Chi phí $20–40 mỗi lần chạy.
- **Bật thinking tự do:** vượt ngân sách 30 giây và tăng gọi nhầm tool.
- **Đổi base sang Hammer / xLAM:** khả năng hỏi lại khi thiếu tham số bằng 0–15%, trong khi Qwen3-4B đạt 78.5%.

Lưu ý khi đọc mục 1:
- Khoảng cách app đã gặp / app lạ trên toàn DEV là do trộn nhóm (hiệu ứng Simpson): mẫu app lạ có 73% thuộc họ LLM, còn mẫu app đã gặp chỉ 27%.
- Nhóm "LLM / app đã gặp" chỉ có 17 mẫu, nên kết luận ở nhóm này còn yếu.
- Con số "khoảng 58 trên public" giả định có transcript chuẩn và chưa tính riêng 11 app public chưa xuất hiện ở đâu.

## 3. Giả thuyết xếp hạng
**H1. Suy luận ngắn ở MỖI bước, viết trước `<tool_call>`** (khoảng 30–60 token mỗi bước)
- **Ví dụ:** "Tập mới nhất dài 42 phút > 30 → mở tập trước: ep_4961."
- **Đích:** họ LLM (40 điểm, 59% DEV). Lỗi của họ này nằm ở bước 2 trở đi, SAU khi đã đọc kết quả tool (lọc điều kiện, lặp cho nhiều thực thể, tính toán), nên kế hoạch viết một lần ở bước đầu sẽ không đủ.
- **Dữ liệu:** LLM lớn viết lời giải thích cho từng bước của 1127 trajectory gold (cách STaR / rationalization), có nhìn thấy kết quả tool thật. Chỉ giữ những lời giải thích mà kết luận khớp với call gold.
- **Code cần sửa:**
  - `append_step` phải lưu cả phần text của model, không lưu content rỗng như hiện nay.
  - Giữ phần suy luận của các bước trước trong history.
  - Đã kiểm: template Qwen3-2507 hiển thị đúng dạng content + tool_call.
- **Kỳ vọng:** +4–8 điểm. ⚠ Đây là suy đoán: bằng chứng chỉ có ở bài toán 1 lượt (IFFC) và ở mô hình 1B cho quyết định từ chối (Qualgo, AST không đổi). Chưa có kết quả nào xác minh cho suy luận nhiều bước trên kết quả tool ở 4B.
- **Rủi ro:** độ trễ (khoảng +1–2 giây mỗi mẫu, trong ngân sách 30 giây); gọi nhầm tool (Reasoning Trap).
- **Chi phí:** API của LLM lớn + $3 SFT + $0.3 eval.

**H2. DPO on-policy sau SFT** (không cần API)
- **Cặp dữ liệu:** (bước gold, bước model làm sai) tại bước đầu tiên lệch. Có cả cặp "hỏi tốt hơn đoán" và "làm tốt hơn hỏi" để không hỏi thừa.
- ⚠ **Bẫy:**
  - v2 đã thuộc gần hết TRAIN (loss khoảng 0.03), nên chạy greedy trên TRAIN sẽ ra rất ít cặp.
  - Cách khắc phục: chạy trên bản TRAIN đã đổi tên bằng `mask.py` với seed mới, rồi đảo ngược tên trước khi replay và chấm.
- **Cổng kiểm:** chạy v2 trên TRAIN (cả bản gốc và bản đổi tên, khoảng $1.5), đếm số mẫu lệch ở bước đầu. Dưới 10% thì không làm.
- **Kỳ vọng:** +2–5 điểm (When2Call, Relign, ETO). Riêng GRPO sau SFT đã có tiền lệ làm giảm điểm (AWPO).
- **Chi phí:** $1.5 + $4.

**H3. Dữ liệu tổng hợp theo DẠNG KHÓ, chạy trên môi trường train có sẵn**
- **Cách làm:**
  - Dùng `ToolEnv("train", id)` với state thật của 1545 mẫu.
  - Viết yêu cầu MỚI tập trung vào các dạng khó: điều kiện, nhiều thực thể, tính toán, tham chiếu history, hỏi lại (xoá tham số / search ra 2 người).
  - LLM lớn giải trong simulator thật, với quy ước của gold ghi trong prompt.
  - Lọc bằng replay và một linter kiểm "read trước write" theo x-ref.
- **Lợi thế so với tạo app mới:** không cần tạo app, rẻ và nhanh hơn, và tránh hẳn vùng xám dùng schema của public.
- **Kỳ vọng:** +5–15 điểm (HomeFlow +6 nhờ kiểm state; AgentScaler-4B +16).
- **Pha sau, nếu cần thêm độ phủ app:** mới tạo app mới.
- **Chi phí:** API + 1–2 ngày công + $3–6 SFT.

**H5 (chưa kiểm, xếp sau). Đưa 2–4 trajectory train giống nhất vào prompt, CẢ lúc train lẫn lúc test**
- **Đích:** 46 mẫu đúng state nhưng F1 thiếu (cách viết ngắn của gold, tham số tuỳ chọn).
- **Bằng chứng:** chỉ có gián tiếp. Thêm lúc inference giúp model chưa SFT (SEER) nhưng hại model đã SFT (DAIL-SQL). Chưa paper nào thử biến thể "train + test" cho tool calling.

**Đã bỏ H4 (validate + retry):** v2 chỉ có 10/446 call bị simulator từ chối (4 lỗi schema), nên không còn gì đáng kể để sửa.

## 4. Thứ tự đề xuất
- **Không cần API:** cổng kiểm H2 ($1.5) → nếu đạt thì DPO ($4).
- **Cần API của LLM lớn:** H3 (hiệu quả kỳ vọng lớn nhất) và H1. Hai hướng này dùng chung phần hạ tầng sinh dữ liệu và kiểm bằng simulator.
- **Quy trình mỗi bước:** kiểm local → chấm DEV bằng cấu hình chuẩn → so từng mẫu bằng `compare.py` → theo dõi tỉ lệ hỏi thừa trên mẫu exec.

---
# ĐỢT 2 — Paper Trung Quốc + lời giải các cuộc thi (10/11/2026): hướng REFACTOR cả hệ thống

## 5. Bằng chứng chính (đã đọc bảng kết quả)
| Công trình | Đơn vị | Model | Điều tạo bước nhảy lớn nhất |
|---|---|---|---|
| MiCU 2606.01099 | Xiaomi + ĐH Vũ Hán | **Qwen3-4B-Instruct** | 50K mẫu, SFT dễ → khó: **+59.6** (34 → 94.6). Thứ tự quan trọng: trộn lẫn 93.0, khó → dễ 66.4. CoT chỉ +0.27, RL chỉ +0.66 |
| HomeFlow 2606.01230 | Midea | Qwen3-4B | Simulator + MCTS sinh 15K trajectory có kiểm state: SFT +45 (30.9 → 81.5); dữ liệu có kiểm so với role-play: +5.9; RL +3.1. **Hành động viết bằng Python** |
| FunReason-MT 2510.24645 | Ant + ĐH Chiết Giang | **Qwen3-4B-Instruct-2507** | 17K mẫu multi-turn tổng hợp: BFCL multi-turn 15.8 → 46.9 (SFT) → 57.8 (RL). **Model công khai, Apache-2.0, phát hành 22/10/2025** |
| EGPO 2508.05118 | Ant | Qwen3-4B-2507 | RL thưởng bằng exact match: multi-turn 15.8 → 56.3. Thêm thưởng định dạng suy luận thì *giảm* điểm |
| ToolACE-MT 2508.12685 | Huawei | 8B LoRA | Bỏ bước tinh chỉnh dữ liệu: 40.3 → 20.9; bỏ bước kiểm: → 32.5 |
| TL-Training 2412.15495 | Phúc Đán | 7B | **1,217 trajectory** (gần bằng ta): bỏ bước sai khỏi loss + tăng trọng số token tên tool → 59.3 → 77.8 |
| Beyond ReAct 2511.10037 | Bắc Hàng + Baidu | Qwen3-4B | Lập cả đồ thị call một lần: SFT 3K mẫu, 4B thắng GPT-4o |
| VehicleWorld 2509.06736 | Phúc Đán | 7B | Điều khiển bằng state/code: 33 → 49 (nhưng Qwen3 gọi tool kiểu thường vẫn tốt) |
| α-UMi 2401.07324 | Alibaba + ĐH Trung Sơn | 7B×3 | Tách planner/caller chỉ có lợi khi có bước train chung trước (không có thì 45 < 53) |
| AFAC2024 hạng 1–2 | Tianchi / Ant | GLM-4-9B / Qwen2-7B | Pipeline quanh 1 model sinh: liên kết thực thể +5.6, chuẩn hoá số +1.0–1.3, few-shot truy hồi, vote nhiều LoRA +0.9–1.1 |
| CPDC 2025 hạng 1–2 | AIcrowd | Qwen3-8B/14B | Chọn lọc context +0.13–0.15 (lớn hơn SFT). SFT trên dữ liệu tổng hợp **không kiểm**: −0.03 |

## 6. Kết luận chung
1. **Bước nhảy lớn luôn đến từ DỮ LIỆU:** 8–50K trajectory tổng hợp có kiểm bằng môi trường (+30–60), không phải từ kỹ thuật train. Ta mới có 1.1K mẫu gold. Dữ liệu không kiểm thì vô ích, thậm chí có hại.
2. **Curriculum:** đừng train khó → dễ (MiCU: 66.4). Dễ → khó chỉ hơn trộn lẫn +1.6 (94.6 so với 93.0).
3. **RL / CoT chỉ thêm +0.3–4 khi SFT đã mạnh.** Để làm sau cùng, hoặc bỏ.
4. **Context engineering lúc inference** (few-shot truy hồi, chuẩn hoá số/giờ bằng code, chọn tool) đóng góp vài điểm rẻ.
5. **Biểu diễn hành động:** code hợp với lỗi của ta (điều kiện, lặp, tính toán), nhưng bằng chứng ở model nhỏ còn trái chiều. Rủi ro cao khi chỉ còn 6 ngày.

## 7. Kiến trúc v3 đề xuất (vừa 6 ngày, hạn Docker 16/11)
**A. Điểm xuất phát: A/B Qwen3-4B-Instruct-2507 với FunReason-MT.**
- Bước 1: so `chat_template` của FunReason-MT với base. `modal_app.py` hiện luôn dùng tokenizer của base, nên template khác sẽ làm hỏng run mà không báo lỗi.
- Bước 2: chấm zero-shot trên DEV ($0.3).
- Bước 3: nếu có triển vọng mới SFT trên TRAIN ($3.5).
- Lưu ý: FunReason-MT được train để viết suy luận trước call, nên `append_step` phải giữ phần text đó trong history.

**B. Bộ máy dữ liệu, sinh NGƯỢC (đòn bẩy chính). Nhãn đúng quy ước gold ngay từ cách xây dựng.**
1. Trên một môi trường TRAIN (giữ nguyên history; tool call trong history đã được áp vào state), lấy mẫu một "kế hoạch" từ các pattern đào được trong gold:
   - chọn write tool trong các app đã cài;
   - dựng chuỗi read → write theo x-ref;
   - chọn thực thể thật bằng cách chạy read trên chính môi trường đó;
   - tính giờ theo current_time;
   - điều kiện: nhánh được quyết định bởi kết quả read thật;
   - mẫu hỏi lại: bỏ một tham số bắt buộc;
   - có cả mẫu rỗng / huỷ / tán gẫu.
2. Chạy kế hoạch trong simulator. Kết quả chạy CHÍNH LÀ nhãn (tool_calls + hash), nên đúng quy ước gold ngay từ đầu.
3. LLM lớn viết câu tiếng Việt diễn đạt đúng kế hoạch đó (phương ngữ, tham chiếu history, "cả hai"...).
4. Lọc hai đường độc lập (nên làm): LLM lớn tự giải câu đó mà không thấy kế hoạch. Chỉ giữ mẫu nếu hash khớp hash kế hoạch và F1 = 1.
5. Đổi tên app/tool (re-skin) để phủ app lạ. Mục tiêu 5–10K mẫu.

Phân vùng dữ liệu:
- Bây giờ chỉ sinh trên môi trường TRAIN.
- Môi trường DEV chỉ dùng khi train lại bản cuối.
- Môi trường FINAL_TEST: không bao giờ.

**C. Train:** thứ tự gold dễ → tổng hợp → gold (không làm khó → dễ). Trộn lẫn gần như ngang, nên đây không phải ưu tiên.

**D. Inference:**
- chuẩn hoá số/giờ tiếng Việt bằng code;
- đưa 2–4 trajectory gần nhất vào prompt (phải có cả lúc train). Cần kiểm giới hạn 16K: hiện prompt dài nhất 9.6K, trong khi mẫu 62 tool có thể vượt;
- (tuỳ chọn) vote 3 adapter theo state hash nếu còn ngân sách thời gian.

**E. Bỏ / để sau:** GRPO, thinking dài, tách nhiều model.

**Bước 0 khi đã có API (khoảng $5):** cho LLM lớn chạy zero-shot trên DEV, với quy ước gold ghi trong prompt. Điểm số và các lỗi quy ước của nó cho biết cách viết prompt sinh câu, và có cần bước lọc 2 đường hay không.

## Chi phí ước tính
| Hạng mục | Chi phí |
|---|---|
| A/B base | ~$4 |
| Mỗi lần chấm DEV | ~$0.3 |
| Train 5–10K mẫu (thời gian tỉ lệ theo token: 5–10 giờ A100) | $13–25 mỗi run; lịch có 2 run → **$30–50** |
| API LLM lớn: 5–10K câu (gấp đôi nếu có lọc 2 đường) | tuỳ provider |

**Phương án dự phòng (quyết cuối ngày 12/11):** nếu dữ liệu tổng hợp đã kiểm chưa ra kịp, nộp v2 (hoặc base FunReason-MT nếu thắng A/B) + chuẩn hoá + few-shot truy hồi, train lại trên TRAIN+DEV.

## 8. Lịch 6 ngày
| Ngày | Việc |
|---|---|
| 10–11/11 | A/B base (A). Dựng bộ máy dữ liệu, sinh thử 500 mẫu, kiểm chất lượng |
| 12–13/11 | Sinh 5–10K mẫu. Train v3 (curriculum). Chấm DEV, so họ LLM / template |
| 14/11 | Few-shot truy hồi / chuẩn hoá (D). Chốt công thức |
| 15/11 | Train lại TRAIN+DEV(+tổng hợp) → FINAL_TEST 1 lần → đóng Docker |
| 16/11 | Dự phòng, nộp |
