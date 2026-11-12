# V3 — Thiết kế hoàn chỉnh (11/11/2026)

Nguồn: đọc toàn văn MiCU (Xiaomi, arXiv 2606.01099) và HomeFlow (Midea, arXiv 2606.01230), đề bài, và chẩn đoán SFT v2 trên DEV.
Nguyên tắc: không sinh dữ liệu khi chưa đo được (a) ta sai ở đâu, (b) dữ liệu sinh ra có giống phân bố thật và nhãn có đúng quy ước gold không.

## 1. Hai paper thực sự nói gì

### MiCU (Xiaomi) — Qwen3-4B, 34.1 → 94.6
| Cách train (bảng 3) | Điểm |
|---|---|
| Chỉ dữ liệu "dễ" (30K, sinh từ template) | **39.3** (base 34.1) |
| Chỉ dữ liệu "khó" (20K câu thật từ log người dùng) | **92.8** |
| Trộn dễ + khó | 93.0 |
| Khó → dễ | 66.4 |
| Dễ → khó | 94.6 |

- **Dữ liệu sinh từ template gần như vô dụng nếu đứng một mình.** Gần như toàn bộ mức tăng đến từ câu THẬT của người dùng, được gán nhãn kỹ.
- Cách gán nhãn câu khó: ~1K mẫu người gán tay làm kho ví dụ → DeepSeek-R1 gán nhãn với ví dụ truy hồi từ kho đó → tự soát lại (SelfRefine) → sinh nhiều lần rồi bỏ phiếu → người duyệt lại.
- Train: full fine-tune, lr 5e-6; dễ 1 epoch rồi khó 2 epoch. CoT + RL: +41 trên base trần, nhưng chỉ +0.9 khi đã có curriculum.
- Đánh giá trên 9K mẫu thật do người gán nhãn, không bao giờ trên dữ liệu sinh. Dữ liệu bão hoà ở 50–75% của 50K.
- Bài của họ là 1 lượt, 1 hành động, không có vòng gọi tool.

### HomeFlow (Midea) — Qwen3-4B, 30.9 → 84.6
| Dữ liệu SFT (cùng 15K mẫu) | Điểm 4B |
|---|---|
| Role-play thuần văn bản (LLM tự lọc) | 75.6 |
| Có Blueprint: đích là điều kiện kiểm được trên state, chạy thật trong simulator | 80.2 |
| Thêm MCTS (rẽ nhánh ở quyết định của agent, k=3) | 81.5 |
| Thêm RL từng bước (96 GPU × 95 giờ) | 84.6 |

- Blueprint = hồ sơ người dùng + state ban đầu + "lát cắt đời sống" + tập điều kiện kiểm được. Lọc 3 lớp: chạy được, trong giới hạn schema, LLM kiểm khớp ý định.
- Teacher (GPT-5) đóng cả vai user lẫn agent trong sandbox. Chỉ giữ trajectory đạt mọi điều kiện.
- Môi trường phải đa dạng: "train trên một môi trường cố định chắc chắn overfit nặng".
- Nhóm khó nhất vẫn là điều khiển kết hợp / có điều kiện: 55.6 → 64.7 → 72.2.
- Họ chỉ chấm state cuối. Ta còn bị chấm danh sách call (kể cả call đọc), nên ta cần thêm một thứ họ không cần: **nhãn đúng quy ước của gold**.

## 2. Ta đang sai ở đâu (SFT v2, DEV 209 mẫu)

| Họ | DEV | Điểm | TRAIN có |
|---|---|---|---|
| Template | 74 | 72.7 | 587 mẫu, 182 bộ khung tool |
| Hội thoại LLM | 123 | 40.2 | **391 mẫu, 295 bộ khung (253 khung chỉ có 1 mẫu)** |

Họ LLM mất 73.5 điểm-mẫu, chia theo chỗ lệch đầu tiên:
| Lỗi | Mất |
|---|---|
| Tự đoán giá trị thay vì hỏi lại (điểm ask của họ LLM: 7.7; TRAIN chỉ có 64 mẫu ask họ LLM) | 15.0 |
| Chọn sai ID trong kết quả đọc (lọc theo điều kiện, tham chiếu) | 11.7 |
| Thiếu tham số tuỳ chọn mà gold có (6/15 là `loyalty.get_points.program`, app chưa gặp) | 10.7 |
| Dừng sớm, thiếu call đọc | 8.4 |
| Gọi tool của app khác | 7.6 |
| Hỏi thừa | 4.3 |
| Còn lại | 15.8 |

Họ template mất 20.2: thừa tham số 3.8, sai tool 3.3, câu chữ lệch gold 2.9, sai `query` 2.3, còn lại rải rác.

Kết luận chẩn đoán:
1. Họ LLM thiếu dữ liệu trầm trọng: 391 mẫu trải trên 295 tổ hợp tool. Khung đã gặp (36.0) cũng không hơn khung mới (41.4), mẫu 1 call cũng chỉ 32 → thiếu ở mức kỹ năng (hỏi lại, chọn thực thể), không phải ở độ dài chuỗi.
2. Sai số / giờ gần như không có (2–3 mẫu) → bỏ ý "chuẩn hoá số, giờ bằng code" ở giai đoạn này.
3. DEV lệch so với public: 59% họ LLM (public 44%). Từ nay báo thêm điểm quy đổi = 0.56 × template + 0.44 × LLM (v2 ≈ 58.4).
4. Giới hạn: nhóm nhỏ (template 74, ask 27), kết luận từng dòng có sai số lớn.

## 3. Áp hai paper vào bài của ta
| Paper | Ta |
|---|---|
| MiCU: dữ liệu "khó" từ câu thật | Họ LLM của gold (391 mẫu) — quá ít, cần hàng nghìn |
| MiCU: dữ liệu "dễ" từ template | Giống ý v3 cũ (code lập kế hoạch + LLM viết câu). Một mình nó ≈ 0 với phân bố thật → không được là trụ chính |
| MiCU: ~1K mẫu tay + truy hồi + bỏ phiếu để gán nhãn | 1127 trajectory gold làm kho ví dụ cho teacher; bỏ phiếu 3 lần giải |
| HomeFlow: Blueprint + chạy thật + chỉ giữ mẫu đạt | Simulator zaic_tools trên môi trường TRAIN; giữ mẫu khi các lần giải độc lập trùng hash và trùng call |
| HomeFlow: môi trường đa dạng | 1127 môi trường TRAIN + đổi tên app/tool (masking) |
| Cả hai: đánh giá chỉ trên dữ liệu thật | DEV / FINAL_TEST chỉ gồm gold |

Điểm khác cốt lõi: ta không có log người dùng, nên câu hỏi phải do LLM viết → rủi ro chính là lệch phân bố (kiểu role-play 75.6 của HomeFlow, hoặc tệ hơn). Vì vậy mọi lô dữ liệu phải qua cổng đo ở mục 4.

## 4. Quy trình V3

### Pha 0 — Đo trước khi làm (~$4, không cần API)
- **0a. Đường cong học:** train 25% / 50% / 100% TRAIN (100% = v2), chấm DEV theo họ. Nếu họ LLM còn dốc lên ở 100% → thêm dữ liệu cùng phân bố sẽ có lợi, và ước được cần bao nhiêu. Nếu phẳng → dữ liệu không phải nút thắt, dừng hướng sinh dữ liệu.
- **0b. Chấm v2 trên 200 mẫu TRAIN:** gần 100 → lỗi DEV là do khái quát; thấp → model chưa học nổi, phải sửa cách train trước.
- **0c. Đào quy ước gold trên TRAIN** (code, miễn phí): read nào đi trước write nào, tham số tuỳ chọn nào hay có, khi nào hỏi. Mỗi luật kèm độ nhất quán; luật < 90% đánh dấu là nhiễu, không ép teacher theo.

### Pha 1 — Kiểm định teacher (cần API LLM lớn, ~$5–10)
Teacher = LLM lớn + sổ quy ước (0c) + 3–5 trajectory gold gần nhất truy hồi từ TRAIN, chạy trong simulator.
- Chấm teacher trên DEV bằng scorer của BTC, theo họ và theo loại (exec / ask / rỗng).
- **Cổng L (nhãn):** chỉ sinh dữ liệu cho những loại mà teacher đạt ≥ 90. Loại nào dưới ngưỡng thì chưa sinh, quay lại sửa sổ quy ước.
- Điểm teacher cũng là trần thực tế của hướng chưng cất.

### Pha 2 — Sinh dữ liệu (lô thử 1K → kiểm → mới nhân lên 5–10K)
Trên môi trường của mẫu TRAIN (không dùng tool của 7 app đang giữ làm "app lạ" trên DEV):
1. Teacher đọc dữ liệu thật trong môi trường (danh bạ, lịch, thiết bị…), viết yêu cầu mới theo phong cách họ LLM, kèm nhãn kỹ năng. Tỉ lệ kỹ năng lấy theo TRAIN gold (ask, rỗng, điều kiện, nhiều thực thể, chọn theo tiêu chí, tham chiếu history), không lấy theo danh sách lỗi DEV.
2. Teacher giải 3 lần độc lập trong simulator. Giữ mẫu khi cả 3 trùng state hash và trùng danh sách call; không có call bị từ chối; qua kiểm "ID phải đến từ kết quả đọc hoặc history".
3. Thêm một phần mẫu kiểu template (kế hoạch do code lập, nhãn đúng từ cách xây dựng) làm tầng "dễ".
- **Cổng I (đầu vào):** so thống kê với gold (độ dài câu, số call, tỉ lệ loại, độ đa dạng khung) và huấn luyện bộ phân loại "thật / sinh" trên câu; AUC phải < 0.75.
- **Cổng U (hiệu quả):** train gold + 1K mẫu sinh, chấm DEV. Phải tăng ở họ được nhắm và không giảm ở họ còn lại. Không đạt → sửa bộ sinh, không nhân lên.
- Đọc tay 50 mẫu ngẫu nhiên mỗi lô (như bước người duyệt của MiCU).

### Pha 3 — Train (công thức MiCU, chỉnh cho ta)
- Giai đoạn A: mẫu sinh kiểu template, 1 epoch. Giai đoạn B: mẫu sinh họ LLM + toàn bộ gold, 2 epoch. Không bao giờ kết thúc bằng dữ liệu sinh thuần.
- Giữ LoRA r32, masking 30%, loss trên bước assistant. Full fine-tune để A/B sau nếu còn ngân sách.
- Không làm RL / CoT: MiCU +0.9, HomeFlow +3.1 với 9.000 giờ GPU.

### Pha 4 — Chốt
Train lại trên TRAIN + DEV + dữ liệu sinh (lúc này được sinh cả trên môi trường DEV và 7 app giữ lại) → chạy FINAL_TEST đúng 1 lần.

## 5. Chống overfit
- Sinh dữ liệu, kho ví dụ của teacher, đào quy ước: chỉ từ TRAIN.
- DEV chỉ để đo; mỗi quyết định ghi lại số lần đã dùng DEV. FINAL_TEST khoá.
- Phân bố kỹ năng của dữ liệu sinh bám TRAIN gold, không bám lỗi DEV.
- 7 app giữ lại không xuất hiện trong dữ liệu sinh cho tới pha 4.
- Mỗi lô phải qua 3 cổng L, I, U trước khi nhân lên.
- Luôn báo điểm theo họ + điểm quy đổi theo tỉ lệ public.

## 6. Thời gian và chi phí (hạn Docker 16/11)
| Ngày | Việc | Chi phí |
|---|---|---|
| 11/11 | Pha 0 | ~$4 GPU |
| 12/11 | Pha 1 + lô thử 1K (cổng I, U) | API + ~$5 GPU |
| 13–14/11 | Nhân lên 5–10K, train V3, chấm DEV | API + $15–25 mỗi lần train |
| 15/11 | Pha 4 | ~$20 GPU |
| 16/11 | Dự phòng | |

Dự phòng: nếu tối 12/11 chưa qua cổng L hoặc U → nộp v2 train lại trên TRAIN + DEV.
Chưa có API: Pha 0 vẫn làm được ngay. Pha 1–2 có thể dùng chính Claude Code làm teacher (chậm hơn, tốn quota của bạn).

## 7. Kết quả Pha 0 (11/11)
| Model | Dữ liệu | Epoch | DEV | DEV template / hội thoại | TRAIN đã học | TRAIN chưa học (cùng phân bố, app quen) |
|---|---|---|---|---|---|---|
| v2 | 1127 | 2 | 54.1 | 72.7 / 40.2 | 78.4 (cả 202 mẫu đều đã học) | — |
| lc50 | 564 | 4 | 49.2 | 69.0 / 35.0 | **90.0** (template 91.7 / hội thoại 83.6) | **61.4** (template 72.7 / hội thoại 33.1) |

1. **Gấp đôi dữ liệu gold: +4.9 DEV** (template +3.7, hội thoại +5.2). Đường cong vẫn đang lên → thêm dữ liệu cùng phân bố còn có lợi. Ngoại suy log-tuyến tính (nếu dữ liệu mới tốt như gold): ×4 ≈ 64, ×8 ≈ 69.
2. **v2 học chưa tới:** 4 epoch đưa điểm trên mẫu đã học từ 76.7 lên 90.0.
3. **Khoảng cách khái quát rất lớn ở họ hội thoại:** 83.6 trên mẫu đã học, nhưng chỉ 33.1 trên mẫu chưa học dù app đã quen. Model đang học thuộc từng mẫu, chưa học được kỹ năng → cần nhiều mẫu đa dạng hơn, không cần thêm epoch.
4. Lưu ý nhiễu: lc50 khác v2 cả số mẫu lẫn số epoch; tập con nhỏ (35 mẫu / nhóm).

## 8. Pilot thầy r1 (Claude Sonnet, 24 mẫu DEV phân tầng, có sổ quy ước + 3 ví dụ gold truy hồi từ TRAIN)
| Nhóm | Thầy r1 | v2 cùng mẫu |
|---|---|---|
| Tổng (24) | 77.0 | 50.4 |
| Hội thoại (14) | 77.6 | 33.1 |
| Hỏi lại (7) | 74.5 | 25.7 |
| Template (8) | 70.2 | 68.3 |
Lỗi của thầy chủ yếu lệch quy ước (thêm bộ lọc cho tool đọc liệt kê; thiếu `end` khi tạo lịch; đọc thừa).
Đã bổ sung quy ước 7b–7d, rút từ thống kê TRAIN (tool đọc liệt kê: 727 lần gọi `{}`; create_event có end +1h: 11/20).
Lưu ý: r2/r3 chạy trên cùng 24 mẫu sau khi chỉnh quy ước → điểm r2/r3 hơi lạc quan; kiểm lại trên mẫu khác trước khi sinh dữ liệu.
Kết quả r2 (sau khi thêm quy ước 7b–7d): **89.3** (hội thoại 90.2, hỏi lại 92.4, template 85.0). Lỗi còn lại: 2 mẫu bỏ qua quy ước `end` +1h, 1 mẫu gọi tên tool không có trong catalog, 1 mẫu ghi `program` theo tên hiển thị, 2 mẫu thiếu 1 call đọc trước khi hỏi lại.
Đồng thuận r1 = r2: 15/24, độ chính xác trong nhóm đồng thuận chỉ 89.8 → lọc bằng đồng thuận KHÔNG bắt được lỗi quy ước lặp lại; phải lọc bằng luật (call ngoài catalog, tool đọc liệt kê có tham số, create_event thiếu end...) thay vì chỉ dựa vào đồng thuận.

## 9. Lô thử sinh dữ liệu bằng Claude Code (11/11)
Quy trình (synth_env.py): người viết (thấy state) viết 3 câu mới / môi trường và chạy đáp án dự định → người giải (không thấy state/đáp án) giải lại → giữ khi trùng state hash + trùng call, qua lọc luật, đúng loại và đúng số thao tác ghi được giao.
- Chỉ dùng môi trường và ví dụ của **train_q50**; cấm 7 app giữ lại; tỉ lệ loại câu và số thao tác ghi do code gán theo gold q50.
- Đợt 1 (24 môi trường): giữ 63/72, nhưng câu ngắn (0.68× câu mẫu) và ít thao tác ghi hơn gold → sửa bộ sinh.
- Đợt 2 (60 môi trường): giữ 158/180; phân bố số thao tác ghi khớp gold (hội thoại 17/62/18/3 so với 16/57/22/5; template 4/52/44/0 so với 3/51/42/4); câu hơi dài hơn gold (117 so với 99).
- Tổng: **221 mẫu** (88% giữ lại), 32 app; chi phí ~4.3M token Claude (≈ 19K token / mẫu giữ lại).
- Chưa phủ: banking, calculator, flights, gold_price, lottery, ride_hailing.
- Hạn chế đã biết: người viết và người giải cùng là Claude, cùng đọc một sổ quy ước → lỗi quy ước hệ thống không bị kiểm chéo bắt. Thước đo khách quan là thí nghiệm dưới.

Thí nghiệm đo (đang chạy): `sft_lc50` (564 gold, 4 epoch) so với `sft_lc50s` (564 gold + 221 mẫu sinh, 4 epoch),
chấm trên **train_h50** (563 mẫu gold chưa dùng để sinh) và **dev_clean** (185 mẫu DEV, bỏ 24 mẫu đã dùng để chỉnh quy ước thầy).

Mốc `sft_lc50` trên train_h50 (563 mẫu gold, app đã quen, câu chưa học): **67.3** (hội thoại 50.3 | template 72.8 | không ID 90.2; hỏi lại 59.5).
Đính chính mục 2: với mẫu lớn hơn, app lạ CÓ ảnh hưởng — họ hội thoại đạt 50.3 khi app đã quen (train_h50) nhưng 35.0 trên DEV (đa số app lạ). Kết luận cũ "app lạ không phải nút thắt" dựa trên 17 mẫu là quá vội. Cả hai đều là nút thắt: câu mới (84 → 50) và app mới (50 → 35).

## 10. Trạng thái 11/11 18:40 — BỊ CHẶN: workspace Modal bị vô hiệu ("workspace ... is disabled")
- Nhiều khả năng đã dùng hết credit miễn phí của tháng (ước đã tiêu ~$25–30).
- `sft_v2e4` (TRAIN đủ, 4 epoch): đã train xong và lưu ở volume, CHƯA chấm được DEV.
- `sft_lc50s` (nửa TRAIN + 221 mẫu sinh): bị ngắt ở bước 122/200, không có gì được lưu → phải train lại (~$4.5).
- Việc còn lại khi có GPU: chấm v2e4 trên DEV ($0.3); train lại lc50s ($4.5); chấm lc50s trên train_h50 + DEV ($1.8).
  Lệnh: `python -m modal run --detach train_modal.py --tag sft_lc50s --splits train_q50 --mask 0.3 --epochs 4 --synth pilot`
        `python -m modal run modal_app.py --split train_h50 --model sft_lc50s --tag sft_lc50s` (và `--split dev`)
        `python cmp_runs.py runs/sft_lc50_train_h50 runs/sft_lc50s_train_h50`

## 11. Hiện tượng gây khó trong dữ liệu (12/11) — `python phenomena.py`
Đo bằng mốc `sft_lc50` trên train_h50 (563 mẫu gold chưa học; mốc chung 67.3). Không dùng DEV.
| Hiện tượng | TRAIN | Điểm | So mốc | Mẫu sinh có? (gold q50 % → mẫu sinh %) |
|---|---|---|---|---|
| Đọc chính tả email / số điện thoại | 23 | 27.5 | −39.8 (13 mẫu) | 1.8 → 0.5 |
| Hỏi lại ngay, không cần đọc gì | 39 | 35.1 | −32.2 (19 mẫu) | 3.5 → 6.8 |
| Tự sửa lời giữa câu (à nhầm, à không, à khoan) | 46 | 38.2 | −29.1 (24 mẫu) | 3.9 → 0.5 |
| Có lời huỷ/sửa nhưng vẫn phải làm một phần | 81 | 50.6 | −16.7 | 7.4 → 8.6 |
| Phương ngữ | 175 | 52.5 | −14.8 | 16.0 → 16.7 |
| ≥ 3 thao tác ghi | 49 | 54.7 | −12.6 | 4.4 → 1.8 |
| Dùng ≥ 3 app | 151 | 57.0 | −10.4 | 12.8 → 3.6 |
| Điều kiện có so sánh số | 65 | 57.0 | −10.3 | 7.1 → 6.3 |
| Tham chiếu history | 156 | 57.5 | −9.8 | 14.5 → 14.5 |
| Câu dài > 150 ký tự | 100 | 57.8 | −9.5 | 9.8 → 1.8 |
| Tự huỷ hẳn (gold rỗng) | 20 | 75.0 | +7.7 (12 mẫu) | — |
Lô thử thiếu: tự sửa lời, đọc chính tả, ≥ 3 app, ≥ 3 thao tác ghi, câu dài, tính toán → bổ sung thành loại câu riêng ở đợt sinh sau.
