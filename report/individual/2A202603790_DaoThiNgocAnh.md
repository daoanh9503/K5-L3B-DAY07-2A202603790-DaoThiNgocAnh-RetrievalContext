# Báo cáo Day 7 — Đào Thị Ngọc Anh · 2A202603790

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** *Introduction to Intellectual Property* (OpenStax, giấy phép CC BY 4.0), dùng 38 trang thuộc chương 3–5 (38 trang khác nhau trong khoảng trang 102–186 của bản PDF gốc), lấy lại nguyên từ Day 6. Index `day6_A` có 265 chunk (chia cố định 500 ký tự, overlap 50). Index `day6_B` có 200 chunk (chia theo cấu trúc, tối đa 800 ký tự) và chỉ dùng để tham khảo. Mỗi chunk giữ `id`, `source_id` (`openstax-iip`), `source`, `page` cùng metadata `chapter`, `section`, `page_id`.
- **Vector store, embedding model, phiên bản và số chiều:** Chroma 1.5.9 (`PersistentClient`, cosine), copy nguyên `chroma_db` của Day 6 vào `starter/data/chroma_db`, không tạo lại index. Embedding dùng `intfloat/multilingual-e5-small` (384 chiều) qua sentence-transformers 6.3.0, chạy trên CPU. Query được thêm tiền tố `query: ` và chuẩn hóa vector, giống hệt cách Day 6 embed passage (`passage: `). Chi phí embedding bằng 0.
- **Agent model, cách chạy và phân chia dev/test:** `gpt-4o-mini` (OpenAI Chat Completions, temperature 0, dùng tool calling với một tool `search`). Benchmark có 24 câu: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng, test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. 20 câu có đáp án lấy lại từ benchmark Day 6. Test Day 6 chỉ dùng để đánh giá, không dùng để chỉnh, nên được giữ làm test Day 7. 4 câu thiếu bằng chứng (u01–u04) được viết mới cho Day 7.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: có `gold = {source_id, page, evidence}`, trong đó evidence là trích dẫn nguyên văn và phải nằm trọn trong một chunk đúng nguồn và trang. Kiểm tra bằng `tests/test_fixture.py` và `hit()` của starter.
  - Câu thiếu bằng chứng: chọn câu gần chủ đề, và kiểm tra bằng tìm từ khóa trên toàn bộ `chunks_A.json`, không chỉ trên top-k:
    - u01: utility patent, nằm ngoài chương 3–5;
    - u02: phí đăng ký bản quyền;
    - u03: luật Việt Nam;
    - u04: thời hạn gia hạn nhãn hiệu.
  - Xác nhận u04: tìm bằng regex `renew|10 years|ten years|section 8|section 9|maintenance|expire|indefinite|perpetu` trên cả 265 chunk. Không chunk nào nêu thời hạn gia hạn nhãn hiệu. Câu liên quan duy nhất (`p142-f05`) chỉ nói nhãn hiệu được bảo hộ "in perpetuity so long as they are not abandoned", không trả lời câu hỏi bao nhiêu năm thì phải gia hạn. Các kết quả khớp khác thuộc về thời hạn bản quyền (chương 3) hoặc bí mật kinh doanh (chương 5). Lưu ý: một agent có thể dựa vào câu "perpetuity" để trả lời nửa vời; trên test, agent đã từ chối đúng.
  - Chấm câu trả lời theo BENCHMARK.md: grounded 0/1/2, citation phải có trong context **và** được đọc lại để xác nhận hỗ trợ câu trả lời, abstention do người chấm xác nhận. Gold không bao giờ được đưa vào agent hoặc retriever. Filter `chapter` chỉ dùng khi câu hỏi có sẵn trường `filter`, coi như người dùng cung cấp.

## Retrieval và thử nghiệm

Bốn phương pháp đều nằm trong `src/retrievers.py` và được đánh giá bằng `src/eval_retrieval.py`. Script dùng chung hàm `hit()` và `pack_context()` với baseline của starter, query gốc, không viết lại query:

1. **Dense:** e5-small + Chroma `day6_A`.
2. **BM25:** `rank-bm25` 0.2.2, k1=1.5, b=0.75, tách từ giống `baseline.py`. Có thêm TF-IDF của starter để đối chiếu.
3. **Hybrid RRF:** lấy 20 candidate từ mỗi nhánh dense và BM25, gộp bằng Reciprocal Rank Fusion với k=60. Không cộng trực tiếp score của hai nhánh vì hai thang điểm khác nhau.
4. **Hybrid + rerank:** lấy 20 candidate từ hybrid, xếp lại bằng cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Hit@1 và Evidence Hit@3 được tính trên số câu có đáp án (dev N=15, test N=5). Context Evidence Rate là tỷ lệ câu có evidence nằm trong context sau khi đóng gói top-3 theo budget ký tự. Thời gian là trung vị mỗi query. Retrieval chạy local nên chi phí bằng 0.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense `day6_A`, top-3, filter theo câu, không rerank, 800 ký tự | Dùng lại index Day 6 | dev | 11/15 (73%) | 15/15 (100%) | 11/15 (73%) | 49 ms / $0 |
| Phương pháp 2 | BM25, top-3, filter theo câu, 800 ký tự | Từ khóa pháp lý (UTSA, trademark…) có thể khớp tốt bằng lexical | dev | 9/15 (60%) | 13/15 (87%) | 9/15 (60%) | 2 ms / $0 |
| Phương pháp 3 | Hybrid RRF (20+20 candidate, k=60), top-3, 800 ký tự | Gộp dense và lexical để bù lỗi của từng nhánh | dev | 12/15 (80%) | 14/15 (93%) | 12/15 (80%) | 52 ms / $0 |
| Cải thiện 1 | Dense, top-3, **1800 ký tự** | Evidence ở rank 2–3 bị cắt khỏi context khi budget là 800, nên tăng budget sẽ tăng Context Evidence Rate | dev | 11/15 (73%) | 15/15 (100%) | **15/15 (100%)** | 58 ms / $0 |
| Cải thiện 2 | Hybrid + rerank cross-encoder (20 candidate), top-3, 1800 ký tự | Rerank đẩy evidence lên rank 1 | dev | 11/15 (73%) | 15/15 (100%) | 15/15 (100%) | 603 ms / $0 |
| Cấu hình đã chọn | Dense, top-3, 1800 ký tự | Chốt trên dev, xem `configs/selected_config.json` | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 49 ms / $0 |
| Baseline | Dense, top-3, 800 ký tự | | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 52 ms / $0 |

Số liệu tham khảo thêm:
- TF-IDF của starter trên dev: Hit@1 9/15, Hit@3 12/15, Context 9/15.
- Trên test: BM25 đạt Hit@1 2/5, Hit@3 3/5; Hybrid RRF đạt 4/5 và 4/5; Hybrid + rerank đạt 5/5 và 5/5. Kết quả nằm ở `starter/results/test-retrieval-b{800,1800}/`.

**Hai lỗi trên dev, nguyên nhân và cách thử**

- **Lỗi dev #1 — q09** ("two primary types of marks registered with the USPTO"): baseline từ chối nhầm. Chunk gold `p147-f01` được retrieve ở rank 2–3 trong cả 3 lượt search. Tuy vậy, budget 800 ký tự chỉ chứa được chunk rank 1 (`p157-f03`, không liên quan), nên evidence bị loại khỏi context. Search lại cũng không giúp, vì chunk rank 1 vẫn chiếm hết budget. Thử Cải thiện 1 (budget 1800): agent trả lời đúng, được grade 2.
- **Lỗi dev #2 — q13** ("Is federal registration required to get protection"): cũng là từ chối nhầm. Chunk evidence `p157-f05` đứng rank 3 và bị cắt do budget 800. Câu hỏi còn mơ hồ vì không nói rõ là trademark. Với budget 1800, agent trả lời đúng, được grade 2.
- Lỗi phụ trên dev ở baseline:
  - q04: đúng một phần. Agent chỉ trích dẫn `p116-f04` và bỏ sót `p116-f03`, nên citation không hỗ trợ toàn bộ câu trả lời.
  - q02: đúng một phần, thiếu điều kiện "chỉ khi kiện ở tòa án liên bang".
  - Với budget 1800, cả hai câu đều đạt grade 2.

**Lý do chọn cấu hình**

Tiêu chí được công bố trước khi chạy test, xếp theo thứ tự ưu tiên:
1. grounded answer trung bình trên dev;
2. từ chối nhầm thấp, đồng thời giữ abstention 2/2;
3. citation hợp lệ;
4. độ trễ và chi phí, dùng khi các tiêu chí trên hòa nhau.

Kết quả so sánh:
- Budget 1800 sửa được cả 2 lỗi từ chối nhầm, đổi lại token tăng khoảng 25%.
- Ở budget 1800, dense và hybrid+rerank có Hit@3 và Context ngang nhau (15/15). Hybrid+rerank còn thấp hơn hybrid RRF ở Hit@1 (11 so với 12) và chậm hơn dense khoảng 10 lần (603 ms so với 58 ms).
- Hybrid RRF có Hit@1 cao nhất (12/15), nhưng Hit@3 chỉ 14/15. Agent đọc cả top-3, nên Hit@3 và Context quan trọng hơn Hit@1.

Vì vậy cấu hình được chốt là **dense, top-3, budget 1800, max 3 lượt search**. Thông tin chốt nằm trong `configs/selected_config.json`. File được tạo lúc 18:18 ngày 16/11/2026, trước khi chạy agent trên test (18:20–18:21), và được commit riêng trước commit chứa kết quả test.

**Vì sao dense Day 7 (Hit@1 11/15, Hit@3 15/15) cao hơn Day 6 index A (9/15, 12/15)?** Định nghĩa hit hai bên gần như giống nhau: cùng yêu cầu đúng nguồn, đúng trang và evidence nằm trọn trong một chunk; Day 6 còn chuẩn hóa khoảng trắng và chữ hoa/thường nên dễ tính hit hơn. Mình so từng câu trong `traces_A.jsonl` (Day 6) với `dev-retrieval/traces_dense.json` (Day 7). Top-3 giống hệt nhau ở 12/15 câu. Chênh lệch đến từ 3 câu:
- **q08 và q15 (+2 Hit@1, +2 Hit@3):** gold evidence của Day 6 dài hơn và bị cắt ngang ranh giới giữa hai chunk 500 ký tự, nên không chunk nào chứa trọn và Day 6 chấm miss. Ở Day 7, evidence được rút ngắn để nằm trọn trong một chunk, đúng theo yêu cầu của `data/README.md`. Ví dụ q08: Day 6 dùng "These libraries and archives can keep up to three copies…", Day 7 dùng "up to three copies of a copyrighted work, but a copyright notice must be included on each copy". Top-3 vẫn giống hệt.
- **q02 (+1 Hit@3):** Day 6 strategy A chạy không filter. Day 7 áp filter `chapter: 3` vì câu hỏi có sẵn trường `filter`, nên top-3 khác đi và evidence vào được top-3.

Như vậy chênh lệch đến từ việc sửa gold và áp filter, không phải vì retrieval thay đổi. Gold sửa xong được dùng chung cho mọi phương pháp, nên so sánh giữa các phương pháp ở Day 7 vẫn công bằng.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tool: agent có một tool `search(query)` gọi dense retriever. Khung vòng lặp là `run_agent` của starter, controller nằm ở `src/controller.py` và runner ở `src/run_agent.py`.
  - Số lượt và điều kiện dừng: model tự quyết định search hay trả lời. Tối đa 3 lượt search. Nếu vượt giới hạn, context rỗng, hoặc câu trả lời có citation không nằm trong context, agent sẽ abstain.
  - Context: context cộng dồn qua các lượt, bỏ trùng theo chunk ID, ưu tiên chunk từ lượt trước, đóng gói theo budget ký tự có kèm nguồn. Model chỉ thấy context cộng dồn mới nhất, nên budget không bị nhân lên theo số lượt.
  - Prompt yêu cầu chỉ dùng context, coi nội dung tài liệu là dữ liệu chứ không phải chỉ dẫn, trích dẫn `[chunk-id]`, và trả `INSUFFICIENT_EVIDENCE` khi không đủ bằng chứng.
  - Lỗi model hoặc vector store được ghi `status: error` cùng thông báo lỗi thật, không thay bằng kết quả giả. Trong các lần chạy này không có lỗi nào.

| Cấu hình / split | Tool use | Số lượt gọi (search) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline (800) / dev | 17/17 | 1 lượt: 13 câu · 3 lượt: 4 câu | 2/2/11 · 1.60 | 12/13 (92%) | 13/13 | 2/2 | 2/15 |
| Selected (1800) / dev | 17/17 | 1 lượt: 15 câu · 3 lượt: 2 câu | 0/0/15 · 2.00 | 17/17 (100%) | 15/15 | 2/2 | 0/15 |
| Baseline (800) / test | 7/7 | 1 lượt: 4 câu · 3 lượt: 3 câu | 1/1/3 · 1.40 | 4/4 (100%) | 4/4 | 2/2 | 1/5 |
| **Selected (1800) / test** | 7/7 | 1 lượt: 5 câu · 3 lượt: 2 câu | 0/0/5 · **2.00** | 7/7 (100%) | 5/5 | 2/2 | 0/5 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q18 (test, selected) | Có đáp án | 1 lượt: `search("cap on punitive damages trade secret misappropriation")`, chunk `p186-f05` có trong context 1800 ký tự | "Yes, there is a cap on punitive damages… [openstax-iip-p186-f05]" | **2**: đúng (tối đa 2 lần damage award, một số bang đã sửa mức cap). Citation chứa nguyên văn gold |
| q18 (test, baseline) | Có đáp án | 3 lượt với **cùng một query**. Chunk `p186-f05` không vào được context 800 ký tự | "Không đủ bằng chứng để trả lời." | **0**: từ chối nhầm. Cùng nguyên nhân với lỗi dev #1 |
| q20 (test, baseline) | Có đáp án | 1 lượt: `search("duration of an injunction")`. Context chỉ có `p185-f02`, thiếu `p185-f01` | Mô tả cách tính "head start" [p185-f02] | **1**: đúng một phần. Citation hỗ trợ đúng câu đã nêu, nhưng thiếu ý chính của gold |
| u03 (test, selected) | Thiếu bằng chứng | 3 lượt search về luật Việt Nam, không có chunk nào phù hợp | "Không đủ bằng chứng để trả lời." | **Đúng**: corpus chỉ nói về luật Mỹ; tìm "Vietnam" trên toàn corpus không có kết quả |
| u04 (test, selected) | Thiếu bằng chứng | 3 lượt search về gia hạn nhãn hiệu | "Không đủ bằng chứng để trả lời." | **Đúng**: không thêm khẳng định nào |

**Chấm context 0/1/2** (BENCHMARK §2, 5 câu test có đáp án):

| Cấu hình | 0 = không có evidence | 1 = có nhưng thiếu điều kiện/nguồn | 2 = đủ evidence và nguồn | Ghi chú |
|---|---|---|---|---|
| Baseline (800) | 1 (q18) | 1 (q20) | 3 (q16, q17, q19) | q18: context chỉ có `p186-f01`, nói punitive damages có thể được áp dụng nhưng không có câu về mức cap (`p186-f05`). q20: chỉ có `p185-f02` nói về "head start", thiếu quy tắc chính trong gold (`p185-f01`: thời gian đối thủ tìm ra bí mật bằng cách hợp lệ) |
| Selected (1800) | 0 | 0 | 5 (q16–q20) | q18 có `p186-f05` (không quá 2 lần damage award, một số bang đã sửa cap). q20 có `p185-f01`. Các câu khác có chunk gold ở rank 1 |

Chấm theo context cuối cùng gửi cho model (event `search` cuối), đối chiếu với gold và đáp án chuẩn, không dựa vào câu trả lời của model.

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace và nhãn của agent: `starter/results/agent-{dev,test}-{baseline,selected}/`, mỗi thư mục có `traces.json`, `reviews.json`, `summary.json`, `config.json`, và riêng test có thêm `review_sheet.md`.
  - Trace retrieval: `starter/results/dev-retrieval*/` và `starter/results/test-retrieval-b{800,1800}/`.
  - Thời gian agent: tổng dev là 63,1 s ở baseline và 60,8 s ở selected; tổng test là 33,1 s ở baseline và 28,1 s ở selected; trung vị khoảng 3–5 s mỗi câu.
  - Token (prompt/completion):

    | Lần chạy | Prompt | Completion | Lượt gọi LLM |
    |---|---|---|---|
    | dev baseline | 12.402 | 1.333 | 42 |
    | dev selected | 15.718 | 1.392 | 38 |
    | test baseline | 6.166 | 624 | 20 |
    | test selected | 7.997 | 706 | 18 |

  - Chi phí USD: đơn giá `gpt-4o-mini` là $0,15 / 1M input token và $0,60 / 1M output token (giá chuẩn, tra cứu ngày 16/11/2026 tại trang model của OpenAI: developers.openai.com/api/docs/models/gpt-4o-mini; trang này ghi model đã bị đánh dấu deprecated). Công thức: `prompt × 0,15/1e6 + completion × 0,60/1e6`.

    | Lần chạy | Chi phí | Mỗi câu |
    |---|---|---|
    | dev baseline | $0,00266 | $0,00016 |
    | dev selected | $0,00319 | $0,00019 |
    | test baseline | $0,00130 | $0,00019 |
    | test selected | $0,00162 | $0,00023 |
    | **Tổng 4 lần chạy** | **$0,00877** | |

    Retrieval, embedding và rerank chạy local nên chi phí bằng 0. Số trên là ước tính từ token ghi trong trace, chưa đối chiếu với trang Usage của tài khoản.

- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau:**
    - Trên dev, khi tăng budget từ 800 lên 1800: Context Evidence Rate tăng từ 11/15 lên 15/15, grounded tăng từ 1.60 lên 2.00, từ chối nhầm giảm từ 2/15 xuống 0/15.
    - Trên test, sau khi đã chốt cấu hình, xu hướng lặp lại: grounded tăng từ 1.40 lên 2.00, từ chối nhầm giảm từ 1/5 xuống 0/5.
    - Abstention đúng ở mọi lần chạy (2/2).
  - **Đánh đổi:** budget 1800 tốn thêm khoảng 25–30% prompt token, nhưng số lượt gọi LLM lại ít hơn, vì agent không phải search lại vô ích. Tổng thời gian vì thế còn giảm nhẹ. Hybrid+rerank không tăng chất lượng mà chậm hơn khoảng 10 lần.
  - **Quan sát:**
    - Ở budget 800, dense trên test vẫn đạt Context 5/5 với *query gốc*, nhưng agent vẫn từ chối nhầm q18. Với query gốc, `p186-f05` đứng rank 1. Agent lại tự viết query thành "cap on punitive damages trade secret misappropriation", và với query này `p186-f05` tụt xuống rank 2, nên bị cắt khỏi context (context chỉ còn `p186-f01`).
    - Nói cách khác, chỉ số retrieval đo trên query gốc không đảm bảo agent nhận được evidence.
    - Agent còn lặp lại **đúng cùng một query** ở các lượt search sau, nên những lượt đó không đem thêm thông tin.
  - **Giới hạn:**
    - Benchmark nhỏ (test chỉ có 5 câu có đáp án và 2 câu thiếu bằng chứng), nên mỗi câu tương đương 20% và kết quả 100% trên test chưa đủ để kết luận chắc chắn.
    - Corpus chỉ gồm 38 trang tiếng Anh về luật Mỹ.
    - Hit được tính theo khớp chuỗi evidence trong một chunk, chưa hỗ trợ câu cần evidence từ nhiều đoạn (multi-hop).
    - Toàn bộ nhãn do một người chấm.
    - Không chạy lặp lại nhiều lần để đo độ dao động của model, dù đã đặt temperature 0.
  - **Hướng cải thiện tiếp theo:**
    - Không cho agent search lại đúng query đã dùng (hoặc gợi ý cho model viết lại query khác đi).
    - Thay vì chỉ giữ chunk của lượt đầu, chọn context theo điểm liên quan.
    - Thử dùng index `day6_B`, chia theo câu.

**Cách chạy lại**

```bash
conda create -n day07 python=3.11 -y && conda activate day07
pip install -r requirements.txt
# Copy chroma_db của Day 6 vào starter/data/chroma_db, đặt OPENAI_API_KEY trong .env
python src/export_chroma.py --chroma starter/data/chroma_db --collection day6_A --output starter/data/chunks_A.json
python -m unittest discover -s tests -v
python src/eval_retrieval.py --split dev --budget 800 --output starter/results/dev-retrieval
python src/eval_retrieval.py --split dev --budget 1800 --output starter/results/dev-retrieval-b1800
python src/run_agent.py --split dev --budget 800 --output starter/results/agent-dev-baseline
python src/run_agent.py --split dev --budget 1800 --output starter/results/agent-dev-selected
# Sau khi chốt cấu hình (configs/selected_config.json):
python src/eval_retrieval.py --split test --budget 1800 --output starter/results/test-retrieval-b1800
python src/run_agent.py --split test --budget 800 --output starter/results/agent-test-baseline
python src/run_agent.py --split test --budget 1800 --output starter/results/agent-test-selected
python starter/agent_metrics.py --cases data/agent_cases.json --traces starter/results/agent-test-selected/traces.json --reviews starter/results/agent-test-selected/reviews.json --split test --output starter/results/agent-test-selected/summary.json
```

Trên Windows nên đặt `$env:PYTHONUTF8="1"`, vì dữ liệu và trace có tiếng Việt.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
