# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature càng thấp thì phản hồi càng ổn định: ở 0.0 model gần như luôn chọn token có xác suất cao nhất, nên gọi lại nhiều lần thường ra cùng một "sự thật" quen thuộc (ví dụ Việt Nam xuất khẩu cà phê/gạo hàng đầu), câu chữ khô và an toàn.
> Tăng lên 0.5 rồi 1.0, model bắt đầu chọn cả những token ít phổ biến hơn nên chủ đề và cách diễn đạt đa dạng hơn, giọng văn tự nhiên, "kể chuyện" hơn.
> Ở 1.5 độ ngẫu nhiên cao đến mức câu văn dễ lan man, dùng từ lạ, thậm chí đưa ra chi tiết thiếu chính xác — tức là temperature đổi tính đa dạng lấy độ tin cậy.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi đặt khoảng 0.2 (trong khoảng 0.0–0.3). Chatbot hỗ trợ khách hàng cần câu trả lời nhất quán, bám đúng chính sách và ít "bịa" — hai khách hỏi cùng một câu phải nhận cùng một thông tin. Không để hẳn 0.0 để câu chữ vẫn tự nhiên, không lặp máy móc; phần sáng tạo không quan trọng bằng độ chính xác trong ngữ cảnh này.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Output mỗi ngày = 10.000 × 3 × 350 = 10,5 triệu token = 10.500 nghìn token.
> GPT-4o: 10.500 × $0.010 = **$105/ngày** (~$3.150/tháng). GPT-4o-mini: 10.500 × $0.0006 = **$6,3/ngày** (~$189/tháng).
> → GPT-4o đắt hơn khoảng **16,7 lần** ($0.010 / $0.0006; giá input cũng chênh đúng 16,7 lần nên tính cả input thì tỉ lệ vẫn vậy).
> GPT-4o xứng đáng khi sai sót đắt giá và cần suy luận sâu: ví dụ phân tích hợp đồng pháp lý, review code phức tạp, tư vấn y tế/tài chính cần độ chính xác cao với lượng request ít.
> Nên dùng mini cho tác vụ đơn giản, khối lượng lớn: phân loại ticket, trả lời FAQ, tóm tắt ngắn, trích xuất thông tin — chất lượng mini đủ tốt mà tiết kiệm ~94% chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, câu trả lời ngắn, câu đơn giản, gần như không có thuật ngữ và dựa vào ví dụ đời thường (kiểu "một cuốn sổ ghi chép mà cả lớp cùng giữ, không ai tẩy xoá được").
> Với persona chuyên gia tài chính, câu trả lời dài và dày đặc thuật ngữ (sổ cái phân tán, hàm băm, cơ chế đồng thuận PoW/PoS, smart contract, tokenization) và lấy ví dụ trong thanh toán xuyên biên giới, DeFi.
> Như vậy cùng một câu hỏi, system prompt quyết định đối tượng người đọc, độ dài, từ vựng và loại ví dụ. System prompt hoạt động như "chỉ thị đạo diễn" đặt ở đầu ngữ cảnh: model điều kiện hoá toàn bộ phần sinh token theo vai trò đó, nên đổi persona là đổi hẳn phong cách mà không cần đổi câu hỏi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tôi đo một đoạn văn tiếng Việt 121 từ (chủ đề AI và mô hình ngôn ngữ lớn): `count_tokens` (tiktoken, encoding `o200k_base` của gpt-4o) cho **147 token**, còn ước lượng 121 / 0.75 cho **161 token** → ước lượng thô cao hơn khoảng **9,5%** (161 so với 147); sai số này đổi theo từng đoạn văn nên không thể tin công thức đếm từ.
> Tiếng Việt tốn token hơn tiếng Anh vì bộ tokenizer BPE được huấn luyện chủ yếu trên văn bản tiếng Anh: từ tiếng Anh phổ biến thường là 1 token, còn chữ tiếng Việt có dấu (ệ, ướ, ổ...) được mã hoá UTF-8 thành nhiều byte và ít xuất hiện trong dữ liệu huấn luyện nên hay bị tách thành nhiều mảnh. Cùng một câu 85 ký tự, tôi đo được: tiếng Anh 15–16 token; tiếng Việt 23 token với `o200k_base` và tới 41 token với `cl100k_base` (encoding cũ của GPT-4/3.5) — gấp ~2,6 lần. Encoding mới đã cải thiện nhiều nhưng tiếng Việt vẫn đắt hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất ở các giao diện tương tác trực tiếp với con người như chatbot, trợ lý viết, trợ lý code, nhất là khi câu trả lời dài: tổng thời gian sinh không đổi, nhưng chữ đầu tiên xuất hiện sau vài trăm mili-giây (time-to-first-token thấp) nên người dùng thấy hệ thống "đang trả lời" thay vì nhìn màn hình trống 10–20 giây, và có thể dừng sớm nếu model đi sai hướng. Ngược lại, non-streaming phù hợp hơn khi không có người chờ đọc từng chữ hoặc cần xử lý trọn vẹn kết quả trước khi dùng: xử lý batch/job nền, gọi API giữa các service, output phải là JSON hợp lệ để parse, cần kiểm duyệt nội dung trước khi hiển thị, hoặc câu trả lời rất ngắn (phân loại, trả về một nhãn) — khi đó streaming chỉ làm code phức tạp hơn mà không lợi gì.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Khi API quá tải, delay cố định vẫn bắn request lại với cùng nhịp nên server không có thời gian hồi phục; exponential backoff (0.1s → 0.2s → 0.4s → ...) giãn khoảng cách nhanh dần, tự động giảm tải khi lỗi kéo dài, trong khi lỗi thoáng qua vẫn được thử lại gần như ngay. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, chúng sẽ lỗi cùng lúc, chờ đúng 1 giây rồi cùng đập lại cùng lúc — tạo ra các đợt sóng request đồng bộ ("thundering herd"/retry storm) khiến server vừa hồi phục đã sập lại, lỗi kéo dài vô hạn. Vì vậy thực tế còn thêm **jitter** (cộng một khoảng ngẫu nhiên vào delay) để phân tán các client, và đặt giới hạn số lần retry / delay tối đa.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: trợ giảng thân thiện của khóa AI. System prompt: **"Bạn là trợ giảng thân thiện của khóa AI thực hành. Luôn trả lời bằng tiếng Việt, ngắn gọn trong 3–5 câu, dùng ví dụ đời thường cho người mới bắt đầu. Nếu không chắc chắn thì nói rõ là không chắc, không bịa thông tin. Khi người học hỏi bài tập, gợi ý hướng làm thay vì đưa đáp án hoàn chỉnh."**
> Lựa chọn từ ngữ quan trọng: (1) "trả lời bằng tiếng Việt" — nếu không chỉ định, model hay chuyển sang tiếng Anh khi gặp thuật ngữ kỹ thuật; (2) "ngắn gọn trong 3–5 câu" — đưa ra con số cụ thể thay vì chỉ nói "ngắn" giúp model tuân thủ đều hơn, đọc dễ trên terminal và giảm output token (loại token đắt gấp 4 lần input); (3) "không bịa thông tin" và "gợi ý thay vì đưa đáp án" — đặt ranh giới hành vi để trợ lý đáng tin và phục vụ đúng mục tiêu học tập.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: history chỉ giữ 3 lượt gần nhất, nên trong phiên dài trợ lý "quên" các thông tin đã nói trước đó (tên người học, bài đang làm), và khi thoát chương trình thì mất sạch — không có bộ nhớ dài hạn.
> Cải thiện đề xuất: **tóm tắt hội thoại cuốn chiếu (summary memory)**. Khi history vượt 6 message, thay vì cắt bỏ, gọi model mini với prompt "Tóm tắt các ý chính và thông tin về người dùng trong đoạn hội thoại sau" để nén các lượt cũ thành 1 đoạn tóm tắt ngắn; lưu nó vào biến `summary` và chèn vào messages dưới dạng một message system thứ hai ("Tóm tắt cuộc trò chuyện trước: ...") ngay sau persona, rồi mới đến 3 lượt gần nhất. Cuối phiên ghi `summary` ra file JSON và nạp lại khi khởi động để có bộ nhớ qua các phiên. Cách này giữ được ngữ cảnh quan trọng mà số input token vẫn gần như cố định; dùng model mini để tóm tắt nên chi phí tăng thêm rất ít.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
