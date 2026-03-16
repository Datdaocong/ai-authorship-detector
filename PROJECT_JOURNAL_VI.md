# Nhật Ký Phát Triển Dự Án

AI Authorship Detector

Tài liệu này ghi lại toàn bộ quá trình xây dựng dự án **AI Authorship Detector** từ ý tưởng ban đầu đến khi hoàn thiện.

Mục tiêu của nhật ký này:

* ghi lại các bước phát triển
* hiểu rõ pipeline của hệ thống
* giúp dễ giải thích dự án khi phỏng vấn

---

# 1. Mục tiêu dự án

Mục tiêu của dự án là xây dựng một hệ thống machine learning có thể phân loại một đoạn văn bản là:

* **Human-written (do con người viết)**
* **AI-generated (do AI tạo ra)**

Thay vì sử dụng deep learning phức tạp, dự án tập trung vào:

* xây dựng pipeline NLP rõ ràng
* feature engineering
* mô hình cổ điển nhưng hiệu quả

---

# 2. Thiết kế hệ thống ban đầu

Pipeline của hệ thống:

Dataset
→ Preprocessing
→ Feature Extraction (TF-IDF)
→ Classifier (LinearSVC)
→ Evaluation
→ Demo Application

Lý do chọn cách tiếp cận này:

* dễ giải thích
* training nhanh
* phù hợp với text classification

---

# 3. Dataset sử dụng

Hai dataset chính được sử dụng:

## HC3 Dataset

Dataset gồm các câu trả lời của:

* con người
* ChatGPT

Dataset này được dùng để:

* train baseline model
* thử nghiệm feature

---

## HAPE Dataset

Human-AI Parallel Corpus.

Dataset này giúp:

* tăng diversity của dữ liệu
* cải thiện khả năng generalization

---

# 4. Chuẩn hóa dữ liệu

Các dataset khác nhau có format khác nhau.

Do đó cần chuẩn hóa về schema chung:

```
text
label
source
subdomain
```

Các script xử lý dữ liệu:

```
prepare_hc3.py
prepare_hape.py
```

Các bước xử lý gồm:

* làm sạch text
* chuẩn hóa label
* chuẩn hóa metadata

---

# 5. Merge dataset

Sau khi xử lý từng dataset, chúng được merge bằng script:

```
merge_datasets.py
```

Script này thực hiện:

* kiểm tra schema dataset
* lọc dữ liệu lỗi
* gộp nhiều dataset
* tạo dataset sample để test nhanh

---

# 6. Thử nghiệm feature

Ba loại feature được thử nghiệm:

## Word n-gram

Feature dựa trên từ.

Ưu điểm:

* nắm bắt cấu trúc ngôn ngữ

Nhược điểm:

* phụ thuộc nhiều vào vocabulary

---

## Character n-gram

Feature dựa trên ký tự.

Ưu điểm:

* bắt được style viết
* ít phụ thuộc từ vựng

Nhược điểm:

* số lượng feature lớn

---

## Hybrid

Kết hợp word + char features.

---

# 7. Lựa chọn model

Model được sử dụng:

```
LinearSVC
```

Lý do:

* hiệu quả với dữ liệu sparse
* training nhanh
* phổ biến trong text classification

---

# 8. Đánh giá mô hình

Các metric được sử dụng:

* accuracy
* precision
* recall
* F1-score
* confusion matrix

Kết quả tốt nhất:

Accuracy ≈ **0.91**

---

# 9. Error Analysis

Để hiểu lỗi của model, các trường hợp sai được trích xuất.

Hai loại lỗi chính:

* AI bị dự đoán là human
* Human bị dự đoán là AI

Phân tích giúp phát hiện:

* câu quá ngắn
* văn phong trung tính
* nội dung khó phân biệt

---

# 10. Logging experiment

Mỗi lần training được lưu vào:

```
logs/experiments.csv
```

Thông tin được lưu:

* dataset
* feature type
* training size
* accuracy
* timestamp

Điều này giúp so sánh các experiment.

---

# 11. Demo Application

Ứng dụng demo được xây dựng bằng:

```
Streamlit
```

Chức năng:

* nhập văn bản
* dự đoán AI / Human
* hiển thị confidence
* xem research summary

Chạy app:

```
streamlit run app/app.py
```

---

# 12. Dọn dẹp repository

Trước khi hoàn thiện project, repo được dọn dẹp:

Đã xóa:

* model thử nghiệm
* dataset tạm
* file cache

Giữ lại:

* pipeline training
* dataset scripts
* model cuối cùng
* demo app

---

# 13. Cấu trúc project cuối

```
ai-authorship-detector
│
├── app
├── analysis
├── data
├── logs
├── model
├── training
├── utils
└── README.md
```

---

# 14. Bài học rút ra

Qua dự án này:

* dữ liệu quan trọng hơn model
* character feature rất mạnh với style detection
* error analysis giúp hiểu model
* pipeline rõ ràng rất quan trọng

---

# 15. Hướng phát triển

Các cải tiến có thể làm trong tương lai:

* sử dụng BERT
* dataset lớn hơn
* hỗ trợ nhiều ngôn ngữ
* cải thiện giao diện demo
