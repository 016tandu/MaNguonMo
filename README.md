# Stock Prediction

## Local Run

1. **Backend**
   - Tạo môi trường ảo và cài đặt phụ thuộc:
     ```bash
     cd backend
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```
   - Đặt biến môi trường cần thiết trong `backend/.env` (các khóa API Polygon, Alpha Vantage, ...).
   - Khởi động server:
     ```bash
     python app.py
     ```
     Ứng dụng Flask sẽ lắng nghe tại `http://127.0.0.1:5000`.

2. **Frontend**
   - Cài đặt phụ thuộc:
     ```bash
     cd frontend
     npm install
     ```
   - Chạy Vite dev server:
     ```bash
     npm run dev
     ```
     Khi chạy trên `localhost`, front-end sẽ tự ưu tiên gọi API `http://127.0.0.1:5000`. Đặt biến `VITE_API_BASE_URL` nếu muốn chỉ định endpoint khác.

---

## Tài Liệu Tiếng Việt

### Tiến Trình Huấn Luyện
- Toàn bộ pipeline thu thập dữ liệu, khám phá, và huấn luyện được trình bày trong notebook `backend/stock-data-analysis-and-model-prediction.ipynb`.
- Dữ liệu lịch sử giá được lấy qua Polygon (ưu tiên) và Alpha Vantage để đảm bảo độ phủ mã cổ phiếu. Bộ dữ liệu được chuẩn hóa và bổ sung các đặc trưng cơ bản (giá mở/đóng/cao/thấp, vốn hóa, P/E, cổ tức...).
- Mô hình RandomForestRegressor được huấn luyện với 1.699 đặc trưng, bao gồm cả one-hot ticker. Sau khi huấn luyện, mô hình được đóng gói thành `backend/models/random_forest_model.pkl` và tái sử dụng cho mọi ticker.
- Notebook cũng ghi lại các biểu đồ kiểm chứng, metric huấn luyện, cùng quy trình lựa chọn siêu tham số.

### Kiến Trúc Ứng Dụng Ở Mức Cao
- **Backend (Flask)**:
  - Cung cấp các route `/api/stock/<ticker>`, `/api/predict/<ticker>`, `/api/info/<ticker>`, `/api/search`.
  - Tự động tải mô hình Random Forest, dựng vector đặc trưng phù hợp, dự báo giá theo ngày làm việc và lưu cache để giảm tải API lên nguồn dữ liệu.
  - Bổ sung thông tin doanh nghiệp (market cap, P/E, 52-week high/low, v.v.) thông qua Alpha Vantage và làm giàu dữ liệu trả về front-end.
- **Frontend (React + Vite)**:
  - Giao diện đồ họa hiển thị lịch sử giá, dự báo, thông tin doanh nghiệp, cùng hộp gợi ý mã cổ phiếu.
  - Tự động ưu tiên backend chạy cục bộ khi phát triển (`localhost:5173` → `http://127.0.0.1:5000`), fallback sang endpoint triển khai nếu không tìm thấy backend nội bộ.
  - Sử dụng Recharts để hiển thị biểu đồ, Axios để gọi API và Lucide cho biểu tượng.
- **Triển khai sản phẩm**:
  - Frontend được build và publish trên Netlify.
  - Backend Flask được deploy trên Render (`https://stock-predictor-server.onrender.com`).
  - Cấu hình CORS cho phép cả môi trường cục bộ và production truy cập an toàn.

Những phần trên giúp đội ngũ nắm rõ pipeline huấn luyện, mô hình đang sử dụng và hình dung nhanh kiến trúc hệ thống từ code cục bộ tới môi trường triển khai.
