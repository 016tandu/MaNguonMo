# Bài thuyết trình: Dự đoán giá cổ phiếu

## Phần 1: Phân tích dữ liệu và Huấn luyện mô hình (Notebook)

Pipeline tổng quan của quá trình xử lý dữ liệu và huấn luyện mô hình trong notebook `stock-data-analysis-and-model-prediction.ipynb` bao gồm các bước chính sau:

-   **Nguồn dữ liệu**:
    -   Dữ liệu được đọc từ file CSV (`stock_market_june2025.csv`).
    -   Bao gồm các cột như `Date`, `Ticker`, `Open Price`, `Close Price`, `High Price`, `Low Price`, `Volume Traded`, và các chỉ số tài chính khác (`Market Cap`, `PE Ratio`, `Dividend Yield`, `EPS`).

-   **Tiền xử lý và Phân tích dữ liệu (EDA)**:
    -   Kiểu dữ liệu của cột `Date` được chuyển đổi sang định dạng `datetime`.
    -   Kiểm tra và xác nhận không có giá trị thiếu (null/NaN) trong bộ dữ liệu.
    -   Phân tích xu hướng của giá mở cửa (`Open Price`) theo thời gian để xem biến động của thị trường.
    -   Phân tích mối quan hệ giữa khối lượng giao dịch (`Volume Traded`) và giá đóng cửa (`Close Price`). Kết quả cho thấy tương quan yếu (hệ số ~0.06), nghĩa là khối lượng giao dịch lớn không phải lúc nào cũng đi kèm với giá tăng hoặc giảm rõ rệt.
    -   Xác định các ngày có khối lượng giao dịch đột biến (outliers) để hiểu về các sự kiện bất thường trên thị trường.
    -   Phân tích phân phối của các chỉ số tài chính quan trọng:
        -   **PE Ratio**: Tập trung chủ yếu ở khoảng 15-35, cho thấy các công ty có định giá ở mức "lành mạnh".
        -   **Dividend Yield**: Phần lớn cổ phiếu có tỷ suất cổ tức từ 2-5%.
        -   **Market Cap**: Dữ liệu tập trung ở các công ty có vốn hóa thị trường nhỏ và vừa, nhưng cũng có sự hiện diện của các công ty lớn.

-   **Chuẩn bị dữ liệu cho mô hình (Feature Engineering)**:
    -   Các cột định danh như `Ticker` được chuyển đổi thành các biến giả (dummy variables) bằng kỹ thuật One-Hot Encoding để mô hình có thể xử lý.
    -   Các giá trị thiếu (nếu có) được xử lý bằng cách điền giá trị trung bình của cột tương ứng.
    -   Dữ liệu được chia thành tập huấn luyện (train) và tập kiểm thử (test) theo tỷ lệ 80:20.

-   **Huấn luyện mô hình (Training)**:
    -   Hai mô hình thuật toán học máy được sử dụng để dự đoán `Close Price`:
        1.  **Random Forest Regressor**: Một mô hình học máy mạnh mẽ dựa trên cây quyết định, có khả năng xử lý tốt các mối quan hệ phi tuyến tính.
        2.  **XGBoost Regressor**: Một thuật toán Gradient Boosting hiệu suất cao, thường cho kết quả chính xác trong các bài toán dự đoán.

-   **Đánh giá mô hình (Evaluation)**:
    -   Hiệu suất của hai mô hình được đánh giá trên tập kiểm thử bằng các chỉ số:
        -   **Mean Squared Error (MSE)**: Đo lường sai số bình phương trung bình.
        -   **Root Mean Squared Error (RMSE)**: Căn bậc hai của MSE, cho biết độ lệch trung bình của dự đoán.
        -   **R-squared (R2)**: Hệ số xác định, cho biết mức độ biến thiên của dữ liệu được giải thích bởi mô hình.
    -   **Kết quả**:
        -   **Random Forest**: MSE = 6.15, RMSE = 2.48, R2 = 1.00.
        -   **XGBoost**: MSE = 41.90, RMSE = 6.47, R2 = 0.99.
    -   Cả hai mô hình đều cho kết quả rất tốt, nhưng **Random Forest** có độ chính xác cao hơn một chút và được chọn làm mô hình cuối cùng để tích hợp vào backend.

---

## Phần 2: Giải thích Backend

-   **Nền tảng**: Backend được xây dựng bằng **Flask**, một micro-framework của Python, giúp tạo ra các API nhẹ và hiệu quả.

-   **Cấu trúc API**:
    ```ascii
    /
    ├── api/
    │   ├── stock/<ticker>  [GET]
    │   ├── predict/<ticker>  [GET]
    │   ├── info/<ticker>     [GET]
    │   └── search            [GET]
    └── health              [GET]
    ```

-   **Chức năng của từng API Route**:
    -   `GET /api/stock/<ticker>`:
        -   Cung cấp dữ liệu lịch sử (OHLCV - Open, High, Low, Close, Volume) cho một mã cổ phiếu (`ticker`) cụ thể.
        -   Dữ liệu được lấy trong một khoảng thời gian (mặc định là 1 năm).
    -   `GET /api/predict/<ticker>`:
        -   Sử dụng mô hình **Random Forest** đã được huấn luyện để dự đoán giá đóng cửa của cổ phiếu trong N ngày tới (mặc định là 30 ngày).
    -   `GET /api/info/<ticker>`:
        -   Cung cấp thông tin tổng quan về công ty như: Tên, Ngành (Sector), Lĩnh vực (Industry), Vốn hóa thị trường (Market Cap), và các chỉ số tài chính khác.
    -   `GET /api/search`:
        -   Tìm kiếm và gợi ý các mã cổ phiếu dựa trên một truy vấn (`q`).
    -   `GET /health`:
        -   Kiểm tra trạng thái hoạt động của backend.

-   **Dịch vụ API bên ngoài**:
    -   Backend tích hợp và sử dụng dữ liệu từ nhiều nhà cung cấp dịch vụ tài chính để đảm bảo tính đầy đủ và chính xác:
        1.  **Alpha Vantage**: Cung cấp thông tin cơ bản về công ty (tên, ngành, lĩnh vực, vốn hóa).
        2.  **Polygon.io**: Cung cấp dữ liệu lịch sử giá cổ phiếu (OHLCV) và thông tin chi tiết về ticker.
        3.  **Finnhub**: Được sử dụng như một nguồn dữ liệu dự phòng để lấy thông tin hồ sơ công ty, tăng cường sự ổn định cho hệ thống.
    -   Hệ thống được thiết kế với cơ chế **fallback**: nếu một API không trả về dữ liệu hoặc dữ liệu không đầy đủ, backend sẽ tự động gọi đến API tiếp theo để lấp đầy thông tin còn thiếu.

---

## Phần 3: Giải thích Frontend

-   **Nền tảng**: Giao diện người dùng được xây dựng bằng **React** với **TypeScript**, giúp tạo ra một ứng dụng web tương tác, an toàn về kiểu dữ liệu và dễ bảo trì.
-   **Bundler**: Dự án sử dụng **Vite** làm công cụ build và server phát triển. Vite cung cấp tốc độ khởi động nhanh và Hot Module Replacement (HMR) hiệu quả.
-   **Các kỹ thuật React được sử dụng**:
    -   **React Hooks**:
        -   `useState`: Để quản lý trạng thái cục bộ của component (ví dụ: mã ticker, dữ liệu cổ phiếu, trạng thái loading/error).
        -   `useEffect`: Để thực hiện các side effects như gọi API khi component được render hoặc khi state thay đổi.
        -   `useMemo`: Để tối ưu hóa hiệu suất bằng cách ghi nhớ (memoize) kết quả của các phép tính phức tạp, tránh tính toán lại không cần thiết (ví dụ: khi xử lý dữ liệu cho biểu đồ).
        -   `useRef`: Để tạo một bộ đệm (cache) cho kết quả tìm kiếm, giúp giảm số lần gọi API không cần thiết khi người dùng tìm kiếm cùng một mã nhiều lần.
    -   **Thư viện bên ngoài**:
        -   `axios`: Để thực hiện các yêu cầu HTTP đến backend API một cách dễ dàng.
        -   `recharts`: Để vẽ các biểu đồ đường (line chart) trực quan hóa dữ liệu giá cổ phiếu lịch sử và dự đoán.
        -   `lucide-react`: Cung cấp bộ icon SVG nhẹ và đẹp mắt.
-   **Styling**:
    -   CSS được viết thuần trong file `App.css` và `index.css`, sử dụng các kỹ thuật CSS hiện đại như Flexbox và Grid để tạo layout linh hoạt và responsive.

---

## Phần 4: Quá trình Deployment

Quá trình triển khai ứng dụng được tự động hóa thông qua quy trình CI/CD (Continuous Integration/Continuous Deployment), thể hiện qua sơ đồ bên dưới.

```mermaid
graph TD
    subgraph Local Development
        A[Local Codebase on Git]
    end

    subgraph Deployment Pipeline
        A -- Git Push --> B{GitHub Repository}
        B -- Trigger Deploy --> C[Netlify]
        B -- Trigger Deploy --> D[Render]
    end

    subgraph Live Application
        C -- Builds & Deploys --> E[Frontend Application]
        D -- Builds & Deploys --> F[Backend API]
        G[User's Browser] -- Accesses --> E
        E -- Fetches Data --> F
        F -- Gets Financial Data --> H{External APIs\n(Polygon, Alpha Vantage, Finnhub)}
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#00c4b3,stroke:#333,stroke-width:2px
    style D fill:#46e3b7,stroke:#333,stroke-width:2px
    style E fill:#add,stroke:#333,stroke-width:2px
    style F fill:#dda,stroke:#333,stroke-width:2px
    style G fill:#f96,stroke:#333,stroke-width:2px
    style H fill:#ff9,stroke:#333,stroke-width:2px
```

-   **Giải thích sơ đồ**:
    1.  **Local Codebase**: Lập trình viên phát triển code trên máy tính cá nhân và đẩy lên kho chứa trên **GitHub**.
    2.  **GitHub Repository**: Đóng vai trò trung tâm, lưu trữ mã nguồn của cả frontend và backend.
    3.  **CI/CD Trigger**: Mỗi khi có một `git push` mới lên nhánh chính, GitHub sẽ tự động kích hoạt (trigger) quy trình triển khai trên **Netlify** (cho frontend) và **Render** (cho backend).
    4.  **Netlify (Frontend)**: Tự động kéo mã nguồn mới nhất từ GitHub, build dự án React thành các file tĩnh (HTML, CSS, JS), và triển khai lên mạng lưới CDN toàn cầu của họ.
    5.  **Render (Backend)**: Tự động kéo mã nguồn mới nhất, cài đặt các dependencies của Python, và khởi chạy ứng dụng Flask. Render cũng quản lý các biến môi trường (environment variables) chứa API keys một cách an toàn.
    6.  **Tương tác người dùng**:
        -   Người dùng truy cập vào trang web được host trên Netlify.
        -   Ứng dụng frontend (React) chạy trên trình duyệt sẽ gọi đến các API của backend được host trên Render để lấy dữ liệu.
        -   Backend xử lý yêu cầu, lấy dữ liệu từ các dịch vụ bên ngoài (Polygon, Alpha Vantage, Finnhub), và trả kết quả về cho frontend để hiển thị.
