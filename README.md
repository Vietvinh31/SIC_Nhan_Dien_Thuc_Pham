# FoodVision AI V2

Ứng dụng Streamlit nhận diện món ăn bằng YOLO và tra cứu dinh dưỡng theo khối lượng người dùng nhập.

## Chạy trên máy cá nhân

Yêu cầu Python 3.10 trở lên. Clone repository, tạo môi trường ảo và cài thư viện:

```powershell
git clone <URL_REPOSITORY>
cd FoodVision-AI-V2
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run website/app.py
```

Trên macOS/Linux, lệnh kích hoạt môi trường ảo là `source .venv/bin/activate`.

Ứng dụng dùng checkpoint đã huấn luyện tại `models/yolo_food/best.pt`. Dataset không cần thiết để chạy suy luận nên không được đưa vào repository. Bảng dinh dưỡng nằm ở `modules/yolo_food/food_items.json`.

## Phạm vi của bản GitHub

- Giữ `models/yolo_food/best.pt` để người clone có thể chạy nhận diện món ăn ngay.
- Không đưa thư mục `datasets/`, dữ liệu train, cache, bản sao checkpoint hay kết quả train lên GitHub.
- Các chức năng nguyên liệu sống/trạng thái cần checkpoint riêng đang nằm ngoài repository trên máy huấn luyện; chúng không thuộc bản chạy tối giản này.

## Chạy trên Streamlit Community Cloud

Tạo repository trên GitHub, push code lên, rồi trong Streamlit Community Cloud chọn repository này và đặt **Main file path** là `website/app.py`. Nền tảng sẽ cài thư viện từ `requirements.txt`.

## Đưa dự án lên GitHub lần đầu

Tạo repository trống trên GitHub, không chọn tạo sẵn README hoặc `.gitignore`. Tại thư mục dự án chạy:

```powershell
git init
git add .
git status
git commit -m "Prepare FoodVision app for GitHub"
git branch -M main
git remote add origin https://github.com/<TEN_TAI_KHOAN>/<TEN_REPOSITORY>.git
git push -u origin main
```

Trước khi commit, xem `git status`: thư mục `datasets/` và checkpoint trùng ở `modules/yolo_food/best.pt` không được xuất hiện. Chỉ checkpoint `models/yolo_food/best.pt` được giữ lại.
