"""FoodVision AI V2. Run: streamlit run website/app.py (sic_foodvision).
All website flows live here. Checkpoints and datasets are read-only.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(r'D:\AI-SAMSUNG\FoodVisionWork')
FOOD_MODEL = ROOT / 'models/yolo_food/best.pt'
RAW_MODEL = WORK / 'runs/raw_train_30/weights/best.pt'
MEAT_MODEL = WORK / 'runs/efficientnet_meat_v1/best.pt'
FISH_MODEL = WORK / 'runs/efficientnet_fish_v1/best.pt'
NUTRITION_JSON = ROOT / 'modules/yolo_food/food_items.json'
RAW_CLASSES = ('raw_shrimp', 'raw_fish', 'raw_chicken', 'raw_pork', 'raw_beef')
RAW_LABELS = dict(zip(RAW_CLASSES, ('Tôm', 'Cá', 'Thịt gà', 'Thịt lợn', 'Thịt bò')))
# Audit: checkpoint, metadata CSVs and archive README only name freshness
# classes, NOT species. Do not add beef/pork without documented evidence.
VERIFIED_MEAT_CLASSES: frozenset[str] = frozenset()
MEAT_SCOPE_NOTE = 'Chưa xác minh loài thịt trong dữ liệu huấn luyện; tạm khóa phân loại trạng thái thịt.'
FIELDS = ('calories', 'protein', 'carbs', 'fat')


def normalize(name):
    return ' '.join(str(name).lower().replace('_', ' ').replace('-', ' ').split())


@st.cache_data(show_spinner=False)
def load_nutrition_db(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('Bảng dinh dưỡng không đúng định dạng.')
    return data


@st.cache_resource(show_spinner=False)
def load_yolo(path):
    if not Path(path).is_file():
        raise FileNotFoundError('Không tìm thấy checkpoint đã cấu hình. Không tự tải mô hình thay thế.')
    from ultralytics import YOLO
    return YOLO(str(path))


@st.cache_resource(show_spinner=False)
def load_classifier(path):
    if not Path(path).is_file():
        raise FileNotFoundError('Thiếu checkpoint phân loại trạng thái.')
    import torch
    from torchvision import models
    torch.set_num_threads(4)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    expected = ['Fresh', 'NonFresh', 'VeryFresh'] if Path(path) == FISH_MODEL else ['Fresh', 'Half-Fresh', 'Spoiled']
    classes = checkpoint['classes']
    if list(classes) != expected:
        raise ValueError('Danh sách lớp checkpoint khác phạm vi đã kiểm tra.')
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes, checkpoint.get('image_size', 224)


def classify_crop(crop, path):
    import torch
    from torchvision import transforms
    model, classes, size = load_classifier(str(path))
    transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    with torch.inference_mode():
        probs = torch.softmax(model(transform(crop).unsqueeze(0)), dim=1)[0]
    index = int(probs.argmax())
    return classes[index], float(probs[index])


def detect(image, path, confidence):
    model = load_yolo(str(path))
    if Path(path) == RAW_MODEL and tuple(model.names.values()) != RAW_CLASSES:
        raise ValueError('Checkpoint YOLO Raw không khớp 5 lớp đã xác minh.')
    result = model.predict(source=image, conf=confidence, iou=.45, imgsz=416,
                           device='cpu', verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            boxes.append({'label': str(result.names[int(box.cls[0].item())]),
                          'confidence': float(box.conf[0].item()),
                          'bbox': box.xyxy[0].tolist()})
    return Image.fromarray(result.plot()[:, :, ::-1]), boxes


def valid_crop(image, bbox):
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
        return None
    w, h = image.size
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, min(w, math.floor(x1))), max(0, min(h, math.floor(y1)))
    x2, y2 = max(0, min(w, math.ceil(x2))), max(0, min(h, math.ceil(y2)))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return image.crop((x1, y1, x2, y2))


def route_detections(image, boxes, confidence, freshness=True, classifier=None):
    """Only verified labels/crops can reach a classifier. Never classify a full upload."""
    classifier = classifier or classify_crop
    rows = []
    for i, box in enumerate(boxes, 1):
        name, score = box['label'], box['confidence']
        row = {'Đối tượng': i, 'Nguyên liệu': RAW_LABELS.get(name, name),
               'YOLO (%)': round(score * 100, 1), 'Trạng thái': '',
               'Phân loại (%)': None, 'Ghi chú': ''}
        if not math.isfinite(score) or score < confidence:
            row['Ghi chú'] = 'Dưới ngưỡng tin cậy; không phân loại.'
        elif name not in RAW_CLASSES:
            row['Ghi chú'] = 'Ngoài phạm vi 5 lớp; không phân loại.'
        elif not freshness:
            row['Ghi chú'] = 'Chế độ chỉ nhận diện.'
        elif name in ('raw_beef', 'raw_pork', 'raw_chicken') and name not in VERIFIED_MEAT_CLASSES:
            row['Ghi chú'] = MEAT_SCOPE_NOTE
        elif name != 'raw_fish' and name not in VERIFIED_MEAT_CLASSES:
            row['Ghi chú'] = 'Chưa có mô hình trạng thái phù hợp.'
        else:
            crop = valid_crop(image, box['bbox'])
            if crop is None:
                row['Ghi chú'] = 'Vùng ảnh không hợp lệ hoặc quá nhỏ; không phân loại.'
            else:
                path = FISH_MODEL if name == 'raw_fish' else MEAT_MODEL
                try:
                    label, prob = classifier(crop, path)
                    row['Trạng thái'] = label
                    row['Phân loại (%)'] = round(prob * 100, 1)
                    row['Ghi chú'] = 'Phân loại hình ảnh thử nghiệm; không xác nhận an toàn.'
                except Exception as exc:
                    row['Ghi chú'] = f'Không chạy được mô hình trạng thái: {exc}'
        rows.append(row)
    return rows


def analyze_raw(image, confidence, freshness=True):
    annotated, boxes = detect(image, RAW_MODEL, confidence)
    return annotated, boxes, route_detections(image, boxes, confidence, freshness)


def nutrition_values(item, grams):
    values = {field: float(item[field]) * grams / 100 for field in FIELDS}
    if not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError('Dữ liệu dinh dưỡng không hợp lệ.')
    return values


def csv_bytes(rows):
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue().encode('utf-8-sig')


def nutrition_panel(db, foods, key):
    st.subheader('Dinh dưỡng theo khẩu phần', icon=':material/nutrition:')
    st.caption('Bảng tra theo 100 g · Bạn nhập khối lượng thực tế · AI không tự cân thực phẩm')
    rows, totals = [], dict.fromkeys(FIELDS, 0.)
    for food in foods:
        grams = st.number_input(f'{food} — khối lượng (g)', min_value=1, max_value=5000,
                                value=100, step=10, key=f'{key}_grams_{food}')
        try:
            values = nutrition_values(db[food], grams)
        except (KeyError, ValueError, TypeError):
            st.warning(f'Không có dữ liệu hợp lệ cho {food}.')
            continue
        for field in FIELDS:
            totals[field] += values[field]
        rows.append({'Món ăn': food, 'Khối lượng (g)': grams, 'kcal': round(values['calories'], 1),
                     'Protein (g)': round(values['protein'], 2), 'Carbs (g)': round(values['carbs'], 2),
                     'Chất béo (g)': round(values['fat'], 2)})
    cols = st.columns(4)
    for col, field, title, unit in zip(cols, FIELDS, ['Năng lượng', 'Protein', 'Carbohydrate', 'Chất béo'], ['kcal', 'g', 'g', 'g']):
        col.metric(title, f'{totals[field]:.1f} {unit}' if rows else '—', border=True)
    if rows:
        st.dataframe(rows, hide_index=True, width='stretch')
        st.download_button('Xuất bảng CSV', csv_bytes(rows), 'foodvision_nutrition.csv', 'text/csv',
                           key=f'{key}_csv', icon=':material/download:')
    else:
        st.caption('Chọn món có dữ liệu dinh dưỡng để bắt đầu.')


def input_image(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError('Ảnh quá lớn. Vui lòng chọn ảnh dưới 25 megapixel.')
        return ImageOps.exif_transpose(image).convert('RGB')


def render_analysis(kind, db):
    is_food = kind == 'food'
    title = 'Nhận diện món ăn' if is_food else 'Nhận diện nguyên liệu & trạng thái'
    st.header(title)
    st.caption('Từ hình ảnh đến thông tin thực phẩm — kết quả rõ ràng, từng bước minh bạch.')
    left, right = st.columns([.9, 1.3], gap='large')
    with left, st.container(border=True):
        st.subheader('Tải ảnh lên', icon=':material/add_photo_alternate:')
        upload = st.file_uploader('Chọn ảnh thực phẩm', type=['jpg', 'jpeg', 'png', 'webp'], key=f'{kind}_upload')
        raw = upload.getvalue() if upload else None
        st.caption('Ảnh rõ nét, đủ sáng và thấy trọn đối tượng sẽ giúp nhận diện tốt hơn.')
        confidence = st.slider('Ngưỡng tin cậy YOLO', .10, .90, .35, .05, key=f'{kind}_conf')
        freshness = False if is_food else st.toggle('Phân tích trạng thái được hỗ trợ', value=True, key='freshness')
        signature = hashlib.sha256(raw).hexdigest() if raw else None
        token = (signature, confidence, freshness)
        cache_key = f'{kind}_result'
        if st.session_state.get(f'{kind}_token') != token:
            st.session_state.pop(cache_key, None)
            st.session_state[f'{kind}_token'] = token
            if is_food:
                st.session_state['suggested_foods'] = []
        run = st.button('Nhận diện món ăn' if is_food else 'Phân tích nguyên liệu', type='primary',
                        width='stretch', disabled=raw is None, key=f'{kind}_run', icon=':material/auto_awesome:')
        if raw:
            try:
                image = input_image(raw)
                with st.expander('Xem ảnh gốc'):
                    st.image(image, width='stretch', alt='Ảnh đầu vào')
            except Exception as exc:
                st.error(f'Không đọc được ảnh: {exc}')
                run = False
        if run:
            st.session_state.pop(cache_key, None)
            try:
                with st.spinner('Đang nhận diện và kiểm tra phạm vi mô hình…'):
                    if is_food:
                        annotated, boxes = detect(image, FOOD_MODEL, confidence)
                        rows = [{'Món ăn': b['label'], 'YOLO (%)': round(b['confidence'] * 100, 1)} for b in boxes]
                    else:
                        annotated, boxes, rows = analyze_raw(image, confidence, freshness)
                st.session_state[cache_key] = {'image': annotated, 'boxes': boxes, 'rows': rows}
            except Exception as exc:
                st.error(f'Chưa thể phân tích ảnh: {exc}')
    result = st.session_state.get(cache_key)
    with right, st.container(border=True):
        st.subheader('Kết quả nhận diện AI', icon=':material/center_focus_strong:')
        if result:
            st.badge(f"Phát hiện {len(result['boxes'])} đối tượng", color='green' if result['boxes'] else 'orange')
            st.image(result['image'], width='stretch', alt='Ảnh với các khung nhận diện YOLO')
            if result['rows']:
                st.dataframe(result['rows'], hide_index=True, width='stretch')
                if not is_food and freshness:
                    blocked = sum(not row['Trạng thái'] for row in result['rows'])
                    if blocked:
                        st.info(f'{blocked} đối tượng chỉ được nhận diện, không có kết luận trạng thái. Xem lý do trong cột Ghi chú.')
                st.download_button('Tải kết quả CSV', csv_bytes(result['rows']), f'foodvision_{kind}.csv', 'text/csv', key=f'{kind}_export')
            else:
                st.info('Không nhận diện được thực phẩm đủ tin cậy. Không chạy mô hình trạng thái. Hãy thử ảnh rõ hơn.')
        else:
            st.markdown('### Sẵn sàng phân tích')
            st.caption('01  Tải ảnh     →     02  Nhận diện     →     03  Xem kết quả')
            st.info('Kết quả sẽ xuất hiện ở đây sau khi bạn nhấn nút phân tích.')
            st.caption('Hỗ trợ ảnh JPG, PNG và WEBP.')
    if is_food and result:
        lookup = {normalize(k): k for k in db}
        matches = sorted({lookup[normalize(b['label'])] for b in result['boxes'] if normalize(b['label']) in lookup})
        missing = sorted({b['label'] for b in result['boxes'] if normalize(b['label']) not in lookup})
        st.session_state['suggested_foods'] = matches
        with st.container(border=True):
            nutrition_panel(db, matches, 'inline')
            if missing:
                st.caption('Chưa có dữ liệu dinh dưỡng: ' + ', '.join(missing))
    if not is_food:
        st.caption('Ngưỡng tin cậy không bảo đảm từ chối mọi ảnh ngoài phạm vi. Nhãn trạng thái không phải kết luận thực phẩm an toàn để ăn.')


def main():
    st.set_page_config(page_title='FoodVision · Phân tích thực phẩm', page_icon=':material/nutrition:', layout='wide')
    with st.sidebar:
        st.title('FoodVision', anchor=False)
        st.caption('AI nhận diện & dinh dưỡng')
        section = st.radio('KHÔNG GIAN LÀM VIỆC', ['Món ăn', 'Nguyên liệu & trạng thái', 'Tính dinh dưỡng', 'Giới thiệu'], key='section')
    st.caption('FOODVISION / KHÁM PHÁ THỰC PHẨM')
    try:
        db = load_nutrition_db(str(NUTRITION_JSON))
    except Exception as exc:
        db = {}
        st.error(f'Không đọc được bảng dinh dưỡng: {exc}')
    if section == 'Món ăn':
        render_analysis('food', db)
    elif section == 'Nguyên liệu & trạng thái':
        render_analysis('raw', db)
    elif section == 'Tính dinh dưỡng':
        st.header('Bữa ăn của bạn')
        st.caption(f'{len(db)} thực phẩm trong bảng tra · Cộng dinh dưỡng nhiều món trong một khẩu phần')
        suggestions = [f for f in st.session_state.get('suggested_foods', []) if f in db]
        if suggestions and st.button('Dùng món vừa nhận diện', icon=':material/auto_awesome:'):
            st.session_state['meal_foods'] = suggestions
        foods = st.multiselect('Chọn thực phẩm', sorted(db), key='meal_foods')
        nutrition_panel(db, foods, 'meal')
    else:
        st.header('Giới thiệu FoodVision AI V2')
        st.caption('Ứng dụng trí tuệ nhân tạo hỗ trợ nhận diện thực phẩm và tìm hiểu dinh dưỡng')
        st.write(
            'FoodVision AI V2 kết hợp thị giác máy tính với bảng dữ liệu dinh dưỡng để '
            'giúp người dùng tìm hiểu thực phẩm từ hình ảnh. Ứng dụng được xây dựng '
            'trong khuôn khổ đồ án AI của Samsung Innovation Campus, hướng đến một '
            'quy trình dễ sử dụng: tải ảnh, xem dự đoán và tính dinh dưỡng theo khẩu phần.'
        )
        st.subheader('Các chức năng chính')
        cols = st.columns(3)
        features = [
            ('Nhận diện thực phẩm', 'Phát hiện món ăn hoặc nguyên liệu trong ảnh, vẽ khung vị trí và hiển thị điểm tin cậy cho từng đối tượng.'),
            ('Phân tích trạng thái', 'Nhận diện nguyên liệu trước, sau đó phân loại vùng ảnh khi có mô hình phù hợp với phạm vi đã xác minh.'),
            ('Tính dinh dưỡng', 'Chọn nhiều thực phẩm, nhập số gram của từng món và xem tổng năng lượng, protein, carbohydrate, chất béo.')]
        for col, (title, description) in zip(cols, features):
            with col, st.container(border=True):
                st.markdown(f'**{title}**')
                st.write(description)
        st.subheader('Hệ thống hoạt động như thế nào?')
        st.markdown(
            '**Với món ăn:** Ảnh tải lên → YOLO nhận diện → đối chiếu bảng dinh dưỡng '
            '→ nhập khối lượng → tính giá trị khẩu phần.\n\n'
            '**Với nguyên liệu sống:** Ảnh tải lên → YOLO xác định loại và vị trí '
            '→ kiểm tra nguyên liệu được hỗ trợ → cắt từng vùng ảnh hợp lệ '
            '→ phân loại trạng thái nếu phù hợp. Các đối tượng không được hỗ trợ chỉ có kết quả nhận diện.'
        )
        st.subheader('Công nghệ và phạm vi sử dụng')
        st.table([
            {'Thành phần': 'YOLO Food', 'Vai trò và phạm vi': 'Nhận diện món ăn và thực phẩm; bộ dữ liệu Food được cấu hình 214 lớp.'},
            {'Thành phần': 'YOLO Raw', 'Vai trò và phạm vi': 'Nhận diện 5 nhóm nguyên liệu: tôm, cá, thịt gà, thịt lợn và thịt bò.'},
            {'Thành phần': 'EfficientNet Fish', 'Vai trò và phạm vi': 'Phân loại ảnh cá theo Fresh / NonFresh / VeryFresh. Đang sử dụng ở mức thử nghiệm; nhãn toàn ảnh được suy ra từ chú thích mắt và da.'},
            {'Thành phần': 'EfficientNet Meat', 'Vai trò và phạm vi': 'Có mô hình với nhãn Fresh / Half-Fresh / Spoiled. Chưa xác minh loài thịt trong dữ liệu, nên chức năng phân loại trạng thái thịt hiện tạm khóa.'},
            {'Thành phần': 'Bảng dinh dưỡng', 'Vai trò và phạm vi': f'{len(db)} mục thực phẩm, tra cứu theo 100 g; chưa bao phủ toàn bộ lớp nhận diện.'},
            {'Thành phần': 'Giao diện', 'Vai trò và phạm vi': 'Streamlit và Python; xử lý ảnh bằng Pillow, suy luận mô hình bằng Ultralytics và PyTorch.'}])
        st.subheader('Cách tính dinh dưỡng')
        st.write('Khối lượng do bạn nhập, không phải AI tự đo từ ảnh. Giá trị được quy đổi từ bảng tra và cộng lại cho các món đã chọn.')
        st.info('Dinh dưỡng khẩu phần = giá trị trên 100 g × khối lượng nhập (g) / 100')
        st.caption('Ví dụ minh họa: một thực phẩm có 200 kcal/100 g thì khẩu phần 150 g tương ứng 300 kcal. Kết quả thực tế phụ thuộc mục dữ liệu đã chọn và cách chế biến.')
        st.subheader('Hướng dẫn sử dụng')
        st.markdown(
            '1. Chọn **Món ăn** hoặc **Nguyên liệu & trạng thái** ở thanh bên.\n'
            '2. Tải ảnh JPG, PNG hoặc WEBP rõ nét, đủ sáng; điều chỉnh ngưỡng tin cậy nếu cần.\n'
            '3. Nhấn **Nhận diện món ăn** hoặc **Phân tích nguyên liệu**, sau đó xem khung và bảng kết quả.\n'
            '4. Với món có dữ liệu dinh dưỡng, nhập số gram ngay dưới kết quả; hoặc mở **Tính dinh dưỡng** để tự chọn nhiều món.\n'
            '5. Dùng **Tải kết quả CSV** hoặc **Xuất bảng CSV** để lưu kết quả.'
        )
        st.subheader('Hiểu đúng kết quả')
        st.write('Điểm tin cậy cho biết mức độ chắc chắn của mô hình đối với một dự đoán; không phải độ chính xác chung và không phải xác suất thực phẩm an toàn. Tăng ngưỡng có thể loại bớt dự đoán yếu nhưng cũng có thể bỏ sót đối tượng.')
        st.write('Ảnh mờ, thiếu sáng, che khuất hoặc khác dữ liệu huấn luyện có thể gây nhận nhầm. Kể cả confidence cao cũng không bảo đảm ảnh thuộc phạm vi mô hình. Khi đổi ảnh hoặc ngưỡng nhận diện, hãy chạy phân tích lại.')
        st.warning('FoodVision là bản demo học thuật. Nhãn trạng thái phản ánh dấu hiệu hình ảnh theo dữ liệu huấn luyện, không thay thế kiểm tra an toàn thực phẩm. Thông tin dinh dưỡng mang tính tham khảo.')
        st.subheader('Định hướng phát triển')
        st.write('Xác minh phạm vi loài của dữ liệu thịt; mở rộng bảng dinh dưỡng; đánh giá trên ảnh độc lập, ảnh ngoài phạm vi và ảnh nhiều nguyên liệu; cải thiện khả năng từ chối đầu vào không phù hợp.')



if __name__ == '__main__':
    main()
