# BÀI TẬP LỚN MÔN TRÍ TUỆ NHÂN TẠO VÀ HỌC MÁY

Bước 1: Chuẩn bị trên Google Colab
Truy cập colab.research.google.com.

- Nhấn New Notebook.

- Đổi môi trường chạy (Rất quan trọng): Vào menu Runtime -> Change runtime type -> Chọn T4 GPU. Việc này giúp huấn luyện AI nhanh gấp 10-20 lần so với dùng CPU.

Bước 2: Lấy API Key từ Kaggle (Để tải dữ liệu)
Vì code của bạn tải dữ liệu trực tiếp từ Kaggle, bạn cần file kaggle.json:
- Đăng nhập vào Kaggle.com.
- Vào phần Settings tài khoản của bạn.
- Tìm mục API, nhấn vào Create New Token.
- Một file tên kaggle.json sẽ được tải về máy tính. Hãy giữ file này.
Bước 3: Chạy Code huấn luyện trên Colab
Chạy code trong 1 cell:
import os
import zipfile
import tensorflow as tf
from google.colab import files
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ==========================================
# 1. CẤU HÌNH KAGGLE & TẢI DATASET
# ==========================================
if not os.path.exists('/content/kaggle.json'):
    print("Hãy upload file kaggle.json:")
    uploaded = files.upload()
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

print("Đang tải dataset từ Kaggle...")
!kaggle datasets download -d youssefmohmmed/human-skin-diseases-image

print("Đang giải nén...")
with zipfile.ZipFile('human-skin-diseases-image.zip', 'r') as zip_ref:
    zip_ref.extractall('skin_data')

# ==========================================
# 2. TỰ ĐỘNG DÒ ĐƯỜNG DẪN ẢNH (Sửa lỗi FileNotFoundError)
# ==========================================
def find_data_dir(root_path):
    for root, dirs, files_list in os.walk(root_path):
        # Nếu thư mục có nhiều hơn 3 thư mục con, khả năng cao đó là thư mục chứa các Class bệnh
        if len(dirs) > 3:
            # Kiểm tra xem bên trong các thư mục con có file ảnh không
            sub_dir = os.path.join(root, dirs[0])
            images = [f for f in os.listdir(sub_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 0:
                return root
    return None

data_path = find_data_dir('skin_data')

if data_path:
    print(f"\n✅ Đã tìm thấy thư mục dữ liệu tại: {data_path}")
    print(f"✅ Các loại bệnh: {sorted(os.listdir(data_path))}")
else:
    raise FileNotFoundError("Không tìm thấy thư mục chứa ảnh. Vui lòng kiểm tra lại file zip.")

# ==========================================
# 3. CHUẨN BỊ DATA GENERATOR
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# 4. XÂY DỰNG VÀ HUẤN LUYỆN (TRANSFER LEARNING)
# ==========================================
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Bắt đầu huấn luyện...")
# Train 10 epochs để có độ chính xác ổn định trước khi tải file .h5
model.fit(train_gen, validation_data=val_gen, epochs=10)

# ==========================================
# 5. LƯU VÀ TẢI FILE .H5
# ==========================================
model_filename = 'skin_disease_final_model.h5'
model.save(model_filename)
print(f"\n✔️ Đã lưu mô hình thành công: {model_filename}")

files.download(model_filename)
--> sau đó file skin_disease_final_model.h5 sẽ tự động được tải về

Bước 3: Tạo giao diện và liên kết dataset với giao diện
- Mở Pycharm tạo 1 folder đặt tên "AIhealth", tạo 1 file 'app.py' và thiết lập code có trên file app.py
- di chuyển file skin_disease_final_model.h5 vào folder AIhealth

Bước 4. chạy file app.py với lệnh streamlit run app.py
  
   
