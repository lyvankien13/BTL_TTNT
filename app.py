import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# --- CẤU HÌNH ---
MODEL_PATH = 'skin_disease_final_model.h5'
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Mụn trứng cá (Acne)', 'Chàm (Eczema)', 'Ung thư hắc tố (Melanoma)',
    'Vảy nến (Psoriasis)', 'Chứng đỏ mặt (Rosacea)',
    'U nang bã nhờn (Steatocystoma)', 'Nấm da (Tinea)'
]


class SkinApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Chẩn đoán Da liễu AI")
        self.root.geometry("1150x850")
        self.root.configure(bg="#f4f7f6")

        # Load Model
        try:
            if os.path.exists(MODEL_PATH):
                self.model = tf.keras.models.load_model(MODEL_PATH)
            else:
                messagebox.showerror("Lỗi", f"Không tìm thấy file {MODEL_PATH}!")
                self.root.destroy()
        except Exception as e:
            messagebox.showerror("Lỗi Model", f"Lỗi khi tải model: {e}")
            self.root.destroy()

        # --- GIAO DIỆN CHÍNH ---
        # Khung bên trái (Ảnh và Kết quả)
        self.left_panel = tk.Frame(root, bg="#f4f7f6", padx=20)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(self.left_panel, text="NHẬN DIỆN BỆNH DA LIỄU", font=("Segoe UI", 18, "bold"),
                 bg="#f4f7f6", fg="#2c3e50").pack(pady=20)

        self.btn_open = tk.Button(self.left_panel, text="TẢI ẢNH LÊN", command=self.load_image,
                                  font=("Segoe UI", 12, "bold"), bg="#27ae60", fg="white",
                                  width=20, height=2, cursor="hand2", relief="flat")
        self.btn_open.pack(pady=10)

        # Khung chứa ảnh
        self.img_container = tk.Frame(self.left_panel, width=550, height=400, bg="#dfe6e9",
                                      highlightthickness=1, highlightbackground="#b2bec3")
        self.img_container.pack_propagate(False)
        self.img_container.pack(pady=10)

        self.img_label = tk.Label(self.img_container, text="Chưa có hình ảnh", bg="#dfe6e9", font=("Arial", 10))
        self.img_label.pack(fill=tk.BOTH, expand=True)

        # Khung hiển thị kết quả (Dùng Label biến số để đảm bảo cập nhật)
        self.res_text = tk.StringVar(value="Kết quả: Đang chờ...")
        self.conf_text = tk.StringVar(value="Độ tin cậy: ---")

        self.res_label = tk.Label(self.left_panel, textvariable=self.res_text,
                                  font=("Segoe UI", 16, "bold"), fg="#e74c3c", bg="#f4f7f6")
        self.res_label.pack(pady=(20, 5))

        self.conf_label = tk.Label(self.left_panel, textvariable=self.conf_text,
                                   font=("Segoe UI", 13), fg="#34495e", bg="#f4f7f6")
        self.conf_label.pack()

        # Khung bên phải (Histogram)
        self.right_panel = tk.Frame(root, bg="white", padx=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(self.right_panel, text="Phân tích mức độ điểm ảnh", font=("Segoe UI", 12, "bold"), bg="white").pack(
            pady=15)
        self.fig, self.ax = plt.subplots(figsize=(5, 6), dpi=90)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            # 1. Reset text cũ trước khi dự đoán
            self.res_text.set("Đang xử lý...")
            self.conf_text.set("Vui lòng đợi...")
            self.root.update_idletasks()

            # 2. Hiển thị ảnh
            img = Image.open(path)
            display_img = ImageOps.contain(img, (540, 390))
            img_tk = ImageTk.PhotoImage(display_img)
            self.img_label.configure(image=img_tk, text="")
            self.img_label.image = img_tk

            # 3. Vẽ Histogram
            self.draw_histogram(img)

            # 4. Chạy dự đoán
            self.predict(path)

    def draw_histogram(self, pil_img):
        gray_img = pil_img.convert('L')
        img_array = np.array(gray_img)
        self.ax.clear()
        self.ax.hist(img_array.ravel(), bins=256, range=(0, 255), color='#2980b9')
        self.ax.set_title("Biểu đồ mức xám")
        self.canvas_plot.draw()

    def predict(self, path):
        try:
            # Tiền xử lý ảnh cho AI
            img_model = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
            x = tf.keras.preprocessing.image.img_to_array(img_model) / 255.0
            x = np.expand_dims(x, axis=0)

            # Dự đoán
            preds = self.model.predict(x)
            idx = np.argmax(preds[0])
            conf = np.max(preds[0]) * 100

            # Cập nhật kết quả lên màn hình
            self.res_text.set(f"Kết quả: {CLASS_NAMES[idx]}")
            self.conf_text.set(f"Độ tin cậy: {conf:.2f}%")

        except Exception as e:
            self.res_text.set("Lỗi khi dự đoán")
            messagebox.showerror("Lỗi", f"Không thể dự đoán ảnh: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SkinApp(root)
    root.mainloop()