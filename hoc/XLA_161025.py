import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Xử lý ảnh - Đặng Cao Minh Anh - Nhóm 05")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        self.original_image = None
        self.original_image_color = None
        self.processed_image = None

        self.setup_ui()

    def setup_ui(self):
        # Header thông tin sinh viên
        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill='x', pady=(0,20))
        header_frame.pack_propagate(False)

        name_label = tk.Label(header_frame, text="🎓 ĐẶNG CAO MINH ANH - NHÓM: 05",
                             font=("Arial", 18, "bold"),
                             fg='#ecf0f1', bg='#34495e')
        name_label.pack(expand=True)

        subtitle = tk.Label(header_frame, text="Xử lý ảnh số - Bộ lọc tần số",
                           font=("Arial", 12),
                           fg='#bdc3c7', bg='#34495e')
        subtitle.pack()

        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20)

        # Control panel
        control_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(fill='x', pady=(0,20))

        # Buttons và controls
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(pady=15)

        btn_style = {
            'font': ('Arial', 10, 'bold'),
            'relief': 'raised',
            'bd': 3,
            'cursor': 'hand2'
        }

        # Browse button
        browse_btn = tk.Button(button_frame, text="📁 Browse",
                              command=self.browse_image,
                              bg='#3498db', fg='white',
                              width=12, **btn_style)
        browse_btn.grid(row=0, column=0, padx=5)

        # Dropdown list cho loại lọc
        tk.Label(button_frame, text="Loại lọc:",
                font=('Arial', 10, 'bold'),
                fg='#ecf0f1', bg='#34495e').grid(row=0, column=1, padx=(20,5))

        self.filter_type = ttk.Combobox(button_frame, values=['LT', 'BT', 'Gauss'],
                                       state='readonly', width=8)
        self.filter_type.set('BT')
        self.filter_type.grid(row=0, column=2, padx=5)

        # Low pass button
        low_btn = tk.Button(button_frame, text="🔽 Low",
                           command=self.apply_lowpass,
                           bg='#27ae60', fg='white',
                           width=10, **btn_style)
        low_btn.grid(row=0, column=3, padx=5)

        # High pass button
        high_btn = tk.Button(button_frame, text="🔼 High",
                            command=self.apply_highpass,
                            bg='#e74c3c', fg='white',
                            width=10, **btn_style)
        high_btn.grid(row=0, column=4, padx=5)

        # D0 parameter
        tk.Label(button_frame, text="D0:",
                font=('Arial', 10, 'bold'),
                fg='#ecf0f1', bg='#34495e').grid(row=0, column=5, padx=(20,5))

        self.d0_var = tk.StringVar(value="30")
        d0_entry = tk.Entry(button_frame, textvariable=self.d0_var,
                           width=8, font=('Arial', 10))
        d0_entry.grid(row=0, column=6, padx=5)

        # Image display area
        image_container = tk.Frame(main_frame, bg='#2c3e50')
        image_container.pack(expand=True, fill='both')

        # Original image
        original_frame = tk.Frame(image_container, bg='#34495e', relief='raised', bd=3)
        original_frame.pack(side='left', padx=(0,10), expand=True, fill='both')

        orig_header = tk.Frame(original_frame, bg='#3498db', height=40)
        orig_header.pack(fill='x')
        orig_header.pack_propagate(False)

        tk.Label(orig_header, text="🖼️ ẢNH GỐC",
                font=("Arial", 14, "bold"),
                fg='white', bg='#3498db').pack(expand=True)

        self.original_label = tk.Label(original_frame, bg='#ecf0f1',
                                      relief='sunken', bd=2)
        self.original_label.pack(expand=True, fill='both', padx=10, pady=10)

        # Output image
        output_frame = tk.Frame(image_container, bg='#34495e', relief='raised', bd=3)
        output_frame.pack(side='right', padx=(10,0), expand=True, fill='both')

        out_header = tk.Frame(output_frame, bg='#e74c3c', height=40)
        out_header.pack(fill='x')
        out_header.pack_propagate(False)

        tk.Label(out_header, text="✨ ẢNH OUTPUT",
                font=("Arial", 14, "bold"),
                fg='white', bg='#e74c3c').pack(expand=True)

        self.output_label = tk.Label(output_frame, bg='#ecf0f1',
                                    relief='sunken', bd=2)
        self.output_label.pack(expand=True, fill='both', padx=10, pady=10)

        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text="📌 Sẵn sàng - Chọn ảnh để bắt đầu",
                                    font=('Arial', 10),
                                    fg='#bdc3c7', bg='#34495e')
        self.status_label.pack(side='left', padx=10, pady=5)

    def browse_image(self):
        """Chọn và tải ảnh từ máy tính"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để xử lý",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            # Đọc ảnh màu để hiển thị
            self.original_image_color = cv2.imread(file_path)
            self.original_image_color = cv2.cvtColor(self.original_image_color, cv2.COLOR_BGR2RGB)
            
            # Đọc ảnh grayscale để xử lý
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Hiển thị ảnh màu gốc
            self.display_color_image(self.original_image_color, self.original_label)
            self.status_label.config(text=f"✅ Đã tải ảnh: {file_path.split('/')[-1]}")

    def display_color_image(self, cv_image, label):
        """Hiển thị ảnh màu"""
        height, width = cv_image.shape[:2]
        max_size = 300
        if height > max_size or width > max_size:
            scale = min(max_size/height, max_size/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height))

        pil_image = Image.fromarray(cv_image)
        photo = ImageTk.PhotoImage(pil_image)
        label.configure(image=photo)
        label.image = photo

    def apply_lowpass(self):
        """Áp dụng bộ lọc thông thấp"""
        if self.original_image is None:
            messagebox.showwarning("⚠️ Cảnh báo", "Vui lòng tải ảnh trước!")
            return

        try:
            d0 = float(self.d0_var.get())
            filter_type = self.filter_type.get()

            self.status_label.config(text=f"🔄 Đang áp dụng lọc thông thấp {filter_type}...")
            self.root.update()

            if filter_type == 'LT':  # Lý tưởng
                self.processed_image = self.ideal_lowpass(self.original_image, d0)
            elif filter_type == 'BT':  # Butterworth
                self.processed_image = self.butterworth_lowpass(self.original_image, d0, 2)
            elif filter_type == 'Gauss':  # Gaussian
                self.processed_image = self.gaussian_lowpass(self.original_image, d0)

            self.display_image(self.processed_image, self.output_label)
            self.status_label.config(text=f"✅ Hoàn thành lọc thông thấp {filter_type} với D0={d0}")
        except ValueError:
            messagebox.showerror("❌ Lỗi", "Giá trị D0 không hợp lệ!")

    def apply_highpass(self):
        """Áp dụng bộ lọc thông cao"""
        if self.original_image is None:
            messagebox.showwarning("⚠️ Cảnh báo", "Vui lòng tải ảnh trước!")
            return

        try:
            d0 = float(self.d0_var.get())
            filter_type = self.filter_type.get()

            self.status_label.config(text=f"🔄 Đang áp dụng lọc thông cao {filter_type}...")
            self.root.update()

            if filter_type == 'LT':  # Lý tưởng
                self.processed_image = self.ideal_highpass(self.original_image, d0)
            elif filter_type == 'BT':  # Butterworth
                self.processed_image = self.butterworth_highpass(self.original_image, d0, 2)
            elif filter_type == 'Gauss':  # Gaussian
                self.processed_image = self.gaussian_highpass(self.original_image, d0)

            self.display_image(self.processed_image, self.output_label)
            self.status_label.config(text=f"✅ Hoàn thành lọc thông cao {filter_type} với D0={d0}")
        except ValueError:
            messagebox.showerror("❌ Lỗi", "Giá trị D0 không hợp lệ!")

    def ideal_lowpass(self, image, d0):
        """Bộ lọc thông thấp lý tưởng - cắt sắc nét tại D0"""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        # Tạo mask hình tròn
        mask = np.zeros((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= d0 ** 2
        mask[mask_area] = 1

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def ideal_highpass(self, image, d0):
        """Bộ lọc thông cao lý tưởng - ngược với thông thấp"""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        # Tạo mask ngược với lowpass
        mask = np.ones((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= d0 ** 2
        mask[mask_area] = 0

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def butterworth_lowpass(self, image, d0, n):
        """Bộ lọc Butterworth thông thấp - chuyển tiếp mượt"""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        y, x = np.ogrid[:rows, :cols]
        d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = 1 / (1 + (d / d0) ** (2 * n))

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def butterworth_highpass(self, image, d0, n):
        """Bộ lọc Butterworth thông cao"""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        y, x = np.ogrid[:rows, :cols]
        d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = 1 / (1 + (d0 / (d + 1e-6)) ** (2 * n))  # Tránh chia 0

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def gaussian_lowpass(self, image, d0):
        """Bộ lọc Gaussian thông thấp - mượt nhất"""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        y, x = np.ogrid[:rows, :cols]
        d_squared = (x - ccol) ** 2 + (y - crow) ** 2
        mask = np.exp(-d_squared / (2 * d0 ** 2))

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def gaussian_highpass(self, image, d0):
        """Bộ lọc Gaussian thông cao"""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        y, x = np.ogrid[:rows, :cols]
        d_squared = (x - ccol) ** 2 + (y - crow) ** 2
        mask = 1 - np.exp(-d_squared / (2 * d0 ** 2))

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def display_image(self, cv_image, label):
        """Hiển thị ảnh grayscale"""
        height, width = cv_image.shape
        max_size = 300
        if height > max_size or width > max_size:
            scale = min(max_size/height, max_size/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height))

        pil_image = Image.fromarray(cv_image)
        photo = ImageTk.PhotoImage(pil_image)
        label.configure(image=photo)
        label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()