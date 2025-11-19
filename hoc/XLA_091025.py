import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from scipy import ndimage
from scipy.ndimage import median_filter

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter - Mean & Median")
        self.root.geometry("800x600")
        
        self.original_image = None
        self.filtered_image = None
        
        # GUI Elements
        self.create_widgets()
        
    def create_widgets(self):
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # Load image button
        tk.Button(control_frame, text="Load Image", command=self.load_image, 
                 bg="lightblue", width=15).pack(side=tk.LEFT, padx=5)
        
        # Filter size input
        tk.Label(control_frame, text="Filter Size:").pack(side=tk.LEFT, padx=5)
        self.filter_size = tk.Entry(control_frame, width=5)
        self.filter_size.insert(0, "3")
        self.filter_size.pack(side=tk.LEFT, padx=5)
        
        # Filter buttons
        tk.Button(control_frame, text="Mean Filter", command=self.apply_mean_filter,
                 bg="lightgreen", width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Median Filter", command=self.apply_median_filter,
                 bg="lightcoral", width=15).pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.ax1.set_title("Original Image")
        self.ax2.set_title("Filtered Image")
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                image = Image.open(file_path).convert('L')
                self.original_image = np.array(image)
                
                # Display original image
                self.ax1.clear()
                self.ax1.imshow(self.original_image, cmap='gray')
                self.ax1.set_title("Original Image")
                self.ax1.axis('off')
                
                # Clear filtered image
                self.ax2.clear()
                self.ax2.set_title("Filtered Image")
                self.ax2.axis('off')
                
                self.canvas.draw()
                messagebox.showinfo("Success", "Image loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def apply_mean_filter(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            size = int(self.filter_size.get())
            if size <= 0 or size % 2 == 0:
                messagebox.showerror("Error", "Filter size must be positive odd number!")
                return
                
            # Apply mean filter
            kernel = np.ones((size, size)) / (size * size)
            self.filtered_image = ndimage.convolve(self.original_image, kernel)
            
            # Display filtered image
            self.ax2.clear()
            self.ax2.imshow(self.filtered_image, cmap='gray')
            self.ax2.set_title(f"Mean Filter ({size}x{size})")
            self.ax2.axis('off')
            
            self.canvas.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid filter size!")
        except Exception as e:
            messagebox.showerror("Error", f"Filter failed: {str(e)}")
    
    def apply_median_filter(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            size = int(self.filter_size.get())
            if size <= 0 or size % 2 == 0:
                messagebox.showerror("Error", "Filter size must be positive odd number!")
                return
                
            # Apply median filter
            self.filtered_image = median_filter(self.original_image, size=size)
            
            # Display filtered image
            self.ax2.clear()
            self.ax2.imshow(self.filtered_image, cmap='gray')
            self.ax2.set_title(f"Median Filter ({size}x{size})")
            self.ax2.axis('off')
            
            self.canvas.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid filter size!")
        except Exception as e:
            messagebox.showerror("Error", f"Filter failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()