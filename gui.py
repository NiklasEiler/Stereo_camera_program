import tkinter as tk
from tkinter import ttk, filedialog
import open3d as o3d
import numpy as np
from PIL import Image, ImageTk

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GUI with Tabs")
        
        self.tab_control = ttk.Notebook(self)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text="Tab 1")
        self.tab_control.add(self.tab2, text="Tab 2")
        
        self.tab_control.pack(expand=1, fill="both")
        
        self.create_tab1()
        self.create_tab2()
        
    def create_tab1(self):
        # Open CSV button
        open_csv_button = ttk.Button(self.tab1, text="Open CSV", command=self.open_csv)
        open_csv_button.grid(row=1, column=1, pady=10)

        # Canvas for displaying point cloud
        self.canvas = tk.Canvas(self.tab1, width=400, height=400)
        self.canvas.grid(row=2, column=1, columnspan=3)

        # Image in the upper right corner
        image_path = "iph_logo.png"  # Replace with actual image path
        img = Image.open(image_path)
        img = img.resize((100, 100))
        self.image_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(self.tab1, image=self.image_tk)
        img_label.grid(row=0, column=2, sticky="ne")

        # Five buttons in the middle
        for i in range(5):
            ttk.Button(self.tab1, text=f"Button {i+1}").grid(row=3+i, column=1, pady=5)

    def create_tab2(self):
        # Open Image button
        open_img_button = ttk.Button(self.tab2, text="Open Image", command=self.open_image)
        open_img_button.grid(row=1, column=1, pady=10)

        # Two input boxes
        ttk.Label(self.tab2, text="Number 1:").grid(row=2, column=1)
        ttk.Entry(self.tab2).grid(row=2, column=2)
        ttk.Label(self.tab2, text="Number 2:").grid(row=3, column=1)
        ttk.Entry(self.tab2).grid(row=3, column=2)

    def open_csv(self):
        csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if csv_file_path:
            # Sample code to create a random point cloud and render it as an image
            points = np.random.rand(100, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            img = pcd_to_image(pcd, width=400, height=400)
            self.render_image(img)

    def open_image(self):
        img_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp")])
        if img_file_path:
            img = Image.open(img_file_path)
            img = img.resize((400, 400))
            self.render_image(img)

    def render_image(self, img):
        self.canvas.delete("all")
        self.image_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

def pcd_to_image(pcd, width, height):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    img = (np.asarray(img) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img

if __name__ == "__main__":
    app = GUI()
    app.mainloop()
