import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ColorPickerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Color Picker")
        self.master.geometry("800x600")  # Adjust the size of the window

        # Create a frame to hold the color bar and color image
        self.color_bar_frame = tk.Frame(master)
        self.color_bar_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.hue_label = ttk.Label(self.color_bar_frame, text="Select Hue Value (10-179):")
        self.hue_label.pack(side="left", padx=5, pady=5)

        self.hue_scale = ttk.Scale(self.color_bar_frame, from_=10, to=179, orient="horizontal", command=self.update_color)
        self.hue_scale.pack(side="left", padx=5, pady=5)

        # Create a canvas for the color image
        self.canvas_color = tk.Canvas(master, width=100, height=320)
        self.canvas_color.pack(side="left", padx=5, pady=75)

        # Create a canvas for the image
        self.canvas_image = tk.Canvas(master, width=800, height=400)
        self.canvas_image.pack(side="top", padx=5, pady=50)

        self.detect_button = ttk.Button(master, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack(side="top", padx=5, pady=5)
        
        self.image = None
        self.image_rgb = None
        self.image_hsv = None

        # Load the image at the beginning
        self.load_image()

    def load_image(self):
        path = "images/balls.jpg"  # Change this to your image path
        self.image = cv2.imread(path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Display the image with detected regions
        self.display_image(self.image_rgb)

    def update_color(self, value):
        hue_value = int(float(value))
        color_image = np.zeros((400, 100, 3), dtype=np.uint8)
        color_image[:, :] = (hue_value, 255, 255)
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_HSV2RGB)
        color_image_rgb = Image.fromarray(color_image_rgb)

        # Display the color image
        color_image_tk = ImageTk.PhotoImage(image=color_image_rgb)
        self.canvas_color.create_image(0, 0, anchor="nw", image=color_image_tk)
        self.canvas_color.image = color_image_tk

    def display_image(self, image):
        img = Image.fromarray(image)

        # Get the original image dimensions
        img_width, img_height = img.size

        # Calculate the aspect ratio
        aspect_ratio = img_width / img_height

        # Define the maximum width for the displayed image
        max_width = 600

        # Calculate the target width and height for the resized image
        target_width = min(img_width, max_width)
        target_height = int(target_width / aspect_ratio)

        # Resize the image while maintaining the aspect ratio
        img = img.resize((target_width, target_height), Image.LANCZOS)

        # Convert the resized image to PhotoImage
        img = ImageTk.PhotoImage(image=img)

        # Clear previous image and display the resized image
        self.canvas_image.delete("all")
        self.canvas_image.create_image(0, 0, anchor="nw", image=img)
        self.canvas_image.image = img


    def detect_objects(self):
        if self.image is None:
            return
        
        self.load_image()

        # Define the hue range based on the current value of the hue scale
        hue_value = int(self.hue_scale.get())
        lower_limit = np.array([hue_value - 8, 100, 100])
        upper_limit = np.array([hue_value + 8, 255, 255])

        # Create a mask to detect objects within the specified hue range
        mask = cv2.inRange(self.image_hsv, lower_limit, upper_limit)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around the detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the updated image with detected objects
        self.display_image(self.image_rgb)


def main():
    root = tk.Tk()
    app = ColorPickerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
