import tkinter as tk
from tkinter import *
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import os

# List of objects YOLOv8 can identify
objects = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]

class YoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO V8 Detecção de Objetos")
        img = PhotoImage(file=os.path.join(os.path.dirname(__file__), 'icon', 'icon.png'))
        self.root.iconphoto(False, img)
        self.root.geometry("1000x350")
        self.root.configure(bg="#006b34")

        # Make the window resizable false
        self.root.resizable(False, False)

        self.labelTitle = tk.Label(self.root, text="Interface YOLO V8", font=("calibri", 30), bg="#006b34", fg="white")
        self.labelTitle.pack(side=tk.TOP)

        # Create a frame for the checkboxes
        self.checkbox_frame = tk.Frame(root, bg="#006b34")
        self.checkbox_frame.pack()

        # Create object selection checklist in a 10x8 grid
        self.selected_objects = []
        self.checkbuttons = []
        self.create_checkbuttons()

        # Create buttons
        self.camera_button = tk.Button(root, text="Open Camera", command=self.open_camera, fg="black", bg="#00bd00")
        self.camera_button.pack(pady=10)

        self.image_button = tk.Button(root, text="Choose Image", command=self.choose_image, fg="black", bg="#00bd00")
        self.image_button.pack(pady=10)

        # Load YOLOv8 model
        self.model = YOLO("yolov8n.pt")

    def create_checkbuttons(self):
        for idx, obj in enumerate(objects):
            var = tk.IntVar()
            checkbutton = tk.Checkbutton(self.checkbox_frame, text=obj, variable=var, command=lambda obj=obj, var=var: self.update_selected_objects(obj, var), bg="#006b34", fg="white", selectcolor="green")
            row = idx // 10
            col = idx % 10
            checkbutton.grid(row=row, column=col, sticky='w')
            self.checkbuttons.append(checkbutton)

    def update_selected_objects(self, obj, var):
        if var.get() == 1:
            if obj not in self.selected_objects:
                self.selected_objects.append(obj)
        else:
            if obj in self.selected_objects:
                self.selected_objects.remove(obj)

    def open_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.detect_and_display(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def choose_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if filepath:
            image = cv2.imread(filepath)
            self.detect_and_display(image)

    def detect_and_display(self, image):
        results = self.model(image, stream=True)  # Use generator for efficiency
        for result in results:  # Pass in each result
            boxes = result.boxes  # Get the bounding boxes' information
            for box in boxes:  # Pass in each bounding box
                x1, y1, x2, y2 = box.xyxy[0]  # Get the x1, y1, x2, and y2 information
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer
                label = self.model.names[int(box.cls[0])]  # Get the label name
                if label in self.selected_objects:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Draw bounding box
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow("Image", image)

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloApp(root)
    root.mainloop()
