from flask import Flask, request, render_template
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("selada.pt")  # Ganti dengan path model Anda

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the uploaded image
    image_file = request.files['image']
    
    # Open the image with PIL
    image = Image.open(image_file.stream).convert("RGB")  # Pastikan gambar dalam format RGB
    
    # Resize the image to a smaller size to reduce processing load (optional)
    image = image.resize((640, 640))  # Ubah ukuran gambar menjadi 640x640

    # Convert the image to numpy array
    image_np = np.array(image)

    # Make predictions
    results = model.predict(image_np)

    # Prepare predictions for display
    predictions = []
    for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        x1, y1, x2, y2 = box
        class_name = model.names[int(cls)]  # Ambil nama kelas dari model
        predictions.append(f"Class: {class_name}, Confidence: {conf:.2f}")

        # Draw bounding boxes and class labels
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{class_name} {conf:.2f}"

        try:
            font = ImageFont.truetype("arial.ttf", 15)  # Tentukan font (bisa menggunakan font default jika font khusus tidak tersedia)
        except IOError:
            font = ImageFont.load_default()  # Menggunakan font default jika font khusus tidak ditemukan

        # Draw background for text label
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    # Convert the image with bounding boxes to a BytesIO object
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Convert the BytesIO object to Base64 to send it as a string
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Render the HTML template with the image and predictions
    return render_template('upload.html', predictions=predictions, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
