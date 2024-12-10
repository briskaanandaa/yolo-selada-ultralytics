from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from ultralytics import YOLO

app = Flask(__name__)

# Load your YOLO model
model = YOLO("selada.pt")  # Replace with your model path

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the uploaded image
    image_file = request.files['image']
    
    # Open the image with PIL
    image = Image.open(image_file.stream).convert("RGB")  # Ensure image is in RGB format
    
    # Convert the image to numpy array
    image_np = np.array(image)

    # Make predictions
    results = model.predict(image_np)

    # Prepare predictions for the display
    predictions = []
    for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        x1, y1, x2, y2 = box
        class_name = model.names[int(cls)]  # Get class name from model names
        predictions.append(f"Class: {class_name}, Confidence: {conf:.2f}")

        # Draw bounding boxes and class labels
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{class_name} {conf:.2f}"

        try:
            font = ImageFont.truetype("arial.ttf", 15)  # Specify a font (can use a default font if unavailable)
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if custom font not found

        # Get text bounding box
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw a background for the text label
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
        # Draw the text
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    # Save the image with bounding boxes to a BytesIO object
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Generate URL for the image
    image_url = '/static/predicted_image.png'

    # Save the image to the static folder
    image.save("static/predicted_image.png")

    # Render the HTML template with the image and predictions
    return render_template('upload.html', image_url=image_url, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
