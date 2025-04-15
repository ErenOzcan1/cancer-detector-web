from flask import Flask, request, send_file, render_template_string
import torch
import torchvision
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

app = Flask(__name__)

# Load Faster R-CNN model
def get_faster_rcnn(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Load model and set to evaluation mode
model = get_faster_rcnn()
model.load_state_dict(torch.load("faster_rcnn_mammo.pth", map_location=torch.device('cpu')))
model.eval()

# HTML content to serve at "/"
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Cancer Detection</title>
</head>
<body>
    <h2>Upload Mammography Image for Detection</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required><br><br>
        <input type="submit" value="Submit Image">
    </form>
</body>
</html>
"""


# Route for the homepage with upload form
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_PAGE)


# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    img = Image.open(file).convert("L")  # Convert to grayscale
    img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).unsqueeze(0)


    with torch.no_grad():
        prediction = model([img_tensor])[0]
        print("Boxes:", prediction["boxes"])
        print("Scores:", prediction["scores"])


    # Prepare the image for drawing
    image_np = img_tensor.squeeze().numpy() * 255
    image_np = image_np.astype(np.uint8)
    vis_img = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Draw bounding boxes
    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score < 0.1:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Return the resulting image
    _, buffer = cv2.imencode('.jpg', vis_img)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
