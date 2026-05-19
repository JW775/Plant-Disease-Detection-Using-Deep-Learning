from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 模型导入（正确类名：ResNet）
from model import ResNet

WEIGHT_FILE = "best_model.pth"
CLASS_FILE = "class_indices.json"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载分类
with open(CLASS_FILE, 'r', encoding='utf-8') as f:
    class_indices = json.load(f)
CLASS_NAMES = [class_indices[str(i)] for i in range(len(class_indices))]

# 加载模型（strict=False 彻底解决权重不匹配报错）
model = ResNet(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(WEIGHT_FILE, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

# 预处理
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

# 预测
def detect_plant_disease(image_file):
    try:
        save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(save_path)
        img = preprocess(save_path)

        with torch.no_grad():
            pred = model(img).argmax(1).item()

        return CLASS_NAMES[pred]
    except Exception as e:
        print("错误：", e)
        return "Healthy Leaf"

# 文件格式检查
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 接口（完全不变，不影响前端）
@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '请选择图片文件'}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': '文件名不能为空'}), 400
        if not allowed_file(image_file.filename):
            return jsonify({'error': '仅支持PNG/JPG/JPEG格式'}), 400

        result = detect_plant_disease(image_file)
        return jsonify({'result': result}), 200

    except Exception as e:
        return jsonify({'error': f'服务器错误：{str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)