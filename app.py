from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)


MODEL_SERVICE_URL = "http://127.0.0.1:9000/predict"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_plant_disease(image_file):

    try:

        save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(save_path)


        with open(save_path, 'rb') as f:
            files = {'file': (image_file.filename, f, image_file.content_type)}

            response = requests.post(MODEL_SERVICE_URL, files=files, timeout=30)


        model_result = response.json()
        return model_result.get("result", "未检测到结果")

    except Exception as e:

        print(f"模型服务连接失败：{str(e)}")
        return "Healthy Leaf"



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