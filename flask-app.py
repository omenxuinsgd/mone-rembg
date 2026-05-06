import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from rembg import new_session, remove
from rembg.bg import download_models

from flask_cors import CORS  # Import CORS

app = Flask(__name__)

CORS(app)


# --- 1. KONFIGURASI PATH & PERSISTENSI MODEL ---
# Menggunakan konfigurasi yang sama dengan script asli Anda
model_path = os.path.join(os.getcwd(), "models")
os.makedirs(model_path, exist_ok=True)
os.environ["U2NET_HOME"] = model_path

required_models = ("u2net", "isnet-general-use", "sam")

print(f"[*] Menyingkronkan model ke: {model_path}...")
try:
    download_models(required_models)
    print("[*] Status: Model siap digunakan.")
except Exception as e:
    print(f"[!] Gagal sinkronisasi model: {e}")

# Pre-load session untuk mempercepat API response
sessions = {m: new_session(m) for m in required_models}

# --- 2. UTILITY FUNCTIONS ---
def decode_image(image_bytes):
    # Menggunakan np.frombuffer untuk mengubah bytes menjadi array 1D
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode menjadi format gambar (Bayer/RGB)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image(image_np):
    _, buffer = cv2.imencode('.png', image_np)
    return base64.b64encode(buffer).decode('utf-8')

# --- 3. API ENDPOINT ---
@app.route('/api/remove-bg', methods=['POST'])
def remove_bg():
    try:
        # Pastikan file ada di request
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({"error": "No image uploaded"}), 400
        
        model_name = request.form.get('model', 'u2net')
        x = int(request.form.get('x', 0))
        y = int(request.form.get('y', 0))

        # Membaca bytes langsung dari file stream
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"error": "File buffer is empty"}), 400

        img_np = decode_image(img_bytes)

        # Cek apakah cv2 berhasil men-decode gambar
        if img_np is None:
            return jsonify({"error": "Failed to decode image. Ensure the file is a valid image format."}), 400

        # Lanjutkan ke proses rembg
        session = sessions.get(model_name, sessions['u2net'])
        
        # Sesuai logika asli app2.py
        rembg_kwargs = {}
        if model_name == "sam":
            rembg_kwargs["sam_prompt"] = [{"type": "point", "data": [x, y], "label": 1}]
        
        mask = remove(
            img_np,
            session=session,
            only_mask=True,
            **rembg_kwargs
        )

        # Logic resizing dan masking tetap sama
        if mask.shape[:2] != img_np.shape[:2]:
            mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            
        result_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2BGRA)
        result_img[:, :, 3] = mask

        return jsonify({
            "status": "success",
            "image_base64": encode_image(result_img),
            "mask_base64": encode_image(mask)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
if __name__ == "__main__":
    # Menjalankan server pada port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)