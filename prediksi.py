from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model saat start
model = joblib.load('model_antrian.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        jumlah_antrian = float(request.args.get('jumlah_antrian', 0))
        jenis_layanan = float(request.args.get('jenis_layanan', 0))
        result = model.predict([[jumlah_antrian, jenis_layanan]])
        return jsonify({'prediksi': max(0, round(result[0],2))})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
