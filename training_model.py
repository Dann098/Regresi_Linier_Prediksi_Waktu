import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv('simulasi_data_antrian.csv')

# Fitur & target
X = data[['jumlah_antrian', 'jenis_layanan']]
y = data['waktu_tunggu']

# Buat model regresi linier
model = LinearRegression()

# Latih model
model.fit(X, y)

# Simpan model
joblib.dump(model, 'model_antrian.pkl')

print("Model Berhasil Dibuat dan Disimpan")
