import matplotlib
matplotlib.use('Agg') # Penting agar matplotlib jalan di background server
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

app = Flask(__name__)

# --- INISIALISASI & TRAINING MODEL (Dilakukan 1x saat server mulai) ---
def train_model():
    df = pd.read_csv('antam_price.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.groupby('Date')['Price'].mean().reset_index().sort_values('Date')
    
    start_date = df['Date'].min()
    df['Time_Index'] = (df['Date'] - start_date).dt.days
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X = df[["Time_Index"]].values
    Y = df[["Price"]].values
    
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    model.fit(X_scaled, Y_scaled, epochs=150, verbose=0) # Epoch dikurangi sedikit agar server cepat nyala
    
    return model, scaler_x, scaler_y, df, start_date

print("Sedang menyiapkan AI, mohon tunggu...")
model, scaler_x, scaler_y, df_emas, start_date = train_model()
print("AI Siap!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tahun = int(request.form['tahun'])
    bulan = int(request.form['bulan'])
    
    target_date = pd.to_datetime(f"{tahun}-{bulan}-01")
    target_day_index = (target_date - start_date).days
    
    # Prediksi
    target_scaled = scaler_x.transform([[target_day_index]])
    pred_scaled = model.predict(target_scaled)
    hasil_final = float(scaler_y.inverse_transform(pred_scaled)[0][0])
    
    # Buat Grafik
    plt.figure(figsize=(10, 5))
    plt.plot(df_emas['Date'], df_emas['Price'], color='deepskyblue', label="Historis", alpha=0.6)
    
    # Garis tren
    X_range = np.linspace(df_emas['Time_Index'].min(), target_day_index, 100).reshape(-1, 1)
    Y_range_pred = scaler_y.inverse_transform(model.predict(scaler_x.transform(X_range)))
    date_range = [start_date + pd.Timedelta(days=int(d)) for d in X_range.flatten()]
    
    plt.plot(date_range, Y_range_pred, color='red', linewidth=2, label="Prediksi AI")
    plt.axvline(x=target_date, color='black', linestyle='--', alpha=0.3)
    plt.title(f"Tren Prediksi Harga Emas: {target_date.strftime('%B %Y')}")
    plt.legend()
    
    # Simpan plot ke memori
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return jsonify({
        'harga': f"Rp {hasil_final:,.0f}",
        'grafik': plot_url,
        'bulan_tahun': target_date.strftime('%B %Y')
    })

if __name__ == '__main__':
    app.run(debug=True)