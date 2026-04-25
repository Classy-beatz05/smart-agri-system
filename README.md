# 🌾 Multilingual Smart Agriculture Recommendation System

**ML-powered crop & fertilizer advisor supporting English, Hindi, Marathi & Kannada**

---

## 📁 Project Structure

```
smart-agri/
├── app.py                    ← Flask backend API
├── train_model.py            ← ML training script (RF vs GBM comparison)
├── requirements.txt
├── models/
│   ├── crop_model.pkl        ← Random Forest crop classifier
│   ├── crop_ms.pkl           ← MinMaxScaler (crop pipeline)
│   ├── crop_sc.pkl           ← StandardScaler (crop pipeline)
│   ├── fert_model.pkl        ← Random Forest fertilizer classifier
│   ├── fert_ms.pkl           ← MinMaxScaler (fertilizer pipeline)
│   ├── fert_sc.pkl           ← StandardScaler (fertilizer pipeline)
│   ├── le_city.pkl           ← LabelEncoder for 50 cities
│   ├── le_crop.pkl           ← LabelEncoder for crop names
│   └── meta.json             ← Model accuracy + config
├── templates/
│   └── index.html            ← Full multilingual UI
├── translations/
│   └── all.json              ← EN / HI / MR / KN translations
└── research/
    ├── confusion_matrix.png  ← Classification results
    ├── roc_curve.png         ← Multi-class ROC curves
    ├── feature_importance.png← Random Forest importances
    ├── heatmap.png           ← Feature correlation
    ├── histograms.png        ← Feature distributions
    └── accuracy_graph.png    ← RF vs GBM comparison
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates models/ and research/)
python train_model.py

# 3. Run the web app
python app.py

# 4. Open in browser
#    http://localhost:5000
```

---

## 🤖 ML Details

### Model Comparison

| Feature          | Random Forest            | Gradient Boosting        |
|------------------|--------------------------|--------------------------|
| Trees/Estimators | 150                      | 100 (depth=5, lr=0.1)    |
| Crop Accuracy    | ~89–99%                  | ~88–97%                  |
| Predict Time     | < 5ms                    | < 8ms                    |
| RAM Usage        | ~80 MB                   | ~40 MB                   |
| Winner           | ✅ (better accuracy)     | ✅ (lighter for Pi)       |

**Winner for deployment: Random Forest** — higher accuracy, still fast enough for Pi 4.

### Feature Engineering

| Feature        | Transformation                        | Reason                          |
|----------------|---------------------------------------|---------------------------------|
| Month          | sin(2π×m/12), cos(2π×m/12)           | Prevents Dec–Jan discontinuity  |
| City           | LabelEncoder                          | 50 Indian cities                |
| Soil nutrients | MinMaxScaler → StandardScaler (chain) | Normalize multi-scale features  |

### Crop Model Inputs
`N, P, K, temperature, humidity, pH, rainfall, EC, month_sin, month_cos, city_enc`

### Fertilizer Model Inputs
`N, P, K, temperature, humidity, pH, rainfall, crop_enc`

### Supported Crops (22)
Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute,
Kidney Beans, Lentil, Maize, Mango, Mothbeans, Mungbean, Muskmelon, Orange,
Papaya, Pigeonpeas, Pomegranate, Rice, Watermelon

### Fertilizers (6)
Urea (46-0-0), DAP (18-46-0), MOP (0-0-60), SSP (0-16-0+S),
NPK 10-26-26, NPK 12-32-16

---

## 🌍 Multilingual Support

| Language | Code | TTS Voice    |
|----------|------|--------------|
| English  | en   | en-IN        |
| Hindi    | hi   | hi-IN        |
| Marathi  | mr   | mr-IN        |
| Kannada  | kn   | kn-IN        |

All translations are in `translations/all.json`. Add new languages by adding a new key.

---

## 🍓 Raspberry Pi 4 Deployment

### Hardware Requirements
- Raspberry Pi 4 (2GB RAM minimum, 4GB recommended)
- 16GB microSD card
- Power supply 5V/3A
- Optional: DHT22 sensor (temp/humidity), soil EC probe

### Setup on Raspberry Pi

```bash
# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Python dependencies
pip3 install flask scikit-learn numpy pandas --break-system-packages

# 3. Clone or copy project
git clone <your-repo> smart-agri
cd smart-agri

# 4. Train models (or copy pre-trained models/)
python3 train_model.py

# 5. Run server (accessible on LAN)
python3 app.py
# Access from phone: http://192.168.x.x:5000
```

### Auto-start on Boot

```bash
# Create systemd service
sudo nano /etc/systemd/system/cropai.service
```

```ini
[Unit]
Description=Smart Agriculture AI
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/smart-agri
ExecStart=/usr/bin/python3 app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable cropai
sudo systemctl start cropai
```

### Sensor Integration (Optional)

```python
# Read DHT22 sensor for real-time temperature & humidity
import Adafruit_DHT
sensor = Adafruit_DHT.DHT22
pin = 4
humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
```

---

## 📊 Research Outputs

All plots saved to `research/` after running `train_model.py`:

- **confusion_matrix.png** — Classification accuracy per crop
- **roc_curve.png** — AUC-ROC for multi-class prediction
- **feature_importance.png** — Most influential soil/climate factors
- **heatmap.png** — Feature correlation matrix
- **histograms.png** — Distribution of all 8 input features
- **accuracy_graph.png** — RF vs GBM cross-validation comparison

---

## 🔌 API Reference

### POST /predict

```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9,
  "ec": 0.45,
  "month": 7,
  "city": "Mumbai",
  "lang": "en"
}
```

**Response:**
```json
{
  "success": true,
  "prediction_ms": 3.2,
  "crop": {
    "name": "rice", "display": "Rice",
    "emoji": "🌾", "color": "#27ae60",
    "confidence": 97.4, "season": "Kharif",
    "yield": "3.0–5.0 t/ha"
  },
  "fertilizer": {
    "name": "Urea", "full": "Urea (46% N)",
    "npk": "46-0-0", "icon": "🟢",
    "desc": "High-nitrogen; promotes rapid growth.",
    "apply": "Split 2–3 times during growth",
    "confidence": 99.1
  },
  "advice_flags": ["advice_high_humidity"]
}
```

---

## 📄 License
MIT License — Free for research and educational use.
