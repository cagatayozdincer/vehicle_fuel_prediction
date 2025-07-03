# 🚗 Araç Yakıt Tüketimi Tahmini

Bu proje, farklı araçlara ait teknik özelliklere dayanarak ortalama yakıt tüketimini tahmin etmeyi amaçlamaktadır. Python ile geliştirilmiş bu makine öğrenimi uygulamasında veri analizi, görselleştirme, ön işleme ve farklı regresyon modelleri kullanılarak sonuçlar karşılaştırılmıştır.

---

## 📁 Veri Seti

Veri seti `arac_verileri.csv` dosyasında yer almaktadır ve şu değişkenleri içerir:

- `brand_model`: Araç markası ve modeli  
- `cylinder`: Silindir sayısı  
- `hp`: Beygir gücü  
- `weight`: Ağırlık (kg)  
- `acceleration`: 0-100 km/s hızlanma süresi  
- `model_year`: Araç üretim yılı  
- `target`: Ortalama yakıt tüketimi (L/100 km)

**Not:** Veri, kullanıcı tarafından yapay olarak üretilmiştir.

---

## 🛠️ Kullanılan Kütüphaneler

- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit-learn  
- xgboost  
- scipy

---

## 🧪 Uygulanan Modeller

1. **Linear Regression**  
2. **Ridge Regression** (GridSearchCV ile hiperparametre optimizasyonu)  
3. **Lasso Regression**  
4. **ElasticNet Regression**  
5. **Random Forest Regressor**  
6. **XGBoost Regressor** (GridSearchCV ile hiperparametre araması yapılmıştır)

---

## 📊 Model Performans Karşılaştırması

| Model               | R² Skoru | MSE        |
|--------------------|----------|------------|
| Linear Regression  | 0.6227   | 0.1466     |
| Ridge Regression   | 0.6226   | 0.1466     |
| Lasso Regression   | 0.6302   | 0.1437     |
| ElasticNet         | 0.6300   | 0.1438     |
| Random Forest      | 0.5595   | 0.1711     |
| XGBoost (raw)      | 0.4321   | 0.2206     |
| XGBoost (tuned)    | 0.5938   | 0.1578     |

---

## 🏁 En İyi Model

Model karşılaştırmalarına göre, **Lasso Regression** en yüksek doğruluk (R² = **0.6302**) ve en düşük hata (MSE = **0.1437**) ile en iyi performansı göstermiştir. Bu nedenle projenin nihai değerlendirmesi Lasso modeli üzerinden yapılmıştır.

---

## ⚙️ Kurulum ve Çalıştırma

1. Bu repository'yi klonlayın:

```bash
git clone https://github.com/kullanici_adin/vehicle_fuel_prediction.git
cd vehicle_fuel_prediction

