# 🚗 Araç Yakıt Tüketimi Tahmini

Bu proje, farklı araçlara ait teknik özelliklere dayanarak ortalama yakıt tüketimini tahmin etmeyi amaçlamaktadır. Python ile geliştirilmiş bu makine öğrenimi uygulamasında veri analizi, görselleştirme, ön işleme ve farklı regresyon modelleri kullanılarak sonuçlar karşılaştırılmıştır.

---

## 📊 Kullanılan Veri Seti

Veri seti, proje içerisinde `arac_verileri.csv` adıyla yer almakta olup aşağıdaki özellikleri içermektedir:

- `brand_model`: Araç markası ve modeli  
- `cylinder`: Silindir sayısı  
- `hp`: Beygir gücü  
- `weight`: Ağırlık (kg)  
- `acceleration`: 0-100 km/s hızlanma süresi (saniye)  
- `model_year`: Üretim yılı  
- `target`: Ortalama yakıt tüketimi (L/100 km)

**Not:** Veri seti kullanıcı tarafından oluşturulmuş yapay verilerden oluşmaktadır.

---

## ⚙️ Kullanılan Kütüphaneler

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy


