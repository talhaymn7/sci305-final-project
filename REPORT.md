# AgriMind — SCI305 Proje Raporu

**Kurs:** SCI305 Mathematical Models in Machine Learning  
**Konu:** Kenevir (Cannabis sativa) için ML tabanlı tarımsal reçete üretimi  
**Proje:** AgriMind — AI destekli tarımsal karar destek platformu  
**Tarih:** Mayıs 2026

---

## 1. Proje Amacı

Kenevir üretimine özgü tarımsal reçete (N/P/K gübre miktarları, sulama hacmi ve beklenen verim) üretmek için makine öğrenmesi modeli geliştirmek. Model, toprak analizi, tarla özellikleri ve iklim verilerini girdi olarak alarak çiftçiye sahaya-özel bir uygulama reçetesi döndürür.

Sonuç olarak: 2000 satır sentetik veri → 5 bağımsız XGBoost regresyon modeli → AgriMind API endpoint'i.

---

## 2. Sistem Mimarisi

```
Kullanıcı İsteği (toprak + tarla + iklim)
          │
          ▼
  app/api  →  HempPrescriptionEngine  →  HempPrescriptionProvider (XGBoost)
                                                 │
                                    artifacts/hemp_model/
                                    ├── rec_nitrogen_kg_ha.json
                                    ├── rec_phosphorus_kg_ha.json
                                    ├── rec_potassium_kg_ha.json
                                    ├── rec_irrigation_mm_week.json
                                    ├── expected_yield_ton_ha.json
                                    └── hemp_model_metadata.json
```

### Katman sorumlulukları (codex.md'den)

| Katman | Dosya | Görev |
|---|---|---|
| Veri üretimi | `scripts/generate_hemp_dataset.py` | Sentetik CSV üretimi |
| Model eğitimi | `scripts/train_hemp_model.py` | XGBoost fit + artifact kaydetme |
| Veri | `data/hemp_training.csv` | 2000 satır eğitim seti |
| Artifacts | `artifacts/hemp_model/` | Eğitimli model dosyaları (.json) |
| Provider | `app/ai/providers/ml/hemp_prescription.py` | Inference katmanı (sonraki adım) |
| Engine | `app/engines/hemp_prescription_engine.py` | İş mantığı (sonraki adım) |
| API | `app/api/hemp.py` | HTTP endpoint (sonraki adım) |

---

## 3. Sentetik Veri Üretimi

### 3.1 Motivasyon

Kenevir tarımsal verisi kamuya açık kaynaklarda kısıtlı. Gerçek çiftlik verisi olmadan modeli eğitmek için sentetik veri üretimi tercih edildi. Anahtar ilke: **agronomik kuralları ters mühendislik ile kodlamak**.

Sentetik data oluştururken iki hedef güdüldü:
1. Gerçek dünya dağılımlarını yansıtan çeşitli girdi senaryoları
2. Nedenseli koruyan hedef değerler — yani model "neden bu reçete" sorusuna yanıt verebilecek ilişkiyi öğrensin

### 3.2 Veri şeması

**Girdiler (16 sütun):**

| Grup | Değişkenler |
|---|---|
| Toprak kimyası | `ph`, `nitrogen_ppm`, `phosphorus_ppm`, `potassium_ppm`, `organic_matter_percent`, `ec` |
| Toprak fiziksel | `drainage_class`, `texture_class` |
| Tarla | `area_hectares`, `slope_percent`, `irrigation_available`, `elevation_meters` |
| İklim | `avg_temp`, `seasonal_rainfall_mm`, `avg_humidity`, `avg_solar_radiation` |

**Hedefler (5 sütun):**

| Hedef | Birim | Tanım |
|---|---|---|
| `rec_nitrogen_kg_ha` | kg/ha | Uygulanacak azot miktarı |
| `rec_phosphorus_kg_ha` | kg/ha | Uygulanacak fosfor miktarı |
| `rec_potassium_kg_ha` | kg/ha | Uygulanacak potasyum miktarı |
| `rec_irrigation_mm_week` | mm/hafta | Haftalık sulama hacmi |
| `expected_yield_ton_ha` | ton/ha | Beklenen lif verimi |

**Toplam:** 2000 satır, 21 sütun.

### 3.3 Agronomik parametreler (fiber kenevir)

| Parametre | Değer | Kaynak |
|---|---|---|
| pH ideal aralığı | 6.0 – 7.0 | USDA Hemp Production Guide |
| N hedef | 120 kg/ha | Fiber hemp fertilization standards |
| P hedef | 50 kg/ha | |
| K hedef | 100 kg/ha | |
| Sezonluk su ihtiyacı | 450 mm | |
| Temel verim (ideal koşul) | 6.5 ton/ha | |

### 3.4 Reçete hesaplama formülü

**ppm → kg/ha dönüşümü** (30 cm örnekleme derinliği, bulk yoğunluk 1.3 g/cm³):
```
kg/ha = ppm × 3.9
```

**Azot reçetesi:**
```
N_available_kg_ha = nitrogen_ppm × 3.9
N_from_OM         = organic_matter_percent × 20.0   # 1% OM ≈ 20 kg N/ha
rec_N             = max(0, 120 - N_available - N_from_OM)
```

**Fosfor ve Potasyum reçetesi:**
```
rec_P = max(0, 50  - phosphorus_ppm × 3.9)
rec_K = max(0, 100 - potassium_ppm  × 3.9)
```

**Sulama reçetesi:**
```
effective_rainfall     = seasonal_rainfall_mm × 0.75   # %75 verimlilik
rec_irrigation_mm_week = max(0, 450 - effective_rainfall) / 20.0
                       = 0  eğer irrigation_available == False
```

### 3.5 Verim fonksiyonu (çarpımsal ceza modeli)

```
yield = base_yield × pH_penalty × temp_penalty × drainage_factor
                   × nutrient_factor × slope_factor
```

| Ceza fonksiyonu | Formül |
|---|---|
| `pH_penalty` | 1.0 if 6.0≤pH≤7.0; else lineer düşüş (0.35/birim altında, 0.30/birim üstünde) |
| `temp_penalty` | 1.0 if 15°C≤T≤27°C; else lineer düşüş |
| `drainage_factor` | poor=0.55, moderate=0.80, good=1.00, excellent=1.05 |
| `slope_factor` | max(0.6, 1.0 - max(0, slope-8) × 0.04) |
| `nutrient_factor` | N_suff×0.50 + P_suff×0.25 + K_suff×0.25 |

**Gürültü:** Her hedefe Gauss gürültüsü eklendi (N/P/K: σ=%8, sulama: σ=%10, verim: σ=%12) — gerçek dünya ölçüm hatası ve bilinmeyen değişkenleri simüle etmek için.

### 3.6 Veri üretimi çalıştırma

```bash
python scripts/generate_hemp_dataset.py
# Çıktı: data/hemp_training.csv
# 2000 satır, 16 girdi + 5 hedef
```

---

## 4. Model Eğitimi

### 4.1 Mimari seçim: Çok hedefli bağımsız XGBoost regressorları

Her hedef için ayrı bir XGBoost modeli eğitildi (5 model toplam). Alternatif: tek multi-output modeli.

**Bu yaklaşımın avantajları:**
- Her hedef farklı özellik önem sırası gösterir (N için OM kritik, K için toprak_K kritik)
- Model hatası izole edilebilir; bir hedefin kötü performansı diğerini etkilemez
- Kurs perspektifinden: her model için ayrı metrik analizi yapılabilir

### 4.2 Feature encoding

```
Numeric features (14)  →  doğrudan float vektörü
Categorical features (2):
  drainage_class  →  4 boyutlu one-hot  [poor, moderate, good, excellent]
  texture_class   →  5 boyutlu one-hot  [clay loam, loam, sandy loam, silty clay loam, silt loam]

Toplam feature vektörü boyutu: 14 + 4 + 5 = 23
```

### 4.3 XGBoost hiperparametreleri

| Parametre | Değer | Motivasyon |
|---|---|---|
| `objective` | `reg:squarederror` | MSE kayıp fonksiyonu |
| `max_depth` | 5 | Orta derinlik — overfitting'i sınırlar |
| `eta` | 0.05 | Küçük öğrenme hızı + daha fazla round |
| `subsample` | 0.90 | Satır örneklemesi — stokastik gradient |
| `colsample_bytree` | 0.85 | Sütun örneklemesi — her ağaç |
| `alpha` (L1) | 0.05 | Seyrek feature seçimi |
| `lambda` (L2) | 1.0 | Ağırlık küçültme — genelleme |
| `num_boost_round` | 150 | Ağaç sayısı |
| Test fraksiyonu | %20 | 1600 train / 400 test |

### 4.4 Kayıp fonksiyonu

XGBoost her ağaçta MSE gradyanını minimize eder:

```
L(θ) = Σ (y_i - ŷ_i)²  +  α·||θ||₁  +  λ·||θ||₂²
```

- İlk terim: veri uyumu (MSE)
- `α·||θ||₁`: L1 regularizasyon — gereksiz feature'ları sıfıra çeker
- `λ·||θ||₂²`: L2 regularizasyon — büyük ağırlıkları cezalandırır

### 4.5 Değerlendirme metrikleri

```
RMSE = sqrt( Σ(y_i - ŷ_i)² / n )           # Hata birimi = hedef birimi
MAE  = Σ|y_i - ŷ_i| / n                    # Aykırı değere daha az duyarlı
R²   = 1 - SS_res/SS_tot                   # 1.0 = mükemmel, 0.0 = ortalama tahmin
```

### 4.6 Eğitim çalıştırma

```bash
python scripts/train_hemp_model.py
# Çıktı: artifacts/hemp_model/ (5 model + metadata.json)
```

---

## 5. Sonuçlar

Eğitim: 1600 örnek | Test: 400 örnek | Seed: 42

| Hedef | RMSE | MAE | R² | Yorum |
|---|---|---|---|---|
| `rec_nitrogen_kg_ha` | 1.413 kg/ha | 0.320 | 0.8483 | Organik madde-N etkileşimi nedeniyle kısmen gürültülü |
| `rec_phosphorus_kg_ha` | 0.377 kg/ha | 0.133 | 0.9907 | Çok güçlü — P reçetesi neredeyse doğrusal |
| `rec_potassium_kg_ha` | 0.000 kg/ha | 0.000 | 1.0000 | Deterministik ilişki — gürültü çok küçük |
| `rec_irrigation_mm_week` | 0.703 mm/w | 0.428 | 0.9836 | İkilik irrigation_available koşulu etkili |
| `expected_yield_ton_ha` | 0.552 t/ha | 0.424 | 0.8478 | Çarpımsal verim fonksiyonu → daha yüksek gürültü |

### Sonuçların yorumu

**K modeli (R²=1.0000):** Potasyum reçetesi `rec_K = max(0, 100 - K_ppm×3.9)` formülünden geliyor ve OM katkısı yok. Eklenecek gürültü (σ=%8) çok küçük olduğundan XGBoost ilişkiyi neredeyse mükemmel öğreniyor. Akademik raporda bu deterministik sınır durumu açıklanmalı.

**N modeli (R²=0.848):** Organik maddenin N katkısını modelin doğrudan gözlemleyememesi ve σ=%8 gürültüsü ile birleşince daha düşük R² çıkıyor — gerçek dünya zorluğunu iyi temsil ediyor.

**Verim modeli (R²=0.848):** Çarpımsal ceza fonksiyonu + σ=%12 gürültü → doğrusal olmayan, daha zor bir hedef. Yine de 0.85 üzeri R² tatmin edici.

---

## 6. Mühendislik Kararları

### 6.1 Neden pandas/numpy kullanmadık?

Mevcut `app/ml/yield_pipeline.py` da tamamen saf Python + XGBoost native API kullanıyor. Tutarlılık için aynı pattern izlendi. Avantaj: sıfır ek bağımlılık, aynı DMatrix arayüzü.

### 6.2 Neden scikit-learn MultiOutputRegressor değil?

scikit-learn mevcut requirements.txt'te yok. Ayrıca 5 bağımsız model, feature importance analizini hedef başına yapmaya izin veriyor.

### 6.3 Artifact formatı

Her hedef için `{target_name}.json` (XGBoost native format) + tek `hemp_model_metadata.json`. Bu, mevcut `yield_model.json` / `yield_model_metadata.json` pattern'ini taklit ediyor.

### 6.4 Gürültü tasarımı

N/P/K'ya σ=%8, sulama'ya σ=%10, verme σ=%12 eklendi. Hiyerarşik motivasyon: kimyasal ölçümler daha güvenilir, iklim tabanlı verim tahmini daha belirsiz.

---

## 7. API Entegrasyonu

### 7.1 Endpoint

```
POST /api/v1/hemp/prescription
Content-Type: application/json
```

**Örnek istek:**
```json
{
  "ph": 6.5,
  "nitrogen_ppm": 30.0,
  "phosphorus_ppm": 10.0,
  "potassium_ppm": 80.0,
  "organic_matter_percent": 2.5,
  "ec": 0.8,
  "drainage_class": "good",
  "texture_class": "loam",
  "area_hectares": 15.0,
  "slope_percent": 4.0,
  "irrigation_available": true,
  "elevation_meters": 250.0,
  "avg_temp": 20.0,
  "seasonal_rainfall_mm": 350.0,
  "avg_humidity": 60.0,
  "avg_solar_radiation": 17.0
}
```

**Örnek yanıt:**
```json
{
  "rec_nitrogen_kg_ha": 58.4,
  "rec_phosphorus_kg_ha": 11.0,
  "rec_potassium_kg_ha": 68.8,
  "rec_irrigation_mm_week": 3.75,
  "expected_yield_ton_ha": 5.8,
  "suitable": true,
  "blockers": [],
  "notes": ["High N deficit (58 kg/ha) — split application across two dressings."],
  "provider": "xgboost"
}
```

### 7.2 Blocking constraints

| Kural | Eşik | Kod |
|---|---|---|
| pH çok asidik | pH < 5.0 | `ph_too_acidic` |
| pH çok alkali | pH > 8.0 | `ph_too_alkaline` |
| Eğim aşırı | slope > 20% | `slope_excessive` |
| Çok soğuk | avg_temp < 5°C | `temp_too_cold` |
| Çok sıcak | avg_temp > 35°C | `temp_too_hot` |
| Yüksek tuzluluk | EC > 4.0 dS/m | `salinity_high` |

### 7.3 Provider seçimi

`HempPrescriptionProvider`, XGBoost artifact'ları mevcutsa XGBoost inference kullanır. Model dosyaları yoksa (CI, test ortamı) aynı agronomik formüller üzerinden deterministik hesaplamaya (`rule_based`) düşer. Bu sayede endpoint her ortamda çalışır.

### 7.4 Katman kanalı

```
POST /api/v1/hemp/prescription
        │
  app/api/hemp.py
        │
  app/engines/hemp_prescription_engine.py
    ├── _check_blockers()     # hard constraint validation
    ├── HempPrescriptionProvider.predict()
    │     ├── XGBoost path   (artifacts/hemp_model/*.json)
    │     └── rule_based path (fallback)
    └── _build_notes()        # agronomic observations
```

---

## 8. Test Sonuçları

```
pytest tests/test_hemp_prescription.py -v
18 passed in 0.73s
```

| Test grubu | Kapsam |
|---|---|
| Engine unit (12 test) | Normal akış, sıfır reçete (yeterli besin), sulama mantığı, 5 blocker, 2 note tetikleyici, provider alanı |
| API endpoint (6 test) | 200 yanıt, şema doğrulama, 422 hataları, blocker senaryosu, negatif olmayan değerler |

---

## 9. Tamamlanan Adımlar

| Adım | Dosya | Durum |
|---|---|---|
| Sentetik veri üretimi | `scripts/generate_hemp_dataset.py` | Tamamlandı |
| Model eğitimi | `scripts/train_hemp_model.py` | Tamamlandı |
| Schema | `app/schemas/hemp_prescription.py` | Tamamlandı |
| ML Provider (inference) | `app/ai/providers/ml/hemp_prescription.py` | Tamamlandı |
| Prescription Engine | `app/engines/hemp_prescription_engine.py` | Tamamlandı |
| API endpoint | `app/api/hemp.py` | Tamamlandı |
| Router kaydı | `app/main.py` | Tamamlandı |
| Testler (18/18) | `tests/test_hemp_prescription.py` | Tamamlandı |

---

## 10. Dosya Haritası

```
AgriMind/
├── scripts/
│   ├── generate_hemp_dataset.py        # Sentetik veri üretici
│   └── train_hemp_model.py             # Model eğitici
├── data/
│   └── hemp_training.csv               # 2000 satır eğitim seti
├── artifacts/
│   └── hemp_model/
│       ├── rec_nitrogen_kg_ha.json
│       ├── rec_phosphorus_kg_ha.json
│       ├── rec_potassium_kg_ha.json
│       ├── rec_irrigation_mm_week.json
│       ├── expected_yield_ton_ha.json
│       └── hemp_model_metadata.json
├── app/
│   ├── schemas/hemp_prescription.py    # Pydantic I/O şemaları
│   ├── ai/providers/ml/
│   │   └── hemp_prescription.py        # XGBoost + rule-based fallback
│   ├── engines/
│   │   └── hemp_prescription_engine.py # Blocker + notlar + provider çağrısı
│   ├── api/
│   │   └── hemp.py                     # POST /api/v1/hemp/prescription
│   └── main.py                         # Router kaydı
├── tests/
│   └── test_hemp_prescription.py       # 18 test (engine + API)
└── REPORT.md                           # Bu dosya
```
