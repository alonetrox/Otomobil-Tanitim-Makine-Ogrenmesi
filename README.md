# 🚗 Otomobil Tanıma ve Bilgilendirme Sistemi

BTK Akademi — Makine Öğrenmesi ve Derin Öğrenme Final Projesi

## 📋 Proje Özeti

Otomobil fotoğraflarından **marka ve model tanıma** yapan, tanımlanan araca ait **motor/şanzıman seçenekleri**, **teknik özellikler**, **kronik sorunlar**, **bakım periyotları** ve **ekspertiz check-list** bilgilerini sunan uçtan uca bir yapay zeka sistemidir.

## 🗂️ Proje Yapısı

```
Otomobil_Tanitim/
├── dataset/                    # Eğitim veri seti (5 sınıf, ~1400 görsel)
│   ├── fiat_egea_2021_2026/
│   ├── hyundai_i20_2023_2026/
│   ├── renault_clio_2023_2026/
│   ├── renault_megane_2021_2026/
│   └── toyota_corolla_2022_2026/
├── data/
│   └── car_database.json       # Araç veritabanı (JSON)
├── models/                     # Eğitilmiş model çıktıları
│   └── class_indices.json
├── train_model.py              # CNN eğitim scripti (ResNet50V2)
├── predictor.py                # Tahmin sınıfı
├── gui_app.py                  # Flet GUI (WinUI3/Fluent Design)
├── requirements.txt            # Bağımlılıklar
└── README.md                   # Bu dosya
```

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 2. Modeli Eğit
```bash
python train_model.py
```
> Aşama 1: Feature Extraction (10 epoch) → Aşama 2: Fine-Tuning (20 epoch)

### 3. Programı Çalıştır
```bash
python gui_app.py
```

## 🛠️ Kullanılan Kütüphaneler

| Bileşen | Kütüphane |
|---------|-----------|
| Derin Öğrenme | TensorFlow / Keras (ResNet50V2) |
| GUI | Flet (Flutter tabanlı, WinUI3/Fluent Design) |
| API | FastAPI + Uvicorn |
| Veri | JSON veritabanı |
| Görüntü İşleme | Pillow, OpenCV |
