# ================================================================
# predictor.py — Otomobil Tanıma Predictor Sınıfı
# Eğitilmiş modeli yükler, tahmin yapar, JSON ile eşleştirir.
# ================================================================

import os, json, numpy as np
from PIL import Image


class CarPredictor:
    """Eğitilmiş CNN modeli ile otomobil tahmin ve JSON veri eşleştirme sınıfı."""

    IMG_SIZE = (224, 224)

    def __init__(self, model_path: str, class_indices_path: str, database_path: str):
        # Lazy import — TensorFlow sadece tahmin anında yüklensin
        import tensorflow as tf
        self.tf = tf

        # Model yükle
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model yüklendi: {model_path}")

        # Sınıf indekslerini yükle  {0: "fiat_egea_2021_2026", ...}
        with open(class_indices_path, "r", encoding="utf-8") as f:
            self.class_indices = json.load(f)

        # JSON veritabanını yükle
        with open(database_path, "r", encoding="utf-8") as f:
            self.database = json.load(f)
        print(f"✅ Veritabanı yüklendi: {len(self.database)} araç")

    # ── Görüntü Ön İşleme ────────────────────────────────────────

    def preprocess_image(self, image_input) -> np.ndarray:
        """Görüntüyü modele uygun formata dönüştür.
        Args:
            image_input: dosya yolu (str) veya PIL.Image veya numpy array
        Returns:
            (1, 224, 224, 3) şeklinde numpy dizisi
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Desteklenmeyen giriş tipi: {type(image_input)}")

        img = img.resize(self.IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        # Modelin içinde (Eğitim sırasında) preprocess_input katmanı zaten eklendi,
        # Bu yüzden burada sadece [0, 255] aralığındaki float32 dizisini gönderiyoruz.
        return np.expand_dims(arr, axis=0)

    # ── Tahmin ────────────────────────────────────────────────────

    def predict(self, image_input) -> dict:
        """Görüntüden araç tahmini yap.
        Returns:
            {
                "label": "fiat_egea_2021_2026",
                "marka": "Fiat",
                "model": "Egea",
                "yil": "2021-2026",
                "confidence": 0.943,
                "top_predictions": [
                    {"label": "...", "confidence": 0.943},
                    ...
                ]
            }
        """
        processed = self.preprocess_image(image_input)
        predictions = self.model.predict(processed, verbose=0)[0]

        # Tüm tahminleri sırala
        sorted_indices = np.argsort(predictions)[::-1]
        top_preds = []
        for idx in sorted_indices:
            label = self.class_indices[str(idx)]
            info = self._parse_label(label)
            top_preds.append({
                "label": label,
                "marka": info["marka"],
                "model": info["model"],
                "confidence": float(predictions[idx]),
            })

        best = top_preds[0]
        return {
            "label": best["label"],
            "marka": best["marka"],
            "model": best["model"],
            "yil": self._parse_label(best["label"])["yil"],
            "confidence": best["confidence"],
            "top_predictions": top_preds,
        }

    # ── Etiket Ayrıştırma ─────────────────────────────────────────

    @staticmethod
    def _parse_label(label: str) -> dict:
        """Klasör adından marka/model/yıl bilgisini çıkar.
        Örnek: 'fiat_egea_2021_2026' → {'marka':'Fiat', 'model':'Egea', 'yil':'2021-2026'}
        """
        parts = label.split("_")
        # Son iki eleman yıl aralığı
        yil_baslangic = parts[-2]
        yil_bitis = parts[-1]
        # İlk eleman marka
        marka = parts[0].capitalize()
        # Ortadaki elemanlar model adı
        model_parts = parts[1:-2]
        model_ad = " ".join(p.capitalize() if not p[0].isdigit() else p for p in model_parts)
        return {
            "marka": marka,
            "model": model_ad,
            "yil": f"{yil_baslangic}-{yil_bitis}",
        }

    # ── JSON Veri Sorgulama ───────────────────────────────────────

    def get_car_details(self, label: str) -> dict | None:
        """Label ile JSON'dan tüm araç bilgilerini çek."""
        return self.database.get(label)

    def get_motor_options(self, label: str) -> list[str]:
        """Motor seçeneklerini listele (ComboBox için)."""
        car = self.database.get(label, {})
        motors = car.get("motor_secenekleri", [])
        return [f"{m['motor_adi']} ({m['beygir']} HP - {m['yakit_tipi']})" for m in motors]

    def get_sanziman_options(self, label: str, motor_index: int) -> list[str]:
        """Seçilen motora ait şanzıman seçeneklerini listele."""
        car = self.database.get(label, {})
        motors = car.get("motor_secenekleri", [])
        if 0 <= motor_index < len(motors):
            sans = motors[motor_index].get("sanziman_secenekleri", [])
            return [s["sanziman"] for s in sans]
        return []

    def get_spec_for_combo(self, label: str, motor_idx: int, sanziman_idx: int) -> dict:
        """Motor + şanzıman kombinasyonuna özel tüm teknik detayları döndür."""
        car = self.database.get(label, {})
        motors = car.get("motor_secenekleri", [])
        if 0 <= motor_idx < len(motors):
            motor = motors[motor_idx]
            sans_list = motor.get("sanziman_secenekleri", [])
            if 0 <= sanziman_idx < len(sans_list):
                sanz = sans_list[sanziman_idx]
                return {
                    "motor_adi": motor["motor_adi"],
                    "yakit_tipi": motor["yakit_tipi"],
                    "silindir_hacmi": motor["silindir_hacmi"],
                    "beygir": motor["beygir"],
                    "tork": motor["tork"],
                    "sanziman": sanz["sanziman"],
                    "yakit_tuketimi": sanz.get("yakit_tuketimi", {}),
                    "hizlanma_0_100": sanz.get("hizlanma_0_100", "N/A"),
                    "max_hiz": sanz.get("max_hiz", "N/A"),
                }
        return {}

    def get_kronik_sorunlar(self, label: str) -> list:
        """Kronik sorunları döndür."""
        car = self.database.get(label, {})
        return car.get("kronik_sorunlar", [])

    def get_bakim_periyotlari(self, label: str) -> dict:
        """Bakım periyotlarını döndür."""
        car = self.database.get(label, {})
        return car.get("bakim_periyotlari", {})

    def get_ekspertiz_checklist(self, label: str) -> list:
        """Ekspertiz check-list'i döndür."""
        car = self.database.get(label, {})
        return car.get("ekspertiz_checklist", [])

    def get_all_cars(self) -> list[dict]:
        """Kütüphane sekmesi için tüm araçların özet bilgilerini listele."""
        cars = []
        for label, data in self.database.items():
            cars.append({
                "label": label,
                "marka": data.get("marka", ""),
                "model": data.get("model", ""),
                "uretim_yillari": data.get("uretim_yillari", ""),
                "segment": data.get("segment", ""),
                "motor_sayisi": len(data.get("motor_secenekleri", [])),
                "fiyat_araligi": data.get("fiyat_araligi", ""),
            })
        return cars
