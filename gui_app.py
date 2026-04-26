# ================================================================
# gui_app.py — Otomobil Tanıma Sistemi — Flet GUI
# WinUI3 / Fluent Design — Modern, Premium Arayüz
# ================================================================

import sys, os, json, threading

# DLL Çatışmasını Önlemek İçin TensorFlow İlk Sırada (Top-Level Import)
try:
    from predictor import CarPredictor
except ImportError:
    CarPredictor = None

import flet as ft
from PIL import Image as PILImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "car_database.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

# ── Renk Paleti (Modernize Edilmiş Zinc & Sky Blue Tonları) ──────
C_BG = "#09090B"         # Derin Obsidyen
C_SURFACE = "#18181B"    # Yüzey Zinc
C_CARD = "#27272A"       # Kart Arka Planı
C_CARD_HOVER = "#3F3F46" # Vurgu Kartı
C_BORDER = "#3F3F46"     # Çerçeve
C_ACCENT = "#38BDF8"     # Sky Blue
C_ACCENT2 = "#0EA5E9"    # Cerulean
C_ACCENT_DARK = "#0284C7"
C_TEXT = "#FAFAFA"
C_TEXT2 = "#A1A1AA"
C_SUCCESS = "#22C55E"
C_WARNING = "#FBBF24"
C_DANGER = "#EF4444"


def load_database() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_predictor():
    """Predictor'ı yüklemeyi dene. TF yoksa veya model yoksa None döner."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(INDICES_PATH):
        return None
    try:
        if CarPredictor:
            return CarPredictor(MODEL_PATH, INDICES_PATH, DB_PATH)
        return None
    except Exception as e:
        print(f"⚠️ Predictor yüklenemedi: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════

def main(page: ft.Page):
    # ── Durum Değişkenleri ────────────────────────────────────────
    database = load_database()
    predictor = None  # Lazy load — ilk tahmin anında yüklenecek
    predictor_loaded = {"value": False}
    current_label = {"value": None}
    selected_motor_idx = {"value": -1}
    selected_sanziman_idx = {"value": -1}
    selected_image_path = {"value": None}

    def show_toast(message, color=C_SUCCESS, icon=ft.Icons.CHECK_CIRCLE):
        snack = ft.SnackBar(
            content=ft.Row([
                ft.Icon(icon, color=C_TEXT),
                ft.Text(message, color=C_TEXT, size=14, weight=ft.FontWeight.BOLD)
            ]),
            bgcolor=color,
            behavior=ft.SnackBarBehavior.FLOATING,
            margin=40,
            duration=3000
        )
        try:
            page.open(snack)
        except AttributeError:
            page.snack_bar = snack
            page.snack_bar.open = True
            page.update()

    # ── Sayfa Ayarları ────────────────────────────────────────────
    page.title = "Otomobil Tanıma Sistemi"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = C_BG
    page.padding = 0
    page.spacing = 0
    page.fonts = {
        "Inter": "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.woff2",
    }
    page.theme = ft.Theme(
        font_family="Inter",
        color_scheme=ft.ColorScheme(
            primary=C_ACCENT2,
            on_primary=C_TEXT,
            surface=C_SURFACE,
            on_surface=C_TEXT,
        ),
    )

    # ── Yardımcı Widget Oluşturucular ─────────────────────────────

    def fluent_card(content, width=None, height=None, padding=20, on_click=None):
        return ft.Container(
            content=content,
            width=width,
            height=height,
            padding=padding,
            bgcolor=C_CARD,
            border=ft.Border.all(1, C_BORDER),
            border_radius=8,
            animate=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
            animate_scale=ft.Animation(300, ft.AnimationCurve.DECELERATE),
            scale=1.0,
            on_click=on_click,
            on_hover=lambda e: _card_hover(e),
        )

    def _card_hover(e):
        e.control.bgcolor = C_CARD_HOVER if e.data == "true" else C_CARD
        e.control.scale = 1.02 if e.data == "true" else 1.0
        e.control.update()

    def section_title(icon, text):
        return ft.Row([
            ft.Icon(icon, color=C_ACCENT, size=20),
            ft.Text(text, size=16, weight=ft.FontWeight.W_600, color=C_TEXT),
        ], spacing=8)

    def info_row(label_text, value_text, icon=None):
        return ft.Row([
            ft.Icon(icon, color=C_ACCENT, size=16) if icon else ft.Container(width=0),
            ft.Text(f"{label_text}:", size=13, color=C_TEXT2, width=130),
            ft.Text(str(value_text), size=13, color=C_TEXT, weight=ft.FontWeight.W_500,
                    expand=True, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.START)

    def badge(text, color=C_ACCENT):
        return ft.Container(
            content=ft.Text(text, size=11, color=C_TEXT, weight=ft.FontWeight.W_600),
            bgcolor=color,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            border_radius=12,
        )

    # ══════════════════════════════════════════════════════════════
    # 📄 SAYFA 1 — ANA SAYFA
    # ══════════════════════════════════════════════════════════════

    def build_home_page():
        stats = [
            (ft.Icons.DIRECTIONS_CAR, "Araç Modeli", str(len(database))),
            (ft.Icons.SETTINGS, "Motor Seçeneği",
             str(sum(len(d.get("motor_secenekleri", [])) for d in database.values()))),
            (ft.Icons.IMAGE, "Eğitim Görseli", "~1453"),
            (ft.Icons.PSYCHOLOGY, "CNN Modeli", "ResNet50V2"),
        ]

        stat_cards = []
        for icon, label, value in stats:
            stat_cards.append(fluent_card(
                ft.Column([
                    ft.Icon(icon, color=C_ACCENT, size=32),
                    ft.Text(value, size=28, weight=ft.FontWeight.BOLD, color=C_TEXT),
                    ft.Text(label, size=12, color=C_TEXT2),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                width=200, height=130,
            ))

        features = [
            (ft.Icons.IMAGE_SEARCH, "Akıllı Tanıma",
             "Fotoğraf yükleyin, yapay zeka aracı tanısın."),
            (ft.Icons.BUILD, "Motor & Şanzıman",
             "Bileşen bazlı teknik özellikleri inceleyin."),
            (ft.Icons.WARNING_AMBER, "Kronik Sorunlar",
             "Bilinen arızalar ve çözüm önerileri."),
            (ft.Icons.LIBRARY_BOOKS, "Araç Kütüphanesi",
             "Tüm modellerin detaylı verileri."),
        ]

        feat_cards = []
        for icon, title, desc in features:
            feat_cards.append(fluent_card(
                ft.Row([
                    ft.Container(
                        content=ft.Icon(icon, color=C_ACCENT, size=28),
                        width=52, height=52,
                        bgcolor=C_ACCENT_DARK,
                        border_radius=10,
                        alignment=ft.Alignment.CENTER,
                    ),
                    ft.Column([
                        ft.Text(title, size=14, weight=ft.FontWeight.W_600, color=C_TEXT),
                        ft.Text(desc, size=12, color=C_TEXT2, max_lines=2),
                    ], spacing=2, expand=True),
                ], spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=16,
            ))

        return ft.Column([
            # Hero
            ft.Container(
                content=ft.Column([
                    ft.Text("🚗 Otomobil Tanıma Sistemi", size=36, weight=ft.FontWeight.W_900, color=C_TEXT),
                    ft.Text("Yapay Zeka ile Araç Tanıma, Teknik Bilgi ve Bakım Rehberi",
                            size=16, color=C_TEXT2),
                    ft.Container(height=4),
                    ft.Text("BTK Akademi — Makine Öğrenmesi ve Derin Öğrenme Final Projesi",
                            size=13, color=C_ACCENT, italic=True),
                ], spacing=6),
                padding=ft.Padding.symmetric(horizontal=25, vertical=35),
                gradient=ft.LinearGradient(
                    begin=ft.Alignment(-1, -1),
                    end=ft.Alignment(1, 1),
                    colors=[C_SURFACE, C_BG],
                ),
                border_radius=16,
                border=ft.Border.all(1, C_BORDER),
            ),
            # İstatistikler
            ft.Row(stat_cards, spacing=16, wrap=True),
            ft.Container(height=10),
            # Özellikler
            section_title(ft.Icons.STAR, "Özellikler"),
            ft.Container(height=6),
            ft.Column(feat_cards, spacing=10),
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)

    # ══════════════════════════════════════════════════════════════
    # 📄 SAYFA 2 — TANIMA
    # ══════════════════════════════════════════════════════════════

    # Dinamik widget referansları
    img_preview = ft.Image(
        src="", width=380, height=280, fit=ft.BoxFit.CONTAIN,
        border_radius=8, visible=False,
    )
    upload_area_text = ft.Text(
        "Bir otomobil fotoğrafı yükleyin", size=14, color=C_TEXT2,
        text_align=ft.TextAlign.CENTER,
    )
    upload_area_icon = ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED, size=48, color=C_BORDER)
    predict_btn = ft.Button(
        "🔍 Tahmin Et", bgcolor=C_ACCENT2, color=C_TEXT,
        height=44, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
        disabled=True, visible=False,
    )
    result_panel = ft.Column([], spacing=10, visible=False, expand=True, scroll=ft.ScrollMode.AUTO)
    loading_ring = ft.ProgressRing(width=32, height=32, color=C_ACCENT, visible=False)
    
    confidence_bar = ft.Container(
        width=0, height=8, bgcolor=C_SUCCESS, border_radius=4,
        animate_size=ft.Animation(1000, ft.AnimationCurve.DECELERATE),
        animate=ft.Animation(800, ft.AnimationCurve.EASE_OUT)
    )
    confidence_bg = ft.Container(
        content=confidence_bar, width=300, height=8, bgcolor=C_BORDER,
        border_radius=4, alignment=ft.Alignment(-1, 0)
    )
    
    confidence_text = ft.Text("", size=14, color=C_SUCCESS, weight=ft.FontWeight.BOLD)

    no_model_banner = ft.Container(visible=False)

    # File picker — Flet 0.83: hiçbir yere eklenmemeli (Inline Service)
    file_picker = ft.FilePicker()

    async def on_upload_click_async(e):
        result = await file_picker.pick_files(
            dialog_title="Otomobil Fotoğrafı Seçin",
            file_type=ft.FilePickerFileType.IMAGE,
            allowed_extensions=["jpg", "jpeg", "png", "webp"],
        )
        if result and len(result) > 0:
            path = result[0].path
            selected_image_path["value"] = path
            img_preview.src = path
            img_preview.visible = True
            upload_area_text.value = os.path.basename(path)
            upload_area_icon.visible = False
            predict_btn.visible = False
            result_panel.visible = False
            page.update()
            
            show_toast("Fotoğraf yüklendi, analiz başlıyor...", C_ACCENT2, ft.Icons.CLOUD_DONE)
            
            # Seçimden hemen sonra analizi otomatik başlat
            on_predict_click(None)

    def on_upload_click(e):
        page.run_task(on_upload_click_async, e)

    def on_reset_click(e):
        selected_image_path["value"] = None
        current_label["value"] = None
        img_preview.visible = False
        img_preview.src = ""
        upload_area_text.value = "Bir otomobil fotoğrafı yükleyin"
        upload_area_icon.visible = True
        predict_btn.disabled = True
        predict_btn.visible = False
        result_panel.visible = False
        page.update()

    def on_predict_click(e):
        nonlocal predictor
        if not selected_image_path["value"]:
            return

        # Lazy load predictor
        if not predictor_loaded["value"]:
            loading_ring.visible = True
            predict_btn.disabled = True
            page.update()
            predictor = load_predictor()
            predictor_loaded["value"] = True

        if predictor is None:
            loading_ring.visible = False
            predict_btn.disabled = False
            show_toast("Model bulunamadı! 'train_model.py' çalıştırın.", C_DANGER, ft.Icons.ERROR)
            no_model_banner.content = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.WARNING_AMBER, color=C_WARNING, size=20),
                    ft.Text("Model dosyası bulunamadı. Önce train_model.py çalıştırın.",
                            color=C_WARNING, size=13),
                ], spacing=8),
                bgcolor="#3D3000", padding=12, border_radius=8,
            )
            no_model_banner.visible = True
            page.update()
            return

        loading_ring.visible = True
        predict_btn.disabled = True
        no_model_banner.visible = False
        page.update()

        # Tahmin yap
        try:
            result = predictor.predict(selected_image_path["value"])
            show_prediction_result(result)
        except Exception as ex:
            no_model_banner.content = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.ERROR, color=C_DANGER, size=20),
                    ft.Text(f"Tahmin hatası: {str(ex)}", color=C_DANGER, size=13),
                ], spacing=8),
                bgcolor="#3D0000", padding=12, border_radius=8,
            )
            no_model_banner.visible = True
        finally:
            loading_ring.visible = False
            predict_btn.disabled = False
            page.update()

    predict_btn.on_click = on_predict_click

    def show_prediction_result(result):
        label = result["label"]
        current_label["value"] = label
        conf = result["confidence"]
        car = database.get(label, {})

        # Animated Container Update
        confidence_bar.width = 300 * conf
        confidence_bar.bgcolor = C_SUCCESS if conf > 0.7 else (C_WARNING if conf > 0.4 else C_DANGER)
        confidence_text.value = f"%{conf*100:.1f}"
        confidence_text.color = confidence_bar.bgcolor

        # Top predictions
        top_chips = []
        for p in result.get("top_predictions", [])[:3]:
            top_chips.append(ft.Container(
                content=ft.Text(f"{p['marka']} {p['model']} — %{p['confidence']*100:.1f}",
                                size=11, color=C_TEXT),
                bgcolor=C_ACCENT_DARK if p["label"] == label else C_BORDER,
                padding=ft.Padding.symmetric(horizontal=10, vertical=4),
                border_radius=12,
            ))

        result_panel.controls = [
            # Araç başlığı
            ft.Row([
                ft.Icon(ft.Icons.DIRECTIONS_CAR, color=C_ACCENT, size=28),
                ft.Column([
                    ft.Text(f"{result['marka']} {result['model']}", size=22,
                            weight=ft.FontWeight.BOLD, color=C_TEXT),
                    ft.Text(f"Üretim: {result.get('yil', '')}  |  {car.get('segment', '')}",
                            size=13, color=C_TEXT2),
                ], spacing=2),
            ], spacing=12),
            # Güven çubuğu (Animated)
            ft.Row([
                ft.Text("Güven:", size=13, color=C_TEXT2),
                confidence_bg,
                confidence_text,
            ], spacing=10),
            ft.Row(top_chips, spacing=6, wrap=True),
            ft.Divider(height=1, color=C_BORDER),
            # Hızlı bilgiler
            ft.Row([
                badge(car.get("segment", ""), C_ACCENT_DARK),
                badge(f"⭐ {car.get('guvenlik_puani', '')}", "#2E7D32"),
                badge(f"💰 {car.get('fiyat_araligi', '')}", "#5D4037"),
            ], spacing=6, wrap=True),
            ft.Container(height=4),
            info_row("Bagaj", car.get("bagaj_hacmi", ""), ft.Icons.LUGGAGE),
            info_row("Lastik", car.get("lastik_boyutu", ""), ft.Icons.TIRE_REPAIR),
            info_row("Ağırlık", car.get("agirlik", ""), ft.Icons.SCALE),
            info_row("Üretim", car.get("uretim_ulkesi", ""), ft.Icons.FACTORY),
            info_row("Sigorta", f"Grup {car.get('sigorta_grubu', '')}", ft.Icons.SECURITY),
        ]
        result_panel.visible = True

        # Kombinasyon dropdown doldur
        motors = car.get("motor_secenekleri", [])
        opts = []
        for m_idx, m in enumerate(motors):
            sans_list = m.get("sanziman_secenekleri", [])
        # Kronik Sorunlar, Bakım, Opsiyonlar statik blokları
        kroniks = car.get("kronik_sorunlar", [])
        bakimlar = car.get("bakim_periyotlari", {})
        checklist = car.get("ekspertiz_checklist", [])

        kronik_items = []
        for k in kroniks:
            severity_color = {"Düşük": C_SUCCESS, "Orta": C_WARNING, "Yüksek": C_DANGER}.get(k["ciddiyet"], C_TEXT2)
            kronik_items.append(ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Container(width=8, height=8, bgcolor=severity_color, border_radius=4),
                        ft.Text(k["sorun"], size=13, weight=ft.FontWeight.W_600, color=C_TEXT),
                        badge(k["ciddiyet"], severity_color),
                    ], spacing=8),
                    ft.Text(k["detay"], size=11, color=C_TEXT2, max_lines=3),
                    ft.Text(f"💡 {k['cozum']}", size=11, color=C_ACCENT, italic=True, max_lines=2),
                ], spacing=4),
                padding=10, bgcolor=C_SURFACE, border_radius=6,
            ))

        bakim_items = []
        for km_key, bdata in bakimlar.items():
            km_label = km_key.replace("_km", " km").replace("_", ".")
            bakim_items.append(ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.BUILD_CIRCLE, color=C_ACCENT, size=16),
                        ft.Text(km_label, size=13, weight=ft.FontWeight.W_600, color=C_TEXT),
                        ft.Text(bdata["tahmini_maliyet"], size=11, color=C_WARNING),
                    ], spacing=8),
                    ft.Column([
                        ft.Text(f"  • {item}", size=11, color=C_TEXT2)
                        for item in bdata["islemler"]
                    ], spacing=2),
                ], spacing=4),
                padding=10, bgcolor=C_SURFACE, border_radius=6,
            ))

        check_items = [
            ft.Row([
                ft.Checkbox(value=False, fill_color=C_ACCENT2, check_color=C_TEXT),
                ft.Text(item, size=12, color=C_TEXT2),
            ], spacing=2)
            for item in checklist
        ]

        # Motor ve Şanzıman Kombinasyonları (Liste Olarak)
        motors = car.get("motor_secenekleri", [])
        motor_cards = []
        
        for m_idx, motor in enumerate(motors):
            sans_list = motor.get("sanziman_secenekleri", [])
            for s_idx, sanz in enumerate(sans_list):
                yt = sanz.get("yakit_tuketimi", {})
                tab_name = f"{motor['motor_adi']} - {sanz['sanziman']}"
                if len(tab_name) > 30:
                    tab_name = f"{motor['motor_adi']} - {sanz['sanziman'][:10]}..."

                motor_content = ft.Column([
                    fluent_card(ft.Column([
                        ft.Text(tab_name, size=15, weight=ft.FontWeight.W_700, color=C_ACCENT),
                        ft.Divider(height=1, color=C_BORDER),
                        info_row("Motor", motor["motor_adi"], ft.Icons.ENGINEERING),
                        info_row("Yakıt", motor["yakit_tipi"], ft.Icons.LOCAL_GAS_STATION),
                        info_row("Hacim", motor["silindir_hacmi"], ft.Icons.WAVES),
                        info_row("Güç", f"{motor['beygir']} HP", ft.Icons.FLASH_ON),
                        info_row("Tork", motor["tork"], ft.Icons.ROTATE_RIGHT),
                        info_row("Şanzıman", sanz["sanziman"], ft.Icons.SETTINGS),
                        info_row("Emisyon", motor.get("emisyon_sinifi", ""), ft.Icons.ECO),
                        ft.Divider(height=1, color=C_BORDER),
                        info_row("0-100 km/s", sanz.get("hizlanma_0_100", "N/A"), ft.Icons.TIMER),
                        info_row("Max Hız", sanz.get("max_hiz", "N/A"), ft.Icons.SPEED),
                        ft.Divider(height=1, color=C_BORDER),
                        ft.Text("⛽ Yakıt Tüketimi", size=13, color=C_ACCENT, weight=ft.FontWeight.W_600),
                        info_row("Şehir İçi", yt.get("sehir_ici", ""), ft.Icons.LOCATION_CITY),
                        info_row("Şehir Dışı", yt.get("sehir_disi", ""), ft.Icons.LANDSCAPE),
                        info_row("Ortalama", yt.get("ortalama", ""), ft.Icons.ANALYTICS),
                    ], spacing=6), padding=14)
                ])

                motor_cards.append(ft.Container(content=motor_content))

        if motor_cards:
            result_panel.controls.append(ft.Container(height=6))
            result_panel.controls.append(section_title(ft.Icons.SPEED, "Seçenek Paketleri"))
            result_panel.controls.append(ft.Column(motor_cards, spacing=12))

        if kronik_items:
            result_panel.controls.append(ft.Container(height=6))
            result_panel.controls.append(section_title(ft.Icons.WARNING_AMBER, "Kronik Sorunlar"))
            result_panel.controls.append(ft.Column(kronik_items, spacing=6))

        if bakim_items:
            result_panel.controls.append(ft.Container(height=6))
            result_panel.controls.append(section_title(ft.Icons.BUILD, "Bakım Periyotları"))
            result_panel.controls.append(ft.Column(bakim_items, spacing=6))

        if check_items:
            result_panel.controls.append(ft.Container(height=6))
            result_panel.controls.append(section_title(ft.Icons.CHECKLIST, "Ekspertiz Check-list"))
            result_panel.controls.append(fluent_card(ft.Column(check_items, spacing=0), padding=10))

        page.update()

    def build_recognition_page():
        upload_area = ft.Container(
            content=ft.Column([
                upload_area_icon,
                upload_area_text,
                ft.Text(".jpg, .png, .webp", size=11, color=C_BORDER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8,
                alignment=ft.MainAxisAlignment.CENTER),
            width=400, height=200,
            border=ft.Border.all(2, C_BORDER),
            border_radius=12,
            alignment=ft.Alignment.CENTER,
            on_click=on_upload_click,
            ink=True,
        )

        left_panel = ft.Column([
            section_title(ft.Icons.IMAGE_SEARCH, "Fotoğraf Yükle"),
            upload_area,
            img_preview,
            ft.Row([
                ft.Button(
                    "📁 Dosya Seç", on_click=on_upload_click,
                    bgcolor=C_ACCENT2, color=C_TEXT, height=40,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                ),
                ft.OutlinedButton(
                    "🔄 Sıfırla", on_click=on_reset_click,
                    height=40,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        side=ft.BorderSide(1, C_BORDER),
                    ),
                ),
            ], spacing=10),
            predict_btn,
            loading_ring,
            no_model_banner,
        ], spacing=12, width=420, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        right_panel = ft.Column([
            section_title(ft.Icons.ANALYTICS, "Tahmin Sonuçları ve Donanımlar"),
            result_panel
        ], spacing=12, expand=True, scroll=ft.ScrollMode.AUTO)

        return ft.Row([
            left_panel,
            ft.VerticalDivider(width=1, color=C_BORDER),
            right_panel,
        ], spacing=20, expand=True, vertical_alignment=ft.CrossAxisAlignment.START)

    # ══════════════════════════════════════════════════════════════
    # 📄 SAYFA 3 — KÜTÜPHANE
    # ══════════════════════════════════════════════════════════════

    library_content = ft.Column([], spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)
    library_detail = ft.Column([], spacing=10, scroll=ft.ScrollMode.AUTO,
                               expand=True, visible=False)
    search_field = ft.TextField(
        hint_text="Araç ara...", width=300, height=42,
        border_color=C_BORDER, focused_border_color=C_ACCENT,
        prefix_icon=ft.Icons.SEARCH, text_style=ft.TextStyle(color=C_TEXT),
        hint_style=ft.TextStyle(color=C_TEXT2),
    )

    def build_library_cards(filter_text=""):
        cards = []
        for label, data in database.items():
            full_name = f"{data.get('marka','')} {data.get('model','')}".lower()
            if filter_text and filter_text.lower() not in full_name:
                continue
            motors_count = len(data.get("motor_secenekleri", []))
            card = fluent_card(
                ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.DIRECTIONS_CAR, color=C_ACCENT, size=36),
                        ft.Column([
                            ft.Text(f"{data['marka']} {data['model']}", size=18,
                                    weight=ft.FontWeight.BOLD, color=C_TEXT),
                            ft.Text(data.get("uretim_yillari", ""), size=12, color=C_TEXT2),
                        ], spacing=2, expand=True),
                        badge(data.get("segment", "").split("/")[0].strip(), C_ACCENT_DARK),
                    ], spacing=12),
                    ft.Divider(height=1, color=C_BORDER),
                    ft.Row([
                        ft.Column([
                            ft.Text(f"⭐ {data.get('guvenlik_puani', '')}", size=11, color=C_TEXT2),
                            ft.Text(f"🔧 {motors_count} motor seçeneği", size=11, color=C_TEXT2),
                        ], spacing=2, expand=True),
                        ft.Column([
                            ft.Text(f"💰 {data.get('fiyat_araligi', '')}", size=11, color=C_TEXT2),
                            ft.Text(f"🛞 {data.get('lastik_boyutu', '')}", size=11, color=C_TEXT2),
                        ], spacing=2, expand=True),
                    ]),
                    ft.Row([
                        ft.TextButton("Detayları Gör →", on_click=lambda e, lbl=label: show_library_detail(lbl),
                                       style=ft.ButtonStyle(color=C_ACCENT)),
                    ], alignment=ft.MainAxisAlignment.END),
                ], spacing=8),
                padding=16,
            )
            cards.append(card)
        return cards

    def on_search_change(e):
        library_content.controls = build_library_cards(search_field.value)
        page.update()

    search_field.on_change = on_search_change

    def show_library_detail(label):
        car = database.get(label, {})
        if not car:
            return

        motors_info = []
        for mi, motor in enumerate(car.get("motor_secenekleri", [])):
            sans_texts = []
            for si, sanz in enumerate(motor.get("sanziman_secenekleri", [])):
                yt = sanz.get("yakit_tuketimi", {})
                sans_texts.append(ft.Container(
                    content=ft.Column([
                        ft.Text(f"⚙️ {sanz['sanziman']}", size=12, weight=ft.FontWeight.W_600, color=C_TEXT),
                        info_row("0-100", sanz.get("hizlanma_0_100", "")),
                        info_row("Max Hız", sanz.get("max_hiz", "")),
                        info_row("Ortalama", yt.get("ortalama", "")),
                    ], spacing=3),
                    padding=8, bgcolor=C_SURFACE, border_radius=6,
                ))

            motors_info.append(fluent_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.ENGINEERING, color=C_ACCENT, size=20),
                    ft.Text(f"{motor['motor_adi']} — {motor['beygir']} HP",
                            size=14, weight=ft.FontWeight.W_600, color=C_TEXT),
                    badge(motor["yakit_tipi"], C_ACCENT_DARK),
                ], spacing=8),
                info_row("Hacim", motor["silindir_hacmi"]),
                info_row("Tork", motor["tork"]),
                ft.Text("Şanzıman Seçenekleri:", size=12, color=C_TEXT2, weight=ft.FontWeight.W_600),
                ft.Column(sans_texts, spacing=6),
            ], spacing=6), padding=14))

        kroniks = []
        for k in car.get("kronik_sorunlar", []):
            sc = {"Düşük": C_SUCCESS, "Orta": C_WARNING, "Yüksek": C_DANGER}.get(k["ciddiyet"], C_TEXT2)
            kroniks.append(ft.Container(
                content=ft.Column([
                    ft.Row([ft.Container(width=8, height=8, bgcolor=sc, border_radius=4),
                            ft.Text(k["sorun"], size=13, weight=ft.FontWeight.W_600, color=C_TEXT),
                            badge(k["ciddiyet"], sc)], spacing=8),
                    ft.Text(k["detay"], size=11, color=C_TEXT2),
                    ft.Text(f"💡 {k['cozum']}", size=11, color=C_ACCENT, italic=True),
                ], spacing=3), padding=10, bgcolor=C_SURFACE, border_radius=6))

        library_detail.controls = [
            ft.Row([
                ft.IconButton(ft.Icons.ARROW_BACK, icon_color=C_ACCENT,
                              on_click=lambda e: hide_library_detail()),
                ft.Text(f"{car['marka']} {car['model']}", size=24,
                        weight=ft.FontWeight.BOLD, color=C_TEXT),
                ft.Container(expand=True),
                badge(car.get("segment", "").split("/")[0].strip(), C_ACCENT_DARK),
            ], spacing=10),
            fluent_card(ft.Column([
                info_row("Üretim Yılları", car.get("uretim_yillari", ""), ft.Icons.CALENDAR_MONTH),
                info_row("Güvenlik", car.get("guvenlik_puani", ""), ft.Icons.SECURITY),
                info_row("Fiyat", car.get("fiyat_araligi", ""), ft.Icons.PAYMENTS),
                info_row("Sigorta Grubu", car.get("sigorta_grubu", ""), ft.Icons.SHIELD),
                info_row("Lastik", car.get("lastik_boyutu", ""), ft.Icons.TIRE_REPAIR),
                info_row("Bagaj", car.get("bagaj_hacmi", ""), ft.Icons.LUGGAGE),
                info_row("Ağırlık", car.get("agirlik", ""), ft.Icons.SCALE),
                info_row("Üretim Yeri", car.get("uretim_ulkesi", ""), ft.Icons.FACTORY),
            ], spacing=5), padding=14),
            section_title(ft.Icons.ENGINEERING, "Motor & Şanzıman Seçenekleri"),
            ft.Column(motors_info, spacing=8),
            section_title(ft.Icons.WARNING_AMBER, "Kronik Sorunlar"),
            ft.Column(kroniks, spacing=6),
        ]
        library_detail.visible = True
        library_content.visible = False
        search_field.visible = False
        page.update()

    def hide_library_detail():
        library_detail.visible = False
        library_content.visible = True
        search_field.visible = True
        page.update()

    def build_library_page():
        library_content.controls = build_library_cards()
        return ft.Column([
            ft.Row([
                section_title(ft.Icons.LIBRARY_BOOKS, "Araç Kütüphanesi"),
                ft.Container(expand=True),
                search_field,
            ], spacing=10),
            library_content,
            library_detail,
        ], spacing=12, expand=True)

    # ══════════════════════════════════════════════════════════════
    # 📄 SAYFA 4 — HAKKINDA
    # ══════════════════════════════════════════════════════════════

    def build_about_page():
        tech_stack = [
            ("Python 3.10+", "Ana programlama dili"),
            ("TensorFlow / Keras", "CNN model eğitimi (ResNet50V2)"),
            ("Flet (Flutter)", "WinUI3/Fluent Design masaüstü GUI"),
            ("FastAPI", "REST API sunucusu"),
            ("Pillow + OpenCV", "Görüntü ön işleme"),
            ("JSON", "Araç veritabanı"),
        ]

        tech_rows = [
            ft.Row([
                ft.Container(width=8, height=8, bgcolor=C_ACCENT, border_radius=4),
                ft.Text(name, size=13, weight=ft.FontWeight.W_600, color=C_TEXT, width=180),
                ft.Text(desc, size=12, color=C_TEXT2),
            ], spacing=10)
            for name, desc in tech_stack
        ]

        return ft.Column([
            section_title(ft.Icons.INFO, "Proje Hakkında"),
            fluent_card(ft.Column([
                ft.Text("🚗 Otomobil Tanıma ve Bilgilendirme Sistemi", size=20,
                         weight=ft.FontWeight.BOLD, color=C_TEXT),
                ft.Container(height=6),
                ft.Text(
                    "Bu proje, BTK Akademi Makine Öğrenmesi ve Derin Öğrenme eğitiminin "
                    "final projesi olarak geliştirilmiştir. Yapay zeka kullanarak otomobil "
                    "fotoğraflarından marka ve model tanıma gerçekleştirir, ardından "
                    "kullanıcıya kapsamlı teknik bilgi, kronik sorunlar ve bakım "
                    "rehberliği sunar.",
                    size=13, color=C_TEXT2, max_lines=10,
                ),
            ], spacing=4)),

            ft.Container(height=8),
            section_title(ft.Icons.CODE, "Teknoloji Yığını"),
            fluent_card(ft.Column(tech_rows, spacing=8), padding=16),

            ft.Container(height=8),
            section_title(ft.Icons.SCHOOL, "Eğitim Bilgileri"),
            fluent_card(ft.Column([
                info_row("Eğitim", "Makine Öğrenmesi ve Derin Öğrenme"),
                info_row("Kurum", "BTK Akademi"),
                info_row("Proje Türü", "Final Projesi"),
                info_row("Model", "ResNet50V2 (Transfer Learning)"),
                info_row("Veri Seti", f"{len(database)} sınıf, ~1453 görsel"),
            ], spacing=6), padding=16),
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)

    # ══════════════════════════════════════════════════════════════
    # NAVİGASYON
    # ══════════════════════════════════════════════════════════════

    pages_cache = {}

    def get_page(index):
        if index not in pages_cache:
            builders = [build_home_page, build_recognition_page,
                        build_library_page, build_about_page]
            pages_cache[index] = builders[index]()
        return pages_cache[index]

    animated_switcher = ft.AnimatedSwitcher(
        content=get_page(0),
        transition=ft.AnimatedSwitcherTransition.SCALE,
        duration=400,
        reverse_duration=300,
        switch_in_curve=ft.AnimationCurve.DECELERATE,
        switch_out_curve=ft.AnimationCurve.EASE_OUT,
    )

    content_area = ft.Container(
        content=animated_switcher,
        expand=True,
        padding=ft.Padding.only(left=30, right=30, top=20, bottom=20),
    )

    def on_nav_change(e):
        idx = e.control.selected_index
        animated_switcher.content = get_page(idx)
        page.update()

    def toggle_theme(e):
        if page.theme_mode == ft.ThemeMode.DARK:
            page.theme_mode = ft.ThemeMode.LIGHT
            page.bgcolor = "#F5F5F5"
            theme_btn.icon = ft.Icons.LIGHT_MODE
        else:
            page.theme_mode = ft.ThemeMode.DARK
            page.bgcolor = C_BG
            theme_btn.icon = ft.Icons.DARK_MODE
        page.update()

    theme_btn = ft.IconButton(
        icon=ft.Icons.DARK_MODE, icon_color=C_TEXT2,
        tooltip="Tema Değiştir", on_click=toggle_theme,
    )

    nav_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=82,
        group_alignment=-0.85,
        bgcolor=C_SURFACE,
        indicator_color=C_ACCENT_DARK,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.HOME_OUTLINED, selected_icon=ft.Icons.HOME,
                label="Ana Sayfa"),
            ft.NavigationRailDestination(
                icon=ft.Icons.IMAGE_SEARCH_OUTLINED, selected_icon=ft.Icons.IMAGE_SEARCH,
                label="Tanıma"),
            ft.NavigationRailDestination(
                icon=ft.Icons.LIBRARY_BOOKS_OUTLINED, selected_icon=ft.Icons.LIBRARY_BOOKS,
                label="Kütüphane"),
            ft.NavigationRailDestination(
                icon=ft.Icons.INFO_OUTLINED, selected_icon=ft.Icons.INFO,
                label="Hakkında"),
        ],
        on_change=on_nav_change,
        expand=True,
    )

    # Logo + Rail + Theme Toggle
    sidebar = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Icon(ft.Icons.DIRECTIONS_CAR, size=34, color=C_TEXT),
                alignment=ft.Alignment(0, 0), height=80,
                gradient=ft.LinearGradient(
                    begin=ft.Alignment(0, -1),
                    end=ft.Alignment(0, 1),
                    colors=[C_ACCENT_DARK, "transparent"]
                ),
            ),
            nav_rail,
            ft.Container(expand=True),
            theme_btn,
            ft.Container(height=15),
        ]),
        width=85,
        bgcolor=C_SURFACE,
    )

    main_layout = ft.Row([
        sidebar,
        ft.VerticalDivider(width=1, color=C_BORDER),
        content_area,
    ], expand=True, spacing=0)

    splash_layout = ft.Container(
        content=ft.Column([
            ft.ProgressRing(width=64, height=64, color=C_ACCENT, stroke_width=6),
            ft.Text("Otomobil Tanıma Sistemi Başlatılıyor", size=24, weight=ft.FontWeight.BOLD, color=C_TEXT),
            ft.Text("Yapay zeka analiz motoru ve veri seti belleğe alınıyor, lütfen bekleyin...", size=14, color=C_TEXT2),
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20),
        alignment=ft.Alignment.CENTER,
        expand=True,
    )

    page.add(splash_layout)
    page.update()

    def _warmup_model():
        nonlocal predictor
        try:
            p = load_predictor()
            if p is not None:
                predictor = p
                # İlk tahmin (dummy) kaldırıldı. TensorFlow'un yan Thread'de donmasını önlüyoruz.
        except Exception as e:
            print(f"Bilinmeyen Yükleme Hatası: {e}")
        finally:
            predictor_loaded["value"] = True

            # Uygulama açılışını ana event loop üzerinden güncelle (thread-safe)
            async def _show_main():
                page.controls.clear()
                page.add(main_layout)
                page.update()

            page.run_task(_show_main)

    threading.Thread(target=_warmup_model, daemon=True).start()


ft.run(main)