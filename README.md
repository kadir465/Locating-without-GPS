# 🛰️ GPS-Denied Visual-Inertial Navigation & Localization Stack

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org/)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-22314E.svg)](https://docs.ros.org/en/humble/)
[![PX4](https://img.shields.io/badge/PX4-uXRCE--DDS-107C41.svg)](https://docs.px4.io/)
[![Deep Learning](https://img.shields.io/badge/Matcher-SuperPoint%20%2B%20LightGlue-orange.svg)](https://github.com/cvg/LightGlue)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

İnsansız Hava Araçları (İHA) için **GPS karartması (jamming)**, **aldatması (spoofing)** veya **sinyal yokluğunda** yüksek hassasiyetli, gerçek zamanlı seyrüsefer ve mutlak konumlandırma sağlayan **hiyerarşik görsel-eylemsel (Visual-Inertial Odometry / Absolute Georeferenced Localization)** seyrüsefer sistemi.

Sistem, hem **Python** (algoritma doğrulama, simülasyon, video/telemetri testleri) hem de **C++ / ROS 2** (NVIDIA Jetson Orin Nano ve Pixhawk / PX4 donanımında düşük gecikmeli uçuş) mimarisiyle tam uyumlu olarak geliştirilmiştir.

---

## 📌 İçindekiler
- [Proje Genel Bakış](#-proje-genel-bakış)
- [Sistem Mimarisi ve Katmanlar](#-sistem-mimarisi-ve-katmanlar)
- [Temel Özellikler](#-temel-özellikler)
- [Dizin Yapısı](#-dizin-yapısı)
- [Kurulum](#-kurulum)
  - [1. Python Ortamı](#1-python-ortamı)
  - [2. C++ & ROS 2 Ortamı (Jetson Orin / Linux)](#2-c--ros-2-ortamı-jetson-orin--linux)
- [Kullanım ve Test Adımları](#-kullanım-ve-test-adımları)
  - [Adım 1: Uçuş Öncesi Referans Harita Üretimi](#adım-1-uçuş-öncesi-referans-harita-üretimi-katman-0)
  - [Adım 2: Sentetik Uçuş Simülasyonu](#adım-2-sentetik-uçuş-simülasyonu)
  - [Adım 3: Video ve Telemetri (SRT) ile Test & Görselleştirme](#adım-3-video-ve-telemetri-srt-ile-test--görselleştirme)
  - [Adım 4: C++ Yüksek Performanslı Runner (WSL / Orin)](#adım-4-c-yüksek-performanslı-runner-wsl--orin)
  - [Adım 5: Gerçek Donanımda Uçuş (ROS 2 + PX4 uXRCE-DDS)](#adım-5-gerçek-donanımda-uçuş-ros-2--px4-uxrce-dds)
- [Koordinat Sistemleri ve Sensör Füzyonu](#-koordinat-sistemleri-ve-sensör-füzyonu)
- [Failsafe ve Güvenlik Modları](#-failsafe-ve-güvenlik-modları)
- [Katkıda Bulunma ve Lisans](#-katkıda-bulunma-ve-lisans)

---

## 🎯 Proje Genel Bakış

Geleneksel otopilotlar GPS kesildiğinde sadece IMU ile ölü kestirim (Dead Reckoning) yapar. Ancak ivmeölçer ve jiroskop gürültüleri zamanla birikerek (drift) uçağın dakikalar içinde rotadan sapmasına neden olur.

Bu proje sorunu **çok sensörlü ve çok katmanlı füzyon** ile çözer:
1. **Kısa vadeli kaymayı önleme:** Yere bakan monoküler kameradan yüksek frekansta (30 Hz) optik akış çıkarılarak yatay hız kestirilir.
2. **Uzun vadeli kaymayı sıfırlama:** Uçuş öncesi indirilen uydu haritası ile canlı kamera görüntüsü derin öğrenme (SuperPoint + LightGlue) ile periyodik olarak (~1 Hz) eşleştirilerek **metre altı / santimetre mertebesinde mutlak coğrafi konum** bulunur.
3. **Zaman gecikmeli füzyon:** Ağır derin öğrenme çıkarımlarının gecikmesi, **Gecikmeli ESKF Halka Tamponu (Ring Buffer Replay)** ile telafi edilir.

---

## 🏗 Sistem Mimarisi ve Katmanlar

```mermaid
flowchart TD
    subgraph PreFlight ["Katman 0: Ön Hazırlık"]
        MB[Mapbox Satellite API] --> MapPipeline[preflight_map_pipeline.py]
        MapPipeline --> GeoTIFF["GeoTIFF Referans Harita (Grayscale + CLAHE)"]
    end

    subgraph Sensors ["Sensör Girişleri"]
        IMU["Pixhawk IMU (100 Hz)<br/>[ax, ay, az, gx, gy, gz]"]
        CAM["Aşağı Bakan Kamera (30 Hz)<br/>[Monoküler Görüntü]"]
        BARO["Barometre / Telemetri<br/>[İrtifa / Z]"]
    end

    subgraph Estimator ["Füzyon Motoru (Orin Nano / C++ & Python)"]
        IMU -->|Tahmin Adımı| ESKF["16-Durumlu Gecikmeli ESKF<br/>[p, v, q, b_a, b_g, t_d]"]
        
        CAM -->|Katman 1.5 - 30 Hz| KLT["KLT Optik Akış<br/>(Pseudo-Velocity)"]
        BARO --> KLT
        KLT -->|Hız Düzeltmesi| ESKF

        CAM -->|Katman 2 & 3 - 1 Hz| LG["SuperPoint + LightGlue<br/>+ Multi-Model RANSAC"]
        GeoTIFF -->|Dinamik ROI Kırpma| LG
        LG -->|Gecikmeli Mutlak Poz (td)| RingBuf["Ring Buffer (Tarihsel Geri Sarma)"]
        RingBuf -->|Gecikmeli Ölçüm Güncellemesi| ESKF
    end

    subgraph Output ["Otopilot Entegrasyonu"]
        ESKF -->|NED Dönüşümü| DDS["PX4 uXRCE-DDS Köprüsü<br/>/fmu/in/vehicle_visual_odometry"]
        DDS --> PX4["Pixhawk Otopilot (EKF2)"]
        ESKF --> Status["/advanced_localization/status<br/>(Failsafe & Kovaryans Takibi)"]
    end
```

### Katman Detayları
* **Layer 0 (Pre-Flight Map):** Uçulacak alanın uydu görüntülerinden CLAHE kontrast iyileştirmesi yapılmış, GSD (Ground Sampling Distance) etiketli GeoTIFF harita oluşturulur.
* **Layer 1 (ESKF - 100 Hz):** 16 durumlu `[p (konum 3), v (hız 3), q (oryantasyon 4), b_a (ivmeölçer bias 3), b_g (jiroskop bias 3), t_d (kamera gecikmesi 1)]` Hata Durumlu Kalman Filtresi.
* **Layer 1.5 (KLT Optik Akış - 30 Hz):** Ardışık video kareleri arasındaki piksel yer değiştirmelerinden ve irtifadan yatay hız vektörü üretilir.
* **Layer 2 & 3 (Hassas Harita Eşleme - 1 Hz):** Uçağın tahmin kovaryansına göre haritadan dinamik bir ROI (Region of Interest) kırpılır. SuperPoint ile çıkarılan öznitelikler LightGlue ile eşleştirilir; Homografi ve Temel Matris (Fundamental) testleri ile mutlak koordinat üretilir.

---

## 🚀 Temel Özellikler

- **Tarihsel Gecikme Telafisi (Historical Replay):** Derin öğrenme eşleştirmesi 100-300 ms sürse dahi, zaman damgalı halka tampon sayesinde filtre geçmişe dönüp ölçümü uygular ve günümüze doğru hatasız ilerler.
- **Çift Dil Desteği:** Araştırma, hiperparametre optimizasyonu ve görselleştirme için Python; uçuş donanımı için sıfır-kopyalama (zero-copy) C++ ve Eigen mimarisi.
- **Modern PX4 Entegrasyonu:** MAVROS bağımlılığı olmadan, PX4 v1.14+ yerel standardı olan **Micro XRCE-DDS** üzerinden doğrudan `/fmu/in/vehicle_visual_odometry` beslemesi.
- **Gelişmiş Failsafe Mekanizması:** Görsel eşleşme kaybı, sensör tutarsızlığı ve kovaryans patlaması anında otomatik tespit ve durum bildirimi.
- **Zengin Test ve Kıyaslama (Benchmark):** DJI / GoPro MP4 videoları ve altyazı (SRT) telemetri dosyaları ile çevrimdışı (offline) birebir uçuş simülasyonu.

---

## 📁 Dizin Yapısı

```text
gps-denied-navigation/
├── advanced_localization_cpp/       # Yüksek performanslı C++ / ROS 2 paketi (Jetson Orin için)
│   ├── CMakeLists.txt              # Derleme kuralları ve bağımlılıklar (Eigen3, OpenCV, YAML)
│   ├── package.xml                 # ROS 2 paket manifestosu (px4_msgs, sensor_msgs vb.)
│   ├── include/                    # C++ başlık dosyaları (eskf, ring_buffer, optical_flow vb.)
│   ├── src/                        # C++ kaynak kodları ve ROS 2 düğümü (localization_node.cpp)
│   └── launch/                     # ROS 2 başlatma dosyaları (localization.launch.py)
├── config/                         # Sistem konfigürasyonları
│   ├── system_config.yaml          # Filtre kovaryansları, sensör hızları, gürültü modelleri
│   ├── system_config_video_lightglue.yaml # Video testleri için optimize LightGlue ayarları
│   └── camera_info_dummy.yaml      # Kamera iç parametreleri (intrinsics: fx, fy, cx, cy)
├── docs/                           # Teknik dökümantasyon ve donanım entegrasyon raporları
│   └── pixhawk_orin_cpp_migration_report.md # Pixhawk + Orin Nano uXRCE-DDS göç raporu
├── scripts/                        # Yardımcı araçlar ve test betikleri
│   ├── preflight_map_pipeline.py   # Mapbox API'den uydu haritası indirip GeoTIFF üreten araç
│   ├── run_video_testing.py        # Video (.mp4) ve telemetri (.srt) ile uçuş simülasyonu
│   ├── profile_lightglue_gpu.py    # LightGlue GPU çıkarım gecikmesi profilleyici
│   └── compress_jpg.py             # Harita ve görsel sıkıştırma aracı
├── simulation/                     # Simülasyon ortamı
│   └── synthetic_flight.py         # Yapay 3B yörünge ve IMU/kamera sentetik veri üreteci
├── src/advanced_localization/      # Çekirdek Python paketi
│   ├── eskf.py                     # 16-durumlu Hata Durumlu Kalman Filtresi motoru
│   ├── ring_buffer.py              # Zaman damgalı halka tampon (TimeIndexedRingBuffer)
│   ├── failsafe.py                 # Durum makinesi ve arıza tespit yöneticisi
│   ├── map_store.py                # GeoTIFF harita yönetimi ve dinamik ROI çıkarımı
│   ├── math_utils.py               # Kuaterniyon, rotasyon ve Lie cebiri matematiği
│   ├── types.py                    # Tip tanımlamaları (ImuSample, VisionMeasurement vb.)
│   ├── video_testing.py            # Video test işleme motoru
│   ├── ros2/                       # Python ROS 2 düğümü (navigation_node.py)
│   └── vision/                     # Bilgisayarlı görü katmanları
│       ├── superpoint_layer.py     # SuperPoint öznitelik çıkarımı
│       ├── lightglue_layer.py      # LightGlue derin öğrenme eşleyicisi & RANSAC
│       ├── optical_flow_layer.py   # OpenCV KLT optik akış hız kestirimi
│       └── roi_selector.py         # Dinamik harita arama penceresi seçici
├── tests/                          # Pytest birim ve entegrasyon testleri
├── requirements.txt                # Python bağımlılık listesi
└── pyproject.toml                  # Python proje metaverileri
```

---

## 🛠 Kurulum

### 1. Python Ortamı

Gereksinimler: Python 3.10+, PyTorch (CUDA destekli önerilir), OpenCV.

```bash
# 1. Depoyu klonlayın
git clone https://github.com/kadir465/Locating-without-GPS.git
cd Locating-without-GPS

# 2. Sanal ortam oluşturun ve aktif edin
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Paketi geliştirici modunda kurun
pip install -e .
```

### 2. C++ & ROS 2 Ortamı (Jetson Orin / Linux)

Gereksinimler: Ubuntu 22.04, ROS 2 Humble, OpenCV 4.x, Eigen3, `px4_msgs`.

```bash
# 1. ROS 2 çalışma alanı oluşturun
mkdir -p ~/gps_denied_ws/src
cd ~/gps_denied_ws/src

# 2. Depoyu ve PX4 mesaj tanımlarını çekin
git clone https://github.com/kadir465/Locating-without-GPS.git
git clone https://github.com/PX4/px4_msgs.git

# 3. Derleyin
cd ~/gps_denied_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select advanced_localization_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

## 🎮 Kullanım ve Test Adımları

### Adım 1: Uçuş Öncesi Referans Harita Üretimi (Katman 0)
Uçulacak bölgenin enlem, boylam ve yarıçapını girerek Mapbox üzerinden CLAHE uygulanmış GeoTIFF haritası oluşturun:

```bash
python scripts/preflight_map_pipeline.py \
  --mapbox-token "MAPBOX_TOKENINIZ" \
  --center-lat 41.0151 \
  --center-lon 28.9795 \
  --radius-m 2000 \
  --gsd 0.2 \
  --output deployment/map_preflight_clahe.tif
```

### Adım 2: Sentetik Uçuş Simülasyonu
Filtrenin matematiksel tutarlılığını yapay IMU ve kamera verileriyle test edin:

```bash
python -m simulation.synthetic_flight
# Birim testleri çalıştırmak için:
pytest
```

### Adım 3: Video ve Telemetri (SRT) ile Test & Görselleştirme
Daha önce kaydedilmiş bir İHA uçuş videosunu (.mp4) ve telemetri altyazısını (.srt) harita üzerinde canlı görselleştirerek test edin:

```bash
python scripts/run_video_testing.py \
  --video data/flight_sample.mp4 \
  --srt data/flight_sample.srt \
  --map-path deployment/map_preflight_clahe.tif \
  --map-gsd 0.2 \
  --show-map-window \
  --csv-output outputs/flight_result.csv
```

### Adım 4: C++ Yüksek Performanslı Runner (WSL / Orin)
Kare hızını (FPS) ve bellek profilini C++ motoruyla doğrudan test etmek için:

```bash
./build/advanced_localization_cpp/video_testing_runner \
  --video data/flight_sample.mp4 \
  --srt data/flight_sample.srt \
  --map-path deployment/map_preflight_clahe.tif \
  --map-gsd 0.2 \
  --profile \
  --csv-output outputs/cpp_benchmark.csv
```

### Adım 5: Gerçek Donanımda Uçuş (ROS 2 + PX4 uXRCE-DDS)
1. **Micro XRCE-DDS Agent'ı Başlatın:**
   ```bash
   # Seri port (Telem portu üzerinden):
   MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600
   # veya Ethernet/UDP üzerinden:
   MicroXRCEAgent udp4 -p 8888
   ```

2. **C++ Seyrüsefer Düğümünü Başlatın:**
   ```bash
   source ~/gps_denied_ws/install/setup.bash
   ros2 launch advanced_localization_cpp localization.launch.py \
     config_yaml:=config/system_config.yaml \
     map_path:=deployment/map_preflight_clahe.tif
   ```
   Düğüm otomatik olarak PX4'ün IMU verisini dinleyecek, filtre çıktısını `/fmu/in/vehicle_visual_odometry` konusuna basarak otopilotun EKF2 modülünü besleyecektir.

---

## 📐 Koordinat Sistemleri ve Sensör Füzyonu

Farklı sistemler arasındaki eksen çakışmalarını önlemek için izolasyon adaptörü kullanılır:

| Sistem / Arayüz | Konum Koordinatı | Açısal Gövde Çerçevesi |
| :--- | :--- | :--- |
| **PX4 Otopilot (uXRCE-DDS)** | **NED** (North-East-Down) | **FRD** (Forward-Right-Down) |
| **Algoritma Çekirdeği (ESKF)** | **ENU** (East-North-Up) | **FLU** (Forward-Left-Up) |
| **Kamera Çerçevesi** | Z-İleri, X-Sağa, Y-Aşağı (Optical Standard) |

*Filtre tüm hesaplamalarını robotik standardı olan **ENU** ekseninde yürütür; PX4'e odometri basarken eksenler otomatik olarak **NED** formatına dönüştürülür.*

---

## 🛡 Failsafe ve Güvenlik Modları

Sistem uçuş güvenliğini sağlamak için sürekli durum analizi yapar ve `/advanced_localization/status` üzerinden otopilota bilgi sağlar:

* `VISION_LOSS`: Görsel özelliklerin yetersizliği (örneğin su üstü veya bulut içi uçuş). Bu durumda filtre otomatik olarak KLT Optik Akış ve IMU dead-reckoning moduna geçer.
* `INCONSISTENCY`: Optik akış hızı ile harita eşleşmesi arasında Mahalanobis mesafe kapısını aşan tutarsızlık tespiti. Hatalı eşleşme dışlanır (outlier rejection).
* `HIGH_COVARIANCE`: Kovaryans matrisinin $P_{xy}$ izi eşik değeri aştığında otopilota güven seviyesinin düştüğünü bildirir.

---

## 📚 Referanslar ve Teknolojiler

- **LightGlue:** [LightGlue: Local Feature Matching at Light Speed (ICCV 2023)](https://github.com/cvg/LightGlue)
- **SuperPoint:** [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
- **ESKF Mantığı:** [Joan Solà - Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508)
- **PX4 Micro XRCE-DDS:** [PX4 ROS 2 User Guide](https://docs.px4.io/main/en/ros2/user_guide.html)
