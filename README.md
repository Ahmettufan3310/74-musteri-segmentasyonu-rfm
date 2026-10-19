# 🛒 Müşteri Segmentasyonu — RFM & Kümeleme Motoru

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Veri%20İşleme-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Arayüz-FF4B4B?logo=streamlit&logoColor=white)
![Durum](https://img.shields.io/badge/Durum-Geliştiriliyor-yellow)

Bu sistem; e-ticaret ve perakende işlem kayıtlarını analiz ederek müşterileri satın alma davranışlarına göre otomatik segmentlere ayıran, her segment için pazarlama stratejisi öneren **uçtan uca bir makine öğrenmesi karar destek sistemidir**. Kullanıcı bir CSV dosyası yükler; sistem veriyi temizler, RFM özelliklerini çıkarır, müşterileri kümeler ve sonuçları etkileşimli bir panelde sunar.

## 📑 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Öne Çıkan Özellikler](#-öne-çıkan-özellikler)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Veri Seti ve Temizleme](#-veri-seti-ve-temizleme)
- [RFM ve Segmentler](#-rfm-ve-segmentler)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Proje Durumu](#-proje-durumu)
- [Ekip](#-ekip)
- [Kaynaklar](#-kaynaklar)

## 📌 Proje Hakkında

E-ticaret ve perakende sektöründe tüm müşterilere tek tip pazarlama uygulamak verimsizdir. Bu proje, bir işletmenin geçmiş satış verilerini analiz ederek müşterileri **benzer alışveriş davranışına sahip homojen gruplara** ayırır ve her grup için uygun pazarlama stratejisi önerir.

Amaç; yalnızca bir kümeleme algoritması çalıştırmak değil, verinin kurallı biçimde işlenmesi, RFM değerlerinin doğrulanması, model sonuçlarının uygun metriklerle değerlendirilmesi ve sonuçların anlaşılır biçimde sunulmasıyla **güvenilir, tekrarlanabilir bir analiz hattı** kurmaktır.

## 🚀 Öne Çıkan Özellikler

- **Kurallı Veri Temizleme:** Eksik müşteri kaydı, iptal/iade faturaları, hatalı adet–fiyat ve ürün dışı kalemlerin kural bazında ayıklanması; her adımda etkilenen kayıt sayısının raporlanması.
- **RFM Özellik Mühendisliği:** Her müşteri için Recency, Frequency ve Monetary değerlerinin hesaplanması.
- **Çoklu Kümeleme Yaklaşımı:** K-Means (temel model) ile birlikte Hiyerarşik Kümeleme ve DBSCAN'in uygulanıp karşılaştırılması.
- **Objektif Model Değerlendirme:** Elbow yöntemi, Silhouette Skoru ve Davies-Bouldin Endeksi ile optimum küme sayısının ve en uygun modelin belirlenmesi.
- **Boyut İndirgeme ve Görselleştirme:** PCA ile segmentlerin 2B/3B görselleştirilmesi.
- **Kural Tabanlı Segment Profilleme:** Kümelerin iş dünyası segmentlerine (VIP, Sadık, Riskli…) dönüştürülmesi ve pazarlama önerileri.
- **Etkileşimli Arayüz:** CSV yükleme, müşteri sorgulama ve sonuç indirme sunan Streamlit paneli.

## 🧭 Sistem Mimarisi

```
CSV  →  Veri Temizleme  →  RFM Çıkarımı  →  Ön İşleme  →  Kümeleme  →  Değerlendirme  →  PCA  →  Segment Profili  →  Streamlit
                                            (log + scale)   (3 algoritma)  (2 metrik)              (strateji önerileri)
```

| Katman | Açıklama |
|--------|----------|
| **Veri Temizleme** | Eksik, iade/iptal, hatalı ve ürün dışı kayıtların kural bazında ayıklanması |
| **RFM Çıkarımı** | Müşteri bazında Recency, Frequency, Monetary hesabı |
| **Ön İşleme** | Çarpık dağılımların `log1p` ile düzeltilmesi, `StandardScaler` ile ölçeklendirme |
| **Kümeleme** | K-Means, Hiyerarşik Kümeleme, DBSCAN |
| **Değerlendirme** | Elbow, Silhouette, Davies-Bouldin ile model kıyası |
| **Görselleştirme** | PCA ile boyut indirgeme ve küme grafikleri |
| **Profilleme** | Kümelerin iş segmentlerine dönüştürülmesi ve strateji önerileri |
| **Arayüz** | Streamlit tabanlı etkileşimli panel |

## 📊 Veri Seti ve Temizleme

[UCI Machine Learning Repository — Online Retail](https://doi.org/10.24432/C5BW33) veri seti kullanılır. İngiltere merkezli bir çevrimiçi perakende firmasının yaklaşık bir yıllık (2010–2011) işlem kayıtlarını içerir.

| Temizleme Adımı | Çıkarılan | Kalan Satır |
|-----------------|:---------:|:-----------:|
| Başlangıç (ham veri) | — | 541.909 |
| Müşteri numarası boş kayıtlar | 135.080 | — |
| İptal/iade faturaları + sıfır/negatif adet–fiyat | — | 397.884 |
| Ürün dışı kodlar (POST, M, C2, DOT, BANK CHARGES, PADS) | 1.547 | 396.337 |
| Aynı gün tamamı iptal edilen çok büyük siparişlerin asıl satırları | 4 | 396.333 |
| Birebir yinelenen (kopya) satırlar | 5.187 | 391.146 |
| **Analize hazır veri** | — | **391.146 satır / 4.333 müşteri** |

> Veri koddan doğrudan indirildiği için depoya ham veri dosyası eklenmemiştir; herkes aynı kodu çalıştırarak aynı sonucu üretebilir.

## 🏷️ RFM ve Segmentler

**RFM Tanımları**

| Metrik | Tanım |
|--------|-------|
| **Recency (R)** | Referans tarih ile müşterinin son alışverişi arasındaki gün sayısı |
| **Frequency (F)** | Müşterinin benzersiz fatura (sipariş) sayısı |
| **Monetary (M)** | Müşterinin toplam ürün harcaması (Quantity × UnitPrice) |

**Hedeflenen Segmentler** (küme profillerine ve P33/P67 eşiklerine göre kural tabanlı adlandırılır)

| Segment | Profil |
|---------|--------|
| 🏆 Şampiyonlar / VIP | Yakın zamanda, sık ve çok harcayan müşteriler |
| 💙 Sadık Müşteriler | Düzenli gelen, harcaması iyi müşteriler |
| 🌱 Potansiyel Sadıklar | Yakın zamanda gelmiş, sadık olmaya aday müşteriler |
| ⚠️ Riskli / Terk Etmek Üzere | Eskiden değerli, uzun süredir gelmeyen müşteriler |
| 😴 Uykudaki Müşteriler | Çok uzun süredir gelmeyen, az aktif müşteriler |

## 📁 Proje Yapısı

```
74-musteri-segmentasyonu-rfm/
├── notebooks/
│   ├── 01_veri_eda.ipynb        # Veri temizleme ve keşifçi veri analizi
│   ├── 02_rfm.ipynb             # RFM hesabı ve ön işleme
│   └── ...                      # Kümeleme, PCA (geliştiriliyor)
├── data/
│   └── rfm_olcekli.csv          # Ön işlenmiş RFM tablosu
├── app/
│   └── app.py                   # Streamlit arayüzü (geliştiriliyor)
├── requirements.txt
└── README.md
```

## ⚙️ Kurulum ve Çalıştırma

```bash
# Depoyu klonla
git clone https://github.com/Ahmettufan3310/74-musteri-segmentasyonu-rfm.git
cd 74-musteri-segmentasyonu-rfm

# Gerekli kütüphaneleri kur
pip install -r requirements.txt
```

Analiz defterleri `notebooks/` klasöründedir ve Google Colab veya Jupyter ile çalıştırılabilir. Streamlit arayüzü tamamlandığında:

```bash
streamlit run app/app.py
```

**Kullanılan Teknolojiler:** `Python` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `matplotlib` · `seaborn` · `Streamlit`

## 📈 Proje Durumu

| İş Paketi | Açıklama | Durum |
|-----------|----------|:-----:|
| İP1 | Literatür taraması ve veri seti temini | ✅ Tamamlandı |
| İP2 | Keşifçi veri analizi ve veri temizleme | ✅ Tamamlandı |
| İP3 | RFM hesabı ve ön işleme | 🔄 Devam ediyor |
| İP4 | K-Means ve küme sayısı belirleme | ⏳ Planlandı |
| İP5 | Hiyerarşik Kümeleme, DBSCAN ve karşılaştırma | ⏳ Planlandı |
| İP6 | PCA ve görselleştirme | ⏳ Planlandı |
| İP7 | Segment profilleme ve strateji önerileri | ⏳ Planlandı |
| İP8 | Streamlit arayüzü ve raporlama | ⏳ Planlandı |

## 👥 Ekip

| Üye | Sorumluluk Alanı |
|-----|------------------|
| **Ahmet Furkan Tufan** | K-Means modeli, segment profilleme |
| **Mehmetali Murt** | Veri temini, veri temizleme, alternatif kümeleme modelleri |
| **Muhammed Yılmaz** | RFM ve ön işleme, PCA görselleştirme, Streamlit arayüzü |

## 📚 Kaynaklar

1. Chen, D., Sain, S. L. & Guo, K. (2012). *Data Mining for the Online Retail Industry: A Case Study of RFM Model-Based Customer Segmentation.* Journal of Database Marketing & Customer Strategy Management, 19(3), 197–208.
2. Hughes, A. M. (1994). *Strategic Database Marketing.* Chicago: Probus Publishing.
3. MacQueen, J. B. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.* Proc. Fifth Berkeley Symposium, 1, 281–297.
4. Chen, D. (2015). *Online Retail [Dataset].* UCI Machine Learning Repository.
