# 🛒 Müşteri Segmentasyonu — RFM & Kümeleme Motoru

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Veri%20İşleme-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Arayüz-FF4B4B?logo=streamlit&logoColor=white)
![Durum](https://img.shields.io/badge/Durum-Geliştiriliyor-yellow)

Bu sistem; e-ticaret ve perakende işlem kayıtlarını analiz ederek müşterileri satın alma davranışlarına göre otomatik segmentlere ayıran, her segment için pazarlama stratejisi öneren **uçtan uca bir makine öğrenmesi karar destek sistemidir**. Kullanıcı bir CSV dosyası yükler; sistem veriyi temizler, RFM özelliklerini çıkarır, müşterileri kümeler ve sonuçları etkileşimli bir panelde sunar.

**🌐 Canlı Uygulama:** [musteri-segmentasyonu-74.streamlit.app](https://musteri-segmentasyonu-74.streamlit.app)

## 📑 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Öne Çıkan Özellikler](#-öne-çıkan-özellikler)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Veri Seti ve Temizleme](#-veri-seti-ve-temizleme)
- [RFM ve Segmentler](#-rfm-ve-segmentler)
- [Kümeleme Sonuçları](#-kümeleme-sonuçları)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Sınırlılıklar ve Gelecek Çalışmalar](#-sınırlılıklar-ve-gelecek-çalışmalar)
- [Proje Durumu](#-proje-durumu)
- [Ekip](#-ekip)
- [Kaynaklar](#-kaynaklar)
- [Lisans](#-lisans)

## 📌 Proje Hakkında

E-ticaret ve perakende sektöründe tüm müşterilere tek tip pazarlama uygulamak verimsizdir. Bu proje, bir işletmenin geçmiş satış verilerini analiz ederek müşterileri **benzer alışveriş davranışına sahip homojen gruplara** ayırır ve her grup için uygun pazarlama stratejisi önerir.

Amaç; yalnızca bir kümeleme algoritması çalıştırmak değil, verinin kurallı biçimde işlenmesi, RFM değerlerinin doğrulanması, model sonuçlarının uygun metriklerle değerlendirilmesi ve sonuçların anlaşılır biçimde sunulmasıyla **güvenilir, tekrarlanabilir bir analiz hattı** kurmaktır.

## 🚀 Öne Çıkan Özellikler

- **Kurallı Veri Temizleme:** Eksik müşteri kaydı, iptal/iade faturaları, hatalı adet–fiyat ve ürün dışı kalemlerin kural bazında ayıklanması; her adımda etkilenen kayıt sayısının raporlanması.
- **RFM Özellik Mühendisliği:** Her müşteri için Recency, Frequency ve Monetary değerlerinin hesaplanması.
- **Çoklu Kümeleme Yaklaşımı:** K-Means (temel model) ile birlikte Hiyerarşik Kümeleme ve DBSCAN'in uygulanıp karşılaştırılması.
- **Objektif Model Değerlendirme:** Elbow yöntemi, Silhouette Skoru ve Davies-Bouldin Endeksi ile optimum küme sayısının ve en uygun modelin belirlenmesi.
- **Boyut İndirgeme ve Görselleştirme:** PCA ile segmentlerin iki boyutlu düzlemde görselleştirilmesi.
- **Kural Tabanlı Segment Profilleme:** Kümelerin iş dünyası segmentlerine (VIP, Sadık, Yeni/Potansiyel, Uykuda) dönüştürülmesi ve pazarlama önerileri.
- **Segment Geçişi Analizi:** Verinin iki döneme bölünerek müşterilerin segmentler arasındaki hareketinin (ör. Sadık → Uykuda) incelenmesi.
- **Kayıp Müşteri (Churn) Tahmini:** Random Forest ile alışverişi bırakma riski yüksek müşterilerin önceden belirlenmesi.
- **Etkileşimli Arayüz:** CSV yükleme, müşteri sorgulama (segment ve churn riski) ve sonuç indirme sunan Streamlit paneli.

## 🧭 Sistem Mimarisi

```
CSV  →  Veri Temizleme  →  RFM Çıkarımı  →  Ön İşleme  →  Kümeleme  →  Değerlendirme  →  PCA  →  Segment Profili  →  Segment Geçişi  →  Churn Tahmini  →  Streamlit
                                            (log + scale)   (3 algoritma)  (2 metrik)              (strateji önerileri)   (2 dönem)        (Random Forest)
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
| **Segment Geçişi** | İki dönem arasında müşterilerin segment değişimlerinin incelenmesi |
| **Churn Tahmini** | Alışverişi bırakma riski taşıyan müşterilerin sınıflandırma modeliyle tahmini |
| **Arayüz** | Streamlit tabanlı etkileşimli panel |

## 📊 Veri Seti ve Temizleme

[UCI Machine Learning Repository — Online Retail](https://doi.org/10.24432/C5BW33) veri seti kullanılır. İngiltere merkezli bir çevrimiçi perakende firmasının yaklaşık bir yıllık (2010–2011) işlem kayıtlarını içerir.

| Temizleme Adımı | Çıkarılan | Kalan Satır |
|-----------------|:---------:|:-----------:|
| Başlangıç (ham veri) | — | 541.909 |
| Müşteri numarası boş kayıtlar | 135.080 | 406.829 |
| İptal/iade faturaları + sıfır/negatif adet–fiyat | 8.945 | 397.884 |
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

**Belirlenen Segmentler** (kümelerin özgün ölçekteki RFM profillerine göre kural tabanlı adlandırılmıştır)

| Segment | Profil |
|---------|--------|
| 🏆 Şampiyonlar / VIP | Yakın zamanda gelmiş, sık alışveriş yapan ve en çok harcayan müşteriler |
| 💙 Sadık / Düzenli Müşteriler | Düzenli gelen, orta seviye harcayan güvenilir müşteriler |
| 🌱 Yeni / Potansiyel Müşteriler | Yakın zamanda gelmiş ancak henüz az alışveriş yapmış müşteriler |
| 😴 Uykudaki Müşteriler | Uzun süredir alışveriş yapmayan, tek siparişli müşteriler |

## 📈 Kümeleme Sonuçları

Uygun küme sayısı, k=2 ile k=10 arasında **Elbow yöntemi** ve **Silhouette Skoru** birlikte değerlendirilerek belirlenmiştir. Silhouette en yüksek değerine k=2'de ulaşsa da (0,433), iki küme anlamlı pazarlama segmentleri üretmediğinden; istatistiksel skor ile iş açısından yorumlanabilirlik birlikte gözetilerek **k=4** seçilmiştir.

**Model metrikleri:** Silhouette Skoru **0,337** · Davies-Bouldin Endeksi **1,012** (düşük değer daha iyi ayrışma gösterir).

| Segment | Recency (gün) | Frequency | Monetary (£) | Müşteri Sayısı |
|---------|:-------------:|:---------:|:-------------:|:--------------:|
| 🏆 Şampiyonlar / VIP | 8 | 10 | 3.623,5 | 714 |
| 💙 Sadık / Düzenli Müşteriler | 54 | 4 | 1.320,2 | 1.187 |
| 🌱 Yeni / Potansiyel Müşteriler | 18 | 2 | 450,3 | 837 |
| 😴 Uykudaki Müşteriler | 179 | 1 | 292,5 | 1.595 |

> En kalabalık grubun uykudaki müşteriler (1.595) olması, geri kazanım kampanyalarına öncelik verilmesi gerektiğini göstermektedir. Değerler her kümenin medyanıdır.

**Algoritma Karşılaştırması**

Aynı RFM tablosu üzerinde üç kümeleme algoritması uygulanıp Silhouette ve Davies-Bouldin metrikleriyle karşılaştırılmıştır. K-Means, hem en yüksek Silhouette hem de en düşük Davies-Bouldin değerleriyle bu veri seti için en uygun model olarak belirlenmiştir.

| Algoritma | Küme Sayısı | Silhouette | Davies-Bouldin |
|-----------|:-----------:|:----------:|:--------------:|
| **K-Means** | 4 | **0,337** | **1,012** |
| Hiyerarşik Kümeleme | 4 | 0,291 | 1,036 |
| DBSCAN | 2 | 0,296 | 1,062 |

> DBSCAN'in yalnızca iki geniş küme üretmesi, RFM verisinin keskin yoğunluk sınırları içermeyen sürekli yapısından kaynaklanmaktadır.
>
> **Not:** DBSCAN'in Silhouette ve Davies-Bouldin skorları, gürültü olarak etiketlenen 75 müşteri (%1,7) hariç tutularak hesaplanmıştır. K-Means ve Hiyerarşik Kümeleme skorları ise tüm 4.333 müşteri üzerinden hesaplandığından, DBSCAN ile karşılaştırma tam olarak aynı veri kümesi üzerinde yapılmamıştır.

**Görselleştirme (PCA)**

Üç boyutlu RFM uzayı, PCA ile iki bileşene indirgenerek segmentler tek bir grafikte görselleştirilmiştir. İlk iki bileşen toplam varyansın **%94'ünü** (PC1 %75,3 + PC2 %18,7) açıklamaktadır; bu, iki boyutlu görünümün veriyi büyük ölçüde temsil ettiğini gösterir. Grafikte dört segmentin birbirinden ayrı bölgelerde toplanması, kümelemenin anlamlılığını görsel olarak doğrulamaktadır.

**Segment Profilleri ve Ciro Payı**

Segmentlerin müşteri sayısı içindeki payı ile toplam ciro içindeki payı karşılaştırıldığında, değerin birkaç segmentte yoğunlaştığı görülmektedir:

| Segment | Müşteri Oranı | Ciro Payı |
|---------|:-------------:|:---------:|
| 🏆 Şampiyonlar / VIP | %16,5 | **%65,1** |
| 💙 Sadık / Düzenli Müşteriler | %27,4 | %23,3 |
| 🌱 Yeni / Potansiyel Müşteriler | %19,3 | %5,2 |
| 😴 Uykudaki Müşteriler | %36,8 | %6,4 |

> VIP müşteriler, müşteri tabanının yalnızca %16,5'ini oluşturmasına rağmen toplam cironun %65'ini sağlamaktadır. VIP ve sadık müşteriler birlikte müşterilerin %44'ü, cironun ise %88'idir.

**Pazarlama Önerileri**

| Segment | Önerilen Aksiyon |
|---------|------------------|
| 🏆 Şampiyonlar / VIP | Sadakat programı, özel indirimler ve yeni ürünlere erken erişim; bu müşterileri elde tutmak birinci önceliktir. |
| 💙 Sadık / Düzenli Müşteriler | Çapraz satış ve üst segment ürün önerileri; düzenli kampanyalarla VIP seviyesine taşıma. |
| 🌱 Yeni / Potansiyel Müşteriler | Hoş geldin kampanyası ve ikinci alışverişe özel indirim; tekrar alım alışkanlığı kazandırma. |
| 😴 Uykudaki Müşteriler | Geri kazanım e-postası ve kişisel indirim kuponu; tepki vermeyenler için düşük maliyetli iletişim. |

**Segment Geçişi Analizi**

Veri iki döneme ayrılmıştır: **1. dönem** Aralık 2010 – Mayıs 2011 (2.715 müşteri), **2. dönem** Haziran – Aralık 2011 (3.540 müşteri). İki dönemin karşılaştırılabilir olması için ölçekleyici ve K-Means modeli 1. dönemde eğitilmiş, aynı model 2. döneme uygulanmıştır. Tabloda satırlar 1. dönemdeki segmenti, sütunlar 2. dönemdeki durumu gösterir (her satır %100):

| 1. Dönem \ 2. Dönem | VIP | Sadık | Yeni / Potansiyel | Uykuda | Alışveriş yok |
|---------|:---:|:-----:|:-----------------:|:------:|:-------------:|
| 🏆 VIP | **%72,6** | %16,5 | %4,1 | %3,9 | %2,8 |
| 💙 Sadık | %22,1 | %32,7 | %14,4 | %12,3 | %18,6 |
| 🌱 Yeni / Potansiyel | %10,7 | %23,0 | %19,5 | %23,1 | %23,7 |
| 😴 Uykuda | %4,6 | %12,9 | %14,4 | %20,8 | **%47,2** |

> VIP müşterilerin %72,6'sı segmentini korurken, uykudaki müşterilerin %47,2'si 2. dönemde hiç alışveriş yapmamıştır. Sadık müşterilerin %22,1'inin VIP'e yükselmesi, sadık segmente yönelik kampanyaların potansiyelini göstermektedir. 1. dönemdeki müşterilerin %29'u (793 kişi) 2. dönemde geri dönmemiş, 2. dönemde ise 1.618 yeni müşteri kazanılmıştır.

**Kayıp Müşteri (Churn) Tahmini**

Veri 9 Eylül 2011 tarihinden ikiye ayrılmıştır: müşteri özellikleri ilk 9 aydan çıkarılmış, son 3 ayda hiç alışveriş yapmayan müşteriler **churn** olarak etiketlenmiştir (3.359 müşterinin %43'ü). RFM değerlerine ek olarak ürün çeşitliliği, ortalama sepet tutarı ve müşteri süresi özellikleri kullanılmıştır. Müşteriler, churn oranı korunarak rastgele %75 eğitim (2.519 müşteri) / %25 test (840 müşteri) olarak ayrılmıştır (`stratify`, `random_state=42`). Aşağıdaki metrikler test kümesindeki 840 müşteri üzerinden hesaplanmıştır; Random Forest için karışıklık matrisi: 273 doğru yakalanan, 87 kaçırılan churn, 185 yanlış alarm, 295 doğru "devam".

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| Lojistik Regresyon | 0,670 | 0,587 | 0,778 | 0,669 | 0,735 |
| **Random Forest** (max_depth=5) | **0,676** | 0,596 | 0,758 | 0,667 | **0,744** |

**Duyarlılık analizi:** 60, 90 ve 120 günlük churn tanımları denenmiş; ROC-AUC 0,731–0,752 aralığında kalmış, yani sonuç seçilen pencereye fazla duyarlı değildir. 60 gün düzenli ama seyrek alışveriş yapan müşterileri de kayıp saydığı, 120 gün ise kaybı geç fark ettirdiği için işletme açısından dengeli olan **90 gün** kullanılmıştır.

**Model ayarı ve çapraz doğrulama:** Ağaç derinliği, test verisine dokunmadan yalnızca eğitim verisi üzerinde **5 katlı çapraz doğrulama** ile değerlendirilmiştir:

| Model | CV ROC-AUC (ort. ± std) |
|-------|:-----------------------:|
| Random Forest, max_depth=3 | 0,747 ± 0,013 |
| **Random Forest, max_depth=5** | **0,745 ± 0,017** |
| Random Forest, max_depth=8 | 0,740 ± 0,020 |
| Random Forest, max_depth=12 | 0,729 ± 0,019 |
| Random Forest, sınırsız | 0,719 ± 0,021 |
| Lojistik Regresyon | 0,747 ± 0,014 |

Derinlik arttıkça skor düşmekte, bu da aşırı öğrenmeyi göstermektedir. max_depth=3 ile 5 arasındaki fark standart sapmanın çok altında olduğundan iki değer pratikte eşdeğerdir; max_depth=5 korunmuştur. Çapraz doğrulama skoru (0,745) ile test skorunun (0,744) neredeyse aynı olması, test sonucunun şansa bağlı olmadığını göstermektedir. Lojistik Regresyon da benzer başarıya ulaşmış; Random Forest, özellik önemlerini yorumlamaya imkân verdiği için tercih edilmiştir. Modelde en etkili özellikler sırasıyla Frequency, Monetary ve Recency'dir.

Eğitilen model, son veri tarihine (9 Aralık 2011) göre hesaplanan güncel özelliklerle tüm müşterilere uygulanmış ve her müşteriye bir **churn risk puanı** ile risk seviyesi (Düşük < 0,4 ≤ Orta < 0,7 ≤ Yüksek) atanmıştır. Bu puanlar müşterileri risk sırasına koymak içindir; kalibre edilmiş kesin olasılıklar değildir ve ileriye dönük olarak ayrıca doğrulanmamıştır.

| Segment | Düşük Risk | Orta Risk | Yüksek Risk |
|---------|:----------:|:---------:|:-----------:|
| 🏆 VIP | **%99,4** | %0,6 | %0,0 |
| 💙 Sadık | %68,4 | %31,1 | %0,5 |
| 🌱 Yeni / Potansiyel | %17,4 | %82,0 | %0,6 |
| 😴 Uykuda | %0,1 | %62,9 | **%37,0** |

> Sonuçlar segment geçişi analiziyle tutarlıdır: VIP müşteriler neredeyse tamamen düşük riskli, uykudaki müşteriler ise en riskli gruptur. Toplam 601 müşterinin risk puanı yüksektir ve bunların 590'ı Uykuda segmentindedir.

## 📁 Proje Yapısı

```
74-musteri-segmentasyonu-rfm/
├── notebooks/
│   ├── 01_veri_eda.ipynb        # Veri temizleme ve keşifçi veri analizi
│   ├── 02_rfm.ipynb             # RFM hesabı ve ön işleme
│   ├── 03_kmeans.ipynb          # K-Means kümeleme ve küme sayısı belirleme
│   ├── 04_alternatif_kumeleme.ipynb  # Hiyerarşik Kümeleme, DBSCAN ve karşılaştırma
│   ├── 05_pca_gorsellestirme.ipynb    # PCA ile boyut indirgeme ve segment görselleştirme
│   ├── 06_segment_profilleme.ipynb    # Segment profilleri ve pazarlama önerileri
│   ├── 07_segment_gecisi.ipynb        # İki dönem arası segment geçişleri
│   └── 08_churn_tahmini.ipynb         # Kayıp müşteri (churn) tahmini ve deneyler
├── data/
│   ├── rfm_ozgun.csv            # Özgün ölçekli RFM tablosu
│   ├── rfm_olcekli.csv          # Ölçeklenmiş RFM tablosu
│   ├── kume_sonuclari.csv       # Küme etiketli müşteri tablosu
│   ├── segment_sonuclari.csv    # Segment adı eklenmiş müşteri tablosu
│   ├── segment_gecisleri.csv    # Müşteri bazında dönemler arası segment geçişleri
│   ├── gecis_matrisi.csv        # Segment geçiş oranları (%)
│   ├── churn_sonuclari.csv      # Müşteri bazında churn olasılığı ve risk seviyesi
│   ├── churn_model.pkl          # Eğitilmiş Random Forest modeli (scikit-learn 1.6.1)
│   └── ornek_islemler.csv       # Arayüzde denemek için örnek işlem dosyası
├── app/
│   └── app.py                   # Streamlit arayüzü
├── .streamlit/
│   └── config.toml              # Arayüz teması
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

Analiz defterleri `notebooks/` klasöründedir ve Google Colab veya Jupyter ile çalıştırılabilir.

**Notebook çalıştırma sırası:** `01_veri_eda` → `02_rfm` → `03_kmeans` → `04_alternatif_kumeleme` → `05_pca_gorsellestirme` → `06_segment_profilleme` → `07_segment_gecisi` → `08_churn_tahmini`

- Ham veri depoya eklenmemiştir; notebook'lar UCI Online Retail verisini `ucimlrepo` ile indirir, bu nedenle **internet bağlantısı gerekir**.
- 03 ve sonraki notebook'lar, önceki adımların ürettiği tabloları GitHub üzerindeki `data/` klasöründen okur.
- Notebook'lar ürettikleri CSV ve model dosyalarını çalıştıkları klasöre kaydeder. Bu dosyalar **elle `data/` klasörüne taşınmıştır**; notebook'ları yeniden çalıştıran biri, oluşan dosyaları `data/` klasörüne kopyalamalıdır. Dosyaların hangi notebook'tan üretildiği [`data/README.md`](data/README.md) içinde listelenmiştir.

Streamlit arayüzünü yerelde çalıştırmak için (Python 3.12 önerilir; `requirements.txt` sürümleri sabitlenmiştir):

```bash
streamlit run app/app.py
```

Arayüz ayrıca Streamlit Community Cloud üzerinde yayındadır: **[musteri-segmentasyonu-74.streamlit.app](https://musteri-segmentasyonu-74.streamlit.app)** (Python 3.12). Uygulama bir süre kullanılmazsa uyku moduna geçer; açılışta "wake up" düğmesine basmak yeterlidir.

**Arayüz sayfaları:** Genel Bakış · Segmentler · Segment Geçişi · Churn Riski · Öncelikli Müşteriler · Müşteri Sorgula · Yeni Veri Analizi · Yöntem

**Kullanılan Teknolojiler:** `Python` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `matplotlib` · `joblib` · `Streamlit` · `Plotly`

## 🚧 Sınırlılıklar ve Gelecek Çalışmalar

- **Tek veri seti, tek yıl:** Çalışma, İngiltere merkezli tek bir çevrimiçi perakendecinin bir yıllık verisine dayanmaktadır; sonuçlar farklı sektör, ülke, dönem veya para birimine doğrudan genellenemez.
- **Churn puanlarının doğrulanması:** Model 9 Eylül 2011 öncesi verilerle eğitilip sonraki 90 günde değerlendirilmiştir. Tüm müşterilere verilen güncel puanlar için ileriye dönük bir dönem bulunmadığından bu puanlar doğrulanamamıştır. Birden fazla geçmiş kesim tarihiyle zaman bazlı (rolling) doğrulama gelecek çalışma olarak bırakılmıştır.
- **Test verisinin kullanımı:** Model karşılaştırması ve parametre denemeleri sırasında test kümesi birden fazla kez incelenmiştir. Bu etkiyi azaltmak için parametre seçimi sonradan çapraz doğrulama ile yeniden yapılmış; çapraz doğrulama ve test skorlarının neredeyse aynı çıkması etkinin küçük olduğunu göstermektedir.
- **Kalibrasyon ve eşikler:** Risk eşikleri (0,4 ve 0,7) yorumlamayı kolaylaştırmak için seçilmiştir; olasılıklar kalibre edilmemiştir. Kalibrasyon eğrisi, Brier skoru ve kampanya maliyet–fayda analizi gelecek çalışmalardır.
- **Segmentlerin iş açısından doğrulanması:** Segment adları ve pazarlama önerileri medyan RFM değerlerine dayanmaktadır; gerçek kampanya sonuçlarıyla veya alan uzmanlarıyla doğrulanmamıştır. PCA grafiği görsel bir incelemedir, tek başına küme kalitesini kanıtlamaz.
- **Yeni veriye uygulama:** Arayüzdeki "Yeni Veri Analizi" sayfası, UCI verisiyle eğitilmiş sabit modelleri kullanır; farklı yapıdaki verilerde sonuçlar yanıltıcı olabilir. Bu durumda arayüz kullanıcıyı uyarır.
- **Tekrar üretilebilirlik:** Notebook çıktıları `data/` klasörüne elle taşınmaktadır; tüm adımları tek komutla çalıştıran bir pipeline ve otomatik testler bulunmamaktadır.

## 📈 Proje Durumu

| İş Paketi | Açıklama | Durum |
|-----------|----------|:-----:|
| İP1 | Literatür taraması ve veri seti temini | ✅ Tamamlandı |
| İP2 | Keşifçi veri analizi ve veri temizleme | ✅ Tamamlandı |
| İP3 | RFM hesabı ve ön işleme | ✅ Tamamlandı |
| İP4 | K-Means ve küme sayısı belirleme | ✅ Tamamlandı |
| İP5 | Hiyerarşik Kümeleme, DBSCAN ve karşılaştırma | ✅ Tamamlandı |
| İP6 | PCA ve görselleştirme | ✅ Tamamlandı |
| İP7 | Segment profilleme ve strateji önerileri | ✅ Tamamlandı |
| İP8 | Segment geçişi analizi (iki dönem karşılaştırması) | ✅ Tamamlandı |
| İP9 | Kayıp müşteri (churn) tahmini | ✅ Tamamlandı |
| İP10 | Streamlit arayüzü | ✅ Tamamlandı |
| İP11 | Final rapor ve sunum | 🔄 Devam ediyor |

## 👥 Ekip

| Üye | Sorumluluk Alanı |
|-----|------------------|
| **Ahmet Furkan Tufan** | K-Means modeli, segment profilleme, Streamlit arayüzü |
| **Mehmetali Murt** | Veri temini, veri temizleme, alternatif kümeleme modelleri, churn tahmini, Streamlit arayüzü |
| **Muhammed Yılmaz** | RFM ve ön işleme, PCA görselleştirme, segment geçişi analizi |

## 📚 Kaynaklar

**RFM ve Veri Seti**

1. Hughes, A. M. (1994). *Strategic Database Marketing.* Chicago: Probus Publishing.
2. Chen, D., Sain, S. L. & Guo, K. (2012). *Data Mining for the Online Retail Industry: A Case Study of RFM Model-Based Customer Segmentation Using Data Mining.* Journal of Database Marketing & Customer Strategy Management, 19(3), 197–208.
3. Chen, D. (2015). *Online Retail [Dataset].* UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

**Kümeleme Algoritmaları**

4. MacQueen, J. B. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.* Proc. Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.
5. Arthur, D. & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding.* Proc. 18th ACM-SIAM Symposium on Discrete Algorithms (SODA), 1027–1035.
6. Ward, J. H. (1963). *Hierarchical Grouping to Optimize an Objective Function.* Journal of the American Statistical Association, 58(301), 236–244.
7. Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.* Proc. 2nd Int. Conf. on Knowledge Discovery and Data Mining (KDD-96), 226–231.

**Model Değerlendirme**

8. Thorndike, R. L. (1953). *Who Belongs in the Family?* Psychometrika, 18(4), 267–276. (Elbow yöntemi)
9. Rousseeuw, P. J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.
10. Davies, D. L. & Bouldin, D. W. (1979). *A Cluster Separation Measure.* IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-1(2), 224–227.
11. Fawcett, T. (2006). *An Introduction to ROC Analysis.* Pattern Recognition Letters, 27(8), 861–874.

**Boyut İndirgeme**

12. Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space.* Philosophical Magazine, 2(11), 559–572.
13. Jolliffe, I. T. (2002). *Principal Component Analysis* (2. baskı). New York: Springer.

**Tahmin Modeli**

14. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
15. Cox, D. R. (1958). *The Regression Analysis of Binary Sequences.* Journal of the Royal Statistical Society: Series B, 20(2), 215–242.

**Yazılım Kütüphaneleri**

16. Pedregosa, F. vd. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
17. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proc. 9th Python in Science Conference, 56–61.
18. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering, 9(3), 90–95.
19. Plotly Technologies Inc. (2015). *Collaborative Data Science.* Montréal, QC: Plotly Technologies Inc. https://plot.ly
20. Streamlit Inc. (2019). *Streamlit: A Faster Way to Build and Share Data Apps* [Yazılım]. https://streamlit.io
21. Joblib Development Team. *Joblib: Running Python Functions as Pipeline Jobs* [Yazılım]. https://joblib.readthedocs.io

## 📄 Lisans

Bu proje **MIT Lisansı** altında dağıtılmaktadır. Ayrıntılar için depodaki [LICENSE](LICENSE) dosyasına bakınız.
