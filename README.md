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
- [Kümeleme Sonuçları](#-kümeleme-sonuçları)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
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
- **Boyut İndirgeme ve Görselleştirme:** PCA ile segmentlerin 2B/3B görselleştirilmesi.
- **Kural Tabanlı Segment Profilleme:** Kümelerin iş dünyası segmentlerine (VIP, Sadık, Riskli…) dönüştürülmesi ve pazarlama önerileri.
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

| Segment | Recency (gün) | Frequency | Monetary (TL) | Müşteri Sayısı |
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
│   ├── 07_segment_gecisi.ipynb        # İki dönem arası segment geçişleri (planlanıyor)
│   └── 08_churn_tahmini.ipynb         # Kayıp müşteri tahmini (planlanıyor)
├── data/
│   ├── rfm_ozgun.csv            # Özgün ölçekli RFM tablosu
│   ├── rfm_olcekli.csv          # Ölçeklenmiş RFM tablosu
│   ├── kume_sonuclari.csv       # Küme etiketli müşteri tablosu
│   ├── segment_sonuclari.csv    # Segment adı eklenmiş müşteri tablosu
│   └── segment_gecisleri.csv    # Dönemler arası segment geçişleri (planlanıyor)
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
| İP3 | RFM hesabı ve ön işleme | ✅ Tamamlandı |
| İP4 | K-Means ve küme sayısı belirleme | ✅ Tamamlandı |
| İP5 | Hiyerarşik Kümeleme, DBSCAN ve karşılaştırma | ✅ Tamamlandı |
| İP6 | PCA ve görselleştirme | ✅ Tamamlandı |
| İP7 | Segment profilleme ve strateji önerileri | ✅ Tamamlandı |
| İP8 | Segment geçişi analizi (iki dönem karşılaştırması) | 🔄 Devam ediyor |
| İP9 | Kayıp müşteri (churn) tahmini | ⏳ Planlandı |
| İP10 | Streamlit arayüzü | ⏳ Planlandı |
| İP11 | Final rapor ve sunum | ⏳ Planlandı |

## 👥 Ekip

| Üye | Sorumluluk Alanı |
|-----|------------------|
| **Ahmet Furkan Tufan** | K-Means modeli, segment profilleme, Streamlit arayüzü |
| **Mehmetali Murt** | Veri temini, veri temizleme, alternatif kümeleme modelleri, churn tahmini |
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

**Boyut İndirgeme**

11. Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space.* Philosophical Magazine, 2(11), 559–572.
12. Jolliffe, I. T. (2002). *Principal Component Analysis* (2. baskı). New York: Springer.

**Tahmin Modeli**

13. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.

**Yazılım Kütüphaneleri**

14. Pedregosa, F. vd. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
15. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proc. 9th Python in Science Conference, 56–61.
16. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering, 9(3), 90–95.

## 📄 Lisans

Bu proje **MIT Lisansı** altında dağıtılmaktadır. Ayrıntılar için depodaki [LICENSE](LICENSE) dosyasına bakınız.
