# Veri Dosyaları

Ham veri (UCI Online Retail) depoya eklenmemiştir; notebook'lar veriyi `ucimlrepo` ile doğrudan indirir. Bu klasörde analiz sırasında üretilen tablolar ve eğitilmiş model bulunur.

| Dosya | Üreten Notebook | İçerik |
|-------|-----------------|--------|
| `rfm_ozgun.csv` | 02_rfm | Müşteri bazında Recency, Frequency, Monetary ve ilk alışverişten bu yana geçen gün (4.333 müşteri) |
| `rfm_olcekli.csv` | 02_rfm | log1p dönüşümü ve StandardScaler uygulanmış RFM değerleri |
| `kume_sonuclari.csv` | 03_kmeans | K-Means (k=4) küme etiketli müşteri tablosu |
| `segment_sonuclari.csv` | 06_segment_profilleme | Küme etiketlerine segment adı (VIP, Sadık, Yeni/Potansiyel, Uykuda) eklenmiş tablo |
| `segment_gecisleri.csv` | 07_segment_gecisi | 1. dönemdeki her müşterinin 1. ve 2. dönem segmenti |
| `gecis_matrisi.csv` | 07_segment_gecisi | Segmentler arası geçiş oranları (%) |
| `churn_sonuclari.csv` | 08_churn_tahmini | Müşteri bazında özellikler, churn olasılığı, risk seviyesi ve segment |
| `churn_model.pkl` | 08_churn_tahmini | Eğitilmiş Random Forest modeli (scikit-learn 1.6.1, `joblib` ile kaydedildi) |
