from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Müşteri Segmentasyonu", page_icon="🛒", layout="wide")

# Dosya yolları app.py'nin konumuna göre bulunur (Streamlit Cloud'da da çalışır)
VERI = Path(__file__).resolve().parent.parent / "data"

SEGMENTLER = ["VIP", "Sadık", "Yeni/Potansiyel", "Uykuda"]  # önem sırası
RISKLER = ["Düşük", "Orta", "Yüksek"]
RFM = ["Recency", "Frequency", "Monetary"]
OZELLIKLER = ["Recency", "Frequency", "Monetary", "IlkAlisveris", "UrunCesidi", "OrtSepet"]

# Renkler tüm sayfalarda aynı
SEGMENT_RENK = {"VIP": "#1f4fd1", "Sadık": "#4fa3e0", "Yeni/Potansiyel": "#8e6bd6", "Uykuda": "#9aa0a6"}
RISK_RENK = {"Düşük": "#2e9e6b", "Orta": "#e0a526", "Yüksek": "#d64545"}
KAYIP_RENK = "#4a4f57"

ETIKET = {
    "VIP": "Şampiyonlar / VIP",
    "Sadık": "Sadık / Düzenli Müşteriler",
    "Yeni/Potansiyel": "Yeni / Potansiyel Müşteriler",
    "Uykuda": "Uykudaki Müşteriler",
}

# README'deki "Pazarlama Önerileri" tablosu
ONERI = {
    "VIP": "Sadakat programı, özel indirimler ve yeni ürünlere erken erişim; bu müşterileri elde tutmak birinci önceliktir.",
    "Sadık": "Çapraz satış ve üst segment ürün önerileri; düzenli kampanyalarla VIP seviyesine taşıma.",
    "Yeni/Potansiyel": "Hoş geldin kampanyası ve ikinci alışverişe özel indirim; tekrar alım alışkanlığı kazandırma.",
    "Uykuda": "Geri kazanım e-postası ve kişisel indirim kuponu; tepki vermeyenler için düşük maliyetli iletişim.",
}
# Kartlardaki kısa aksiyon: önerinin ilk cümlesi
KISA_AKSIYON = {seg: metin.split(";")[0] + "." for seg, metin in ONERI.items()}

ONCELIKLI_AKSIYON = ("Kişisel iletişim (telefon veya özel e-posta) ile ulaşın; özel indirim veya ücretsiz kargo "
                     "teklifiyle elde tutmaya çalışın. Beklenen kaybı en yüksek olanlardan başlayın.")

GEREKLI_SUTUNLAR = ["InvoiceNo", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
URUN_DISI = ["POST", "M", "C2", "DOT", "BANK CHARGES", "PADS"]
HATALI = [("581483", "23843"), ("541431", "23166"), ("540815", "21108"), ("556444", "22502")]


# ---------------- Yardımcı fonksiyonlar ----------------

def sayi(x, ondalik=0):
    """Sayıyı Türkçe biçimde yazar: 1234.5 -> 1.234,5"""
    metin = f"{x:,.{ondalik}f}"
    return metin.replace(",", "X").replace(".", ",").replace("X", ".")


def para(x, ondalik=0):
    return "£" + sayi(x, ondalik)


def yuzde(x, ondalik=1):
    return "%" + sayi(x, ondalik)


def grafik_goster(grafik):
    """Grafiği Türkçe sayı biçimiyle gösterir."""
    grafik.update_layout(separators=",.")
    st.plotly_chart(grafik, width="stretch")


def dagilim_grafigi(degerler, sira, renkler, baslik):
    sayilar = degerler.value_counts().reindex(sira, fill_value=0)
    grafik = px.bar(x=sira, y=sayilar.values, color=sira, color_discrete_map=renkler,
                    text_auto=True, title=baslik, labels={"x": "", "y": "Müşteri"})
    grafik.update_layout(showlegend=False)
    grafik_goster(grafik)


def rozet(metin, renk):
    """Segment veya risk için renkli etiket."""
    return (f"<span style='background:{renk};color:white;padding:2px 10px;border-radius:12px;"
            f"font-size:0.85rem;font-weight:600'>{metin}</span>")


def risk_belirle(olasilik):
    return pd.cut(olasilik, bins=[0, 0.4, 0.7, 1], labels=RISKLER, include_lowest=True).astype(str)


def csv_indir(tablo, dosya_adi, etiket):
    st.download_button(etiket, tablo.to_csv(index=False).encode("utf-8-sig"),
                       file_name=dosya_adi, mime="text/csv")


# ---------------- Veri ve modellerin yüklenmesi ----------------

@st.cache_data
def verileri_yukle():
    # churn_sonuclari.csv; RFM, segment ve risk bilgisini birlikte içerir
    musteriler = pd.read_csv(VERI / "churn_sonuclari.csv")
    gecis = pd.read_csv(VERI / "gecis_matrisi.csv", index_col=0)
    gecisler = pd.read_csv(VERI / "segment_gecisleri.csv")
    return musteriler, gecis, gecisler


@st.cache_resource
def churn_modeli():
    return joblib.load(VERI / "churn_model.pkl")


@st.cache_resource
def kmeans_modeli():
    """K-Means kaydedilmediği için rfm_ozgun.csv üzerinde yeniden eğitilir."""
    rfm = pd.read_csv(VERI / "rfm_ozgun.csv")
    olcekleyici = StandardScaler()
    X = olcekleyici.fit_transform(np.log1p(rfm[RFM]))
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)

    # Kümeleri medyanlara göre adlandır (notebook 07 ile aynı kural)
    rfm["Kume"] = km.labels_
    ozet = rfm.groupby("Kume")[RFM].median()
    vip = ozet["Monetary"].idxmax()
    uykuda = ozet.drop(vip)["Recency"].idxmax()
    kalan = ozet.drop([vip, uykuda])
    isim = {vip: "VIP", uykuda: "Uykuda",
            kalan["Monetary"].idxmax(): "Sadık", kalan["Monetary"].idxmin(): "Yeni/Potansiyel"}
    return olcekleyici, km, isim


# ---------------- Yeni veri için temizleme ve özellik çıkarımı ----------------

def temizle(df):
    """notebook 01-02'deki temizleme kuralları; her adımda kalan satırı raporlar."""
    rapor = [("Başlangıç (yüklenen veri)", len(df))]

    temiz = df.dropna(subset=["CustomerID"])
    rapor.append(("Müşteri numarası boş kayıtlar çıkarıldı", len(temiz)))

    temiz = temiz[~temiz["InvoiceNo"].astype(str).str.startswith("C")]
    temiz = temiz[(temiz["Quantity"] > 0) & (temiz["UnitPrice"] > 0)]
    rapor.append(("İptal/iade faturaları ve sıfır/negatif adet-fiyat çıkarıldı", len(temiz)))

    temiz = temiz[~temiz["StockCode"].isin(URUN_DISI)]
    rapor.append(("Ürün dışı kodlar çıkarıldı", len(temiz)))

    for fatura, urun in HATALI:
        temiz = temiz[~((temiz["InvoiceNo"] == fatura) & (temiz["StockCode"] == urun))]
    rapor.append(("Aynı gün iptal edilen dev siparişler çıkarıldı", len(temiz)))

    temiz = temiz.drop_duplicates()
    rapor.append(("Tekrar eden satırlar çıkarıldı", len(temiz)))

    rapor = pd.DataFrame(rapor, columns=["Adım", "Kalan Satır"])
    rapor["Çıkarılan"] = (-rapor["Kalan Satır"].diff()).fillna(0).astype(int)
    return temiz.copy(), rapor


def ozellik_cikar(temiz):
    """Müşteri bazında RFM ve churn özellikleri (notebook 02 ve 08 ile aynı)."""
    temiz["Tutar"] = temiz["Quantity"] * temiz["UnitPrice"]
    referans = temiz["InvoiceDate"].max() + pd.Timedelta(days=1)

    o = temiz.groupby("CustomerID").agg(
        Son=("InvoiceDate", "max"),
        Ilk=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Tutar", "sum"),
        UrunCesidi=("StockCode", "nunique"),
    ).reset_index()
    o["Recency"] = (referans - o["Son"]).dt.days
    o["IlkAlisveris"] = (referans - o["Ilk"]).dt.days
    o["OrtSepet"] = o["Monetary"] / o["Frequency"]
    return o[["CustomerID"] + OZELLIKLER]


def yeni_veriyi_analiz_et(df):
    temiz, rapor = temizle(df)
    if temiz.empty:
        return None, rapor

    musteri = ozellik_cikar(temiz)

    olcekleyici, km, isim = kmeans_modeli()
    X = olcekleyici.transform(np.log1p(musteri[RFM]))
    musteri["Segment"] = pd.Series(km.predict(X)).map(isim)

    musteri["ChurnOlasiligi"] = churn_modeli().predict_proba(musteri[OZELLIKLER])[:, 1].round(3)
    musteri["Risk"] = risk_belirle(musteri["ChurnOlasiligi"])
    musteri["Oneri"] = musteri["Segment"].map(ONERI)
    return musteri, rapor


def csv_oku(dosya):
    tipler = {"InvoiceNo": str, "StockCode": str}
    try:
        return pd.read_csv(dosya, dtype=tipler)
    except UnicodeDecodeError:
        dosya.seek(0)
        return pd.read_csv(dosya, dtype=tipler, encoding="latin-1")


# ---------------- Sayfalar ----------------

def genel_bakis(musteriler):
    st.title("Müşteri Segmentasyonu Paneli")
    st.markdown(
        "Bu panel, bir çevrimiçi mağazanın bir yıllık satış kayıtlarını inceleyerek müşterileri "
        "**alışveriş davranışlarına göre 4 gruba** ayırır. Her grubun şirkete ne kadar kazandırdığını, "
        "hangi müşterilerin **alışverişi bırakma riski** taşıdığını ve her gruba **hangi pazarlama "
        "aksiyonunun** uygun olduğunu gösterir.")
    with st.expander("Nasıl kullanılır?"):
        st.markdown("""
1. **Genel Bakış:** Tüm müşterilerin özeti ve 4 segmentin kartları.
2. **Segmentler:** Her segmentin tipik (medyan) alışveriş davranışı ve pazarlama önerisi.
3. **Segment Geçişi:** Müşterilerin iki dönem arasında hangi segmente geçtiği.
4. **Churn Riski:** Alışverişi bırakma riskine göre müşteri listesi (filtrelenebilir, indirilebilir).
5. **Öncelikli Müşteriler:** Çok harcayan ama kaybedilme riski olan müşteriler; ilk ulaşılması gerekenler.
6. **Müşteri Sorgula:** Bir müşteri numarası girip o müşterinin profil kartını görün.
7. **Yeni Veri Analizi:** Kendi satış dosyanızı (CSV) yükleyip aynı analizi yaptırın.
8. **Yöntem:** Kullanılan veri, adımlar ve modeller.
""")

    toplam_ciro = musteriler["Monetary"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Müşteri", sayi(len(musteriler)))
    k2.metric("Toplam Ciro", para(toplam_ciro / 1e6, 2) + " Mn")
    k3.metric("Ort. Ciro / Müşteri", para(toplam_ciro / len(musteriler)))
    k4.metric("Yüksek Riskli Müşteri", sayi((musteriler["Risk"] == "Yüksek").sum()))

    ozet = musteriler.groupby("Segment").agg(
        Musteri=("CustomerID", "count"), Ciro=("Monetary", "sum"), Risk=("ChurnOlasiligi", "mean"),
    ).reindex(SEGMENTLER)
    ozet["Oran"] = ozet["Musteri"] / ozet["Musteri"].sum() * 100
    ozet["CiroPayi"] = ozet["Ciro"] / ozet["Ciro"].sum() * 100

    st.subheader("Segmentler (önem sırasına göre)")
    kolonlar = st.columns(2) + st.columns(2)  # 2x2 kartlar
    for kolon, seg in zip(kolonlar, SEGMENTLER):
        bilgi = ozet.loc[seg]
        kolon.markdown(f"""
<div style="border-left:6px solid {SEGMENT_RENK[seg]};border-radius:8px;padding:12px 14px;
            background:rgba(128,128,128,0.08);margin-bottom:14px">
  <div style="font-weight:700;font-size:1.05rem;margin-bottom:6px">{ETIKET[seg]}</div>
  <div style="font-size:1.6rem;font-weight:700">{sayi(bilgi['Musteri'])} <span style="font-size:0.9rem;font-weight:400">müşteri</span></div>
  <div>Müşteri oranı: <b>{yuzde(bilgi['Oran'])}</b></div>
  <div>Ciro payı: <b>{yuzde(bilgi['CiroPayi'])}</b></div>
  <div>Ortalama churn riski: <b>{yuzde(bilgi['Risk'] * 100)}</b></div>
  <div style="margin-top:8px;font-size:0.9rem;opacity:0.85">Önerilen aksiyon: {KISA_AKSIYON[seg]}</div>
</div>""", unsafe_allow_html=True)

    sol, sag = st.columns(2)
    with sol:
        grafik = px.pie(ozet.reset_index(), names="Segment", values="Musteri", color="Segment",
                        color_discrete_map=SEGMENT_RENK, hole=0.45, title="Müşteri Oranı")
        grafik.update_traces(sort=False, textinfo="percent")
        grafik_goster(grafik)
    with sag:
        tablo = ozet.reset_index().melt(id_vars="Segment", value_vars=["Oran", "CiroPayi"],
                                        var_name="Ölçü", value_name="Yüzde")
        tablo["Ölçü"] = tablo["Ölçü"].map({"Oran": "Müşteri Oranı", "CiroPayi": "Ciro Payı"})
        grafik = px.bar(tablo, x="Segment", y="Yüzde", color="Ölçü", barmode="group", text_auto=".1f",
                        title="Müşteri Oranı ve Ciro Payı (%)",
                        color_discrete_sequence=["#9aa0a6", "#1f4fd1"])
        grafik.update_layout(yaxis_title="%", xaxis_title="", legend_title="")
        grafik_goster(grafik)

    vip = ozet.loc["VIP"]
    st.info(f"VIP müşteriler, müşteri tabanının yalnızca {yuzde(vip['Oran'])}'ini oluşturmasına rağmen "
            f"toplam cironun {yuzde(vip['CiroPayi'])}'ini sağlamaktadır.")


def segmentler_sayfasi(musteriler):
    st.title("Segmentler")
    st.write("Değerler her segmentin **medyanıdır** (tipik müşterisi). Recency: son alışverişten bu yana "
             "geçen gün, Frequency: sipariş (fatura) sayısı, Monetary: toplam harcama.")

    profil = musteriler.groupby("Segment").agg(
        Musteri=("CustomerID", "count"),
        Recency=("Recency", "median"),
        Frequency=("Frequency", "median"),
        Monetary=("Monetary", "median"),
    ).reindex(SEGMENTLER)
    profil["Oran"] = (profil["Musteri"] / profil["Musteri"].sum() * 100).round(1)
    profil = profil.rename(columns={"Musteri": "Müşteri Sayısı", "Recency": "Recency (gün)",
                                    "Monetary": "Monetary (£)", "Oran": "Müşteri Oranı (%)"})
    st.dataframe(profil.round(1), width="stretch")

    grafik = px.box(musteriler, x="Segment", y="Monetary", color="Segment", log_y=True,
                    category_orders={"Segment": SEGMENTLER}, color_discrete_map=SEGMENT_RENK,
                    title="Segmentlere göre toplam harcama dağılımı (£, log ölçek)")
    grafik.update_layout(showlegend=False, xaxis_title="", yaxis_title="Monetary (£)")
    grafik_goster(grafik)

    st.subheader("Pazarlama Önerileri")
    for seg in SEGMENTLER:
        with st.container(border=True):
            st.markdown(f"{rozet(ETIKET[seg], SEGMENT_RENK[seg])} &nbsp; "
                        f"{sayi(profil.loc[seg, 'Müşteri Sayısı'])} müşteri", unsafe_allow_html=True)
            st.write(ONERI[seg])


def renk_saydam(hex_renk, saydamlik=0.4):
    r, g, b = (int(hex_renk[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{saydamlik})"


def gecis_sayfasi(gecis, gecisler):
    st.title("Segment Geçişi")
    st.write("Veri iki döneme ayrıldı: **1. dönem** Aralık 2010 – Mayıs 2011, **2. dönem** Haziran – Aralık 2011. "
             f"1. dönemde alışveriş yapan **{sayi(len(gecisler))}** müşterinin 2. dönemde hangi segmentte "
             "olduğu incelendi.")

    # Sankey: sol 1. dönem, sağ 2. dönem
    sag_dugumler = SEGMENTLER + ["Alışveriş yok"]
    akis = gecisler.groupby(["Segment_1", "Segment_2"]).size().reset_index(name="Musteri")
    kaynak = akis["Segment_1"].map(SEGMENTLER.index)
    hedef = akis["Segment_2"].map(sag_dugumler.index) + len(SEGMENTLER)
    dugum_renk = [SEGMENT_RENK[s] for s in SEGMENTLER] + [SEGMENT_RENK.get(s, KAYIP_RENK) for s in sag_dugumler]

    sankey = go.Figure(go.Sankey(
        node=dict(label=[f"{s} (1. dönem)" for s in SEGMENTLER] + [f"{s} (2. dönem)" for s in sag_dugumler],
                  color=dugum_renk, pad=18, thickness=18),
        link=dict(source=kaynak, target=hedef, value=akis["Musteri"],
                  color=[renk_saydam(SEGMENT_RENK[s]) for s in akis["Segment_1"]],
                  hovertemplate="%{source.label} → %{target.label}<br>%{value} müşteri<extra></extra>"),
    ))
    sankey.update_layout(title="Müşteri akışı: 1. dönem → 2. dönem", height=480)
    grafik_goster(sankey)

    st.write("Isı haritası aynı bilgiyi yüzde olarak gösterir: her satır 1. dönemdeki bir segmenttir ve satır toplamı %100'dür.")
    grafik = px.imshow(gecis, text_auto=".1f", color_continuous_scale="Blues", aspect="auto",
                       labels={"x": "2. Dönem", "y": "1. Dönem", "color": "%"})
    grafik.update_layout(height=400)
    grafik_goster(grafik)

    st.subheader("Ne anlama geliyor?")
    st.markdown(f"""
- **VIP müşteriler yerinde duruyor:** 1. dönemdeki VIP'lerin {yuzde(gecis.loc['VIP', 'VIP'])} kadarı 2. dönemde de VIP. Bu grup güvenilir, ama korunmalı.
- **Sadık müşteriler yükselmeye hazır:** Sadıkların {yuzde(gecis.loc['Sadık', 'VIP'])} kadarı VIP'e çıktı. Sadık segmente yapılan kampanyalar en çok getiriyi sağlayabilir.
- **Uykudaki müşterilerin neredeyse yarısı geri gelmiyor:** Uykudakilerin {yuzde(gecis.loc['Uykuda', 'Alışveriş yok'])} kadarı 2. dönemde hiç alışveriş yapmadı. Geri kazanım için erken davranmak gerekir.
- **Not:** 2. dönemde ilk kez gelen 1.618 yeni müşteri 1. dönemde olmadığı için bu grafikte yer almaz.
""")


def churn_sayfasi(musteriler):
    st.title("Churn (Kayıp Müşteri) Riski")
    st.write("Random Forest modeli her müşteriye 0 ile 1 arasında bir **alışverişi bırakma olasılığı** verir. "
             "Risk seviyeleri: **Düşük** (< 0,4), **Orta** (0,4 – 0,7), **Yüksek** (> 0,7).")

    sayilar = musteriler["Risk"].value_counts().reindex(RISKLER, fill_value=0)
    for kutu, risk in zip(st.columns(3), RISKLER):
        kutu.metric(f"{risk} Risk ({yuzde(sayilar[risk] / len(musteriler) * 100)})", sayi(sayilar[risk]))

    sol, sag = st.columns(2)
    with sol:
        dagilim_grafigi(musteriler["Risk"], RISKLER, RISK_RENK, "Risk Dağılımı")
    with sag:
        st.markdown("**Segment × Risk (satır yüzdesi)**")
        tablo = pd.crosstab(musteriler["Segment"], musteriler["Risk"], normalize="index") * 100
        st.dataframe(tablo.reindex(index=SEGMENTLER, columns=RISKLER, fill_value=0).round(1), width="stretch")
        st.markdown("**Segment × Risk (müşteri sayısı)**")
        tablo = pd.crosstab(musteriler["Segment"], musteriler["Risk"], margins=True, margins_name="Toplam")
        st.dataframe(tablo.reindex(index=SEGMENTLER + ["Toplam"], columns=RISKLER + ["Toplam"], fill_value=0),
                     width="stretch")

    st.subheader("Müşteri Listesi")
    f1, f2 = st.columns(2)
    secili_segment = f1.multiselect("Segment", SEGMENTLER, default=SEGMENTLER)
    secili_risk = f2.multiselect("Risk seviyesi", RISKLER, default=["Yüksek"])

    liste = musteriler[musteriler["Segment"].isin(secili_segment) & musteriler["Risk"].isin(secili_risk)]
    liste = liste.sort_values("ChurnOlasiligi", ascending=False)
    liste = liste.assign(CustomerID=liste["CustomerID"].astype(int), OrtSepet=liste["OrtSepet"].round(2))
    st.write(f"{sayi(len(liste))} müşteri listeleniyor.")
    st.dataframe(liste, hide_index=True, width="stretch")
    csv_indir(liste, "churn_listesi.csv", "Listeyi CSV olarak indir")


def oncelikli_sayfasi(musteriler):
    st.title("Öncelikli Müşteriler")
    esik = musteriler["Monetary"].quantile(0.75)
    st.markdown(f"""
**Ölçüt:** Toplam harcaması müşterilerin **en yüksek %25'inde** olan (Monetary ≥ **{para(esik, 2)}**)
**ve** churn riski **Orta** veya **Yüksek** olan müşteriler.

Bu müşteriler şirkete çok kazandırıyor ama kaybedilme ihtimalleri de var; bu yüzden ilk ulaşılması gereken kişilerdir.
**Beklenen kayıp** = Monetary × churn olasılığı. Geçmiş harcamaya dayanan kaba bir önem ölçüsüdür; listeyi sıralamak için kullanılır.
""")

    secim = musteriler[(musteriler["Monetary"] >= esik) & musteriler["Risk"].isin(["Orta", "Yüksek"])].copy()
    secim["BeklenenKayip"] = (secim["Monetary"] * secim["ChurnOlasiligi"]).round(2)
    secim["Oneri"] = ONCELIKLI_AKSIYON
    secim = secim.sort_values("BeklenenKayip", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Öncelikli Müşteri", sayi(len(secim)))
    k2.metric("Toplam Harcamaları", para(secim["Monetary"].sum()))
    k3.metric("Toplam Ciro İçindeki Payı", yuzde(secim["Monetary"].sum() / musteriler["Monetary"].sum() * 100))
    k4.metric("Toplam Beklenen Kayıp", para(secim["BeklenenKayip"].sum()))

    st.success(f"**Önerilen aksiyon:** {ONCELIKLI_AKSIYON}")

    sol, sag = st.columns([1, 2])
    with sol:
        dagilim_grafigi(secim["Segment"], SEGMENTLER, SEGMENT_RENK, "Segmentlere göre dağılım")
    with sag:
        liste = secim[["CustomerID", "Segment", "Risk", "ChurnOlasiligi", "Recency", "Frequency",
                       "Monetary", "BeklenenKayip", "Oneri"]]
        liste = liste.assign(CustomerID=liste["CustomerID"].astype(int))
        st.dataframe(liste, hide_index=True, width="stretch", height=420)
    csv_indir(liste, "oncelikli_musteriler.csv", "Öncelikli müşterileri CSV olarak indir")


def risk_gostergesi(olasilik):
    grafik = go.Figure(go.Indicator(
        mode="gauge+number", value=olasilik * 100,
        number={"prefix": "%", "valueformat": ".1f"},
        title={"text": "Churn olasılığı"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333333", "thickness": 0.25},
               "steps": [{"range": [0, 40], "color": RISK_RENK["Düşük"]},
                         {"range": [40, 70], "color": RISK_RENK["Orta"]},
                         {"range": [70, 100], "color": RISK_RENK["Yüksek"]}]},
    ))
    grafik.update_layout(height=260, margin=dict(t=60, b=10, l=30, r=30))
    grafik_goster(grafik)


def sorgu_sayfasi(musteriler):
    st.title("Müşteri Sorgula")
    girilen = st.text_input("Müşteri numarası (CustomerID) girin", placeholder="ör. 12347")
    if not girilen:
        st.caption(f"Örnek numaralar: {', '.join(str(int(i)) for i in musteriler['CustomerID'].head(3))}")
        return

    try:
        musteri_id = float(girilen.strip())
    except ValueError:
        st.warning("Lütfen yalnızca sayı girin (ör. 12347).")
        return

    satir = musteriler[musteriler["CustomerID"] == musteri_id]
    if satir.empty:
        st.warning(f"{girilen} numaralı müşteri bulunamadı. Lütfen numarayı kontrol edin.")
        return

    secilen = satir.iloc[0]
    seg = secilen["Segment"]
    with st.container(border=True):
        st.markdown(f"### Müşteri {int(secilen['CustomerID'])}")
        st.markdown(f"{rozet(ETIKET[seg], SEGMENT_RENK[seg])} &nbsp; {rozet(secilen['Risk'] + ' risk', RISK_RENK[secilen['Risk']])}",
                    unsafe_allow_html=True)

        sol, sag = st.columns([1, 2])
        with sol:
            risk_gostergesi(secilen["ChurnOlasiligi"])
        with sag:
            # Değerin tüm müşteriler içindeki konumu
            konumlar = [
                ("Son alışveriş", f"{sayi(secilen['Recency'])} gün önce",
                 (musteriler["Recency"] > secilen["Recency"]).mean(), "Daha uzun süredir gelmeyen müşteri oranı"),
                ("Sipariş sayısı", f"{sayi(secilen['Frequency'])} sipariş",
                 (musteriler["Frequency"] < secilen["Frequency"]).mean(), "Daha az sipariş veren müşteri oranı"),
                ("Toplam harcama", para(secilen["Monetary"], 2),
                 (musteriler["Monetary"] < secilen["Monetary"]).mean(), "Daha az harcayan müşteri oranı"),
            ]
            for baslik, deger, oran, aciklama in konumlar:
                st.markdown(f"**{baslik}:** {deger}")
                st.progress(float(oran), text=f"{aciklama}: {yuzde(oran * 100)}")

        k1, k2, k3 = st.columns(3)
        k1.metric("Ortalama sepet", para(secilen["OrtSepet"], 2))
        k2.metric("Farklı ürün sayısı", sayi(secilen["UrunCesidi"]))
        k3.metric("İlk alışverişten bu yana", f"{sayi(secilen['IlkAlisveris'])} gün")

        st.success(f"**Önerilen aksiyon:** {ONERI[seg]}")


def yeni_veri_sayfasi():
    st.title("Yeni Veri Analizi")
    st.write("Online Retail formatında bir CSV yükleyin. Gerekli sütunlar: " + ", ".join(f"`{s}`" for s in GEREKLI_SUTUNLAR))

    ornek = VERI / "ornek_islemler.csv"
    if ornek.exists():
        st.download_button("Örnek dosyayı indir", ornek.read_bytes(),
                           file_name="ornek_islemler.csv", mime="text/csv")

    dosya = st.file_uploader("CSV dosyası seçin", type="csv")
    if dosya is None:
        return

    df = csv_oku(dosya)
    eksik = [s for s in GEREKLI_SUTUNLAR if s not in df.columns]
    if eksik:
        st.error("Dosyada şu sütunlar eksik: " + ", ".join(eksik))
        return

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    with st.spinner("Veri temizleniyor ve müşteriler analiz ediliyor..."):
        sonuc, rapor = yeni_veriyi_analiz_et(df)

    st.subheader("1. Veri Temizleme")
    st.dataframe(rapor, hide_index=True, width="stretch")
    if sonuc is None:
        st.error("Temizleme sonrası analiz edilecek kayıt kalmadı.")
        return

    st.subheader("2. Özet")
    k1, k2, k3 = st.columns(3)
    k1.metric("Müşteri Sayısı", sayi(len(sonuc)))
    k2.metric("Toplam Ciro", para(sonuc["Monetary"].sum()))
    k3.metric("Yüksek Riskli Müşteri", sayi((sonuc["Risk"] == "Yüksek").sum()))

    sol, sag = st.columns(2)
    with sol:
        dagilim_grafigi(sonuc["Segment"], SEGMENTLER, SEGMENT_RENK, "Segment Dağılımı")
    with sag:
        dagilim_grafigi(sonuc["Risk"], RISKLER, RISK_RENK, "Risk Dağılımı")

    st.subheader("3. Müşteri Sonuçları")
    sonuc = sonuc.assign(CustomerID=sonuc["CustomerID"].astype(int), OrtSepet=sonuc["OrtSepet"].round(2),
                         Monetary=sonuc["Monetary"].round(2))
    st.dataframe(sonuc, hide_index=True, width="stretch")
    csv_indir(sonuc, "yeni_veri_sonuclari.csv", "Sonuçları CSV olarak indir")
    st.caption("Not: Recency ve diğer gün hesapları, yüklenen dosyadaki en son tarihin bir gün sonrasına göre yapılır.")


def yontem_sayfasi():
    st.title("Yöntem")
    st.markdown("""
#### 1. Veri seti
UCI Machine Learning Repository'deki **Online Retail** veri seti: İngiltere merkezli bir çevrimiçi perakende
firmasının Aralık 2010 – Aralık 2011 arasındaki işlem kayıtları. Tutarlar **sterlin (£)** cinsindendir.

#### 2. Veri temizleme
| Adım | Çıkarılan | Kalan Satır |
|---|:---:|:---:|
| Başlangıç (ham veri) | — | 541.909 |
| Müşteri numarası boş kayıtlar | 135.080 | 406.829 |
| İptal/iade faturaları + sıfır/negatif adet–fiyat | 8.945 | 397.884 |
| Ürün dışı kodlar (POST, M, C2, DOT, BANK CHARGES, PADS) | 1.547 | 396.337 |
| Aynı gün tamamı iptal edilen çok büyük siparişlerin asıl satırları | 4 | 396.333 |
| Birebir yinelenen (kopya) satırlar | 5.187 | 391.146 |
| **Analize hazır veri** | — | **391.146 satır / 4.333 müşteri** |

#### 3. RFM
- **Recency:** Son alışverişten bu yana geçen gün
- **Frequency:** Benzersiz sipariş (fatura) sayısı
- **Monetary:** Toplam harcama (adet × birim fiyat)

Çarpık dağılımlar `log1p` ile düzeltilmiş, ardından `StandardScaler` ile ölçeklenmiştir.

#### 4. Kümeleme (K-Means, k=4)
Küme sayısı k=2–10 arasında Elbow yöntemi ve Silhouette skoru ile incelendi. En yüksek Silhouette k=2'de
(0,433) çıksa da iki küme anlamlı pazarlama segmenti üretmediği için **k=4** seçildi.
Silhouette **0,337**, Davies-Bouldin **1,012**.

| Algoritma | Küme | Silhouette | Davies-Bouldin |
|---|:---:|:---:|:---:|
| **K-Means** | 4 | **0,337** | **1,012** |
| Hiyerarşik Kümeleme | 4 | 0,291 | 1,036 |
| DBSCAN | 2 | 0,296 | 1,062 |

Kümeler medyan RFM değerlerine göre adlandırıldı: en çok harcayan **VIP**; kalanlardan en uzun süredir
gelmeyen **Uykuda**; kalan ikisinden çok harcayan **Sadık**, az harcayan **Yeni/Potansiyel**.

#### 5. Segment geçişi
Veri iki döneme bölündü: 1. dönem Aralık 2010 – Mayıs 2011 (2.715 müşteri), 2. dönem Haziran – Aralık 2011
(3.540 müşteri). Ölçekleyici ve K-Means 1. dönemde eğitilip aynı model 2. döneme uygulandı.

#### 6. Churn (kayıp müşteri) tahmini
Veri 9 Eylül 2011'den ikiye ayrıldı: özellikler ilk 9 aydan çıkarıldı, son 3 ayda (90 gün) hiç alışveriş yapmayanlar
**churn** sayıldı (3.359 müşterinin %43'ü). Özellikler: Recency, Frequency, Monetary, ilk alışverişten bu yana
geçen gün, ürün çeşidi ve ortalama sepet. Veri %75 eğitim / %25 test olarak ayrıldı.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Lojistik Regresyon | 0,670 | 0,587 | 0,778 | 0,669 | 0,735 |
| **Random Forest** (max_depth=5) | **0,676** | 0,596 | 0,758 | 0,667 | **0,744** |

Model tüm müşterilere uygulanarak risk seviyesi atandı: Düşük (< 0,4), Orta (0,4–0,7), Yüksek (> 0,7).
En etkili özellikler sırasıyla Frequency, Monetary ve Recency'dir.
""")


# ---------------- Ana akış ----------------

musteriler, gecis, gecisler = verileri_yukle()

SAYFALAR = {
    "🏠 Genel Bakış": lambda: genel_bakis(musteriler),
    "🏷️ Segmentler": lambda: segmentler_sayfasi(musteriler),
    "🔄 Segment Geçişi": lambda: gecis_sayfasi(gecis, gecisler),
    "⚠️ Churn Riski": lambda: churn_sayfasi(musteriler),
    "🎯 Öncelikli Müşteriler": lambda: oncelikli_sayfasi(musteriler),
    "🔍 Müşteri Sorgula": lambda: sorgu_sayfasi(musteriler),
    "📤 Yeni Veri Analizi": yeni_veri_sayfasi,
    "📚 Yöntem": yontem_sayfasi,
}

st.sidebar.title("RFM Paneli")
sayfa = st.sidebar.radio("Sayfa seçin", list(SAYFALAR))
st.sidebar.caption("Müşteri Segmentasyonu — RFM & Kümeleme Motoru")
SAYFALAR[sayfa]()
