from itertools import count
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Müşteri Segmentasyonu", page_icon=":material/insights:", layout="wide")

# Dosya yolları app.py'nin konumuna göre bulunur (Streamlit Cloud'da da çalışır)
VERI = Path(__file__).resolve().parent.parent / "data"

SEGMENTLER = ["VIP", "Sadık", "Yeni/Potansiyel", "Uykuda"]  # önem sırası
RISKLER = ["Düşük", "Orta", "Yüksek"]
RFM = ["Recency", "Frequency", "Monetary"]
OZELLIKLER = ["Recency", "Frequency", "Monetary", "IlkAlisveris", "UrunCesidi", "OrtSepet"]

# Renkler tüm sayfalarda aynı
ANA_RENK = "#4f46e5"
SEGMENT_RENK = {"VIP": "#1f4fd1", "Sadık": "#4fa3e0", "Yeni/Potansiyel": "#8e6bd6", "Uykuda": "#9aa0a6"}
RISK_RENK = {"Düşük": "#2e9e6b", "Orta": "#e0a526", "Yüksek": "#d64545"}
KAYIP_RENK = "#6b7280"

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

# Görünüm: tüm özel CSS burada. Renkler yarı saydam seçildi; böylece açık ve koyu temada aynı CSS çalışır.
# Kartların zemini "backdrop-filter: brightness" ile arka planı biraz aydınlatarak oluşur.
CSS = """
<style>
.block-container {padding-top: 4.5rem; padding-bottom: 4rem; max-width: 1380px;}
h1, h2, h3 {letter-spacing: -0.01em;}

.hero {background: linear-gradient(120deg, #4338ca 0%, #4f46e5 45%, #0891b2 100%); color: #fff;
       border-radius: 18px; padding: 26px 32px; margin-bottom: 1.6rem;
       box-shadow: 0 10px 30px rgba(79,70,229,0.25);}
.hero .ust {font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; opacity: 0.8;}
.hero .baslik {font-size: 1.85rem; font-weight: 700; margin: 4px 0 4px; line-height: 1.2;}
.hero .aciklama {font-size: 1rem; opacity: 0.92; max-width: 880px; line-height: 1.5;}

.kart, div[class*="st-key-kutu"] {
    backdrop-filter: brightness(1.45); -webkit-backdrop-filter: brightness(1.45);
    border: 1px solid rgba(128,128,128,0.18); border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.06), 0 4px 14px rgba(16,24,40,0.07);}
div[class*="st-key-kutu"] {border: 1px solid rgba(128,128,128,0.18) !important;}
.kart {padding: 16px 18px; height: 100%; margin-bottom: 1rem;}
.kart .etiket {font-size: 0.85rem; font-weight: 600; opacity: 0.7;}
.kart .deger {font-size: 1.8rem; font-weight: 700; line-height: 1.15; margin: 6px 0 2px;}
.kart .alt {font-size: 0.84rem; opacity: 0.7;}
.kart .satir {display: flex; justify-content: space-between; font-size: 0.9rem; padding: 5px 0;
              border-bottom: 1px solid rgba(128,128,128,0.15);}
.kart .satir b {font-weight: 600;}
.kart .aksiyon {margin-top: 10px; font-size: 0.86rem; line-height: 1.45; opacity: 0.85;}
div[class*="st-key-kutu"] {padding: 8px 10px 6px;}
.kutu-baslik {font-weight: 600; font-size: 1rem;}
.kutu-aciklama {font-size: 0.83rem; opacity: 0.65; margin-top: 2px;}

.not-kutusu {border-radius: 12px; padding: 14px 18px; margin: 0.4rem 0 1.4rem; border: 1px solid;}
.not-kutusu .not-baslik {font-weight: 700; margin-bottom: 4px;}
.not-kutusu ul {margin: 4px 0 0 0; padding-left: 1.1rem;}
.not-kutusu li {margin: 3px 0;}

.rozet {color: #fff; padding: 3px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin-right: 6px;}
.bosluk {height: 0.6rem;}

.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid rgba(128,128,128,0.2);}
.stTabs [data-baseweb="tab"] {padding: 8px 14px; font-weight: 500;}

section[data-testid="stSidebar"] .logo {display: flex; align-items: center; gap: 12px; padding: 4px 4px 18px;}
section[data-testid="stSidebar"] .logo-ikon {width: 40px; height: 40px; border-radius: 11px; display: flex;
    align-items: center; justify-content: center; background: linear-gradient(135deg, #4f46e5, #0891b2);}
section[data-testid="stSidebar"] .logo-ad {font-weight: 700; font-size: 1.05rem; line-height: 1.2;}
section[data-testid="stSidebar"] .logo-alt {font-size: 0.76rem; opacity: 0.6;}
section[data-testid="stSidebar"] .menu-baslik {font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    opacity: 0.55; padding: 0 4px 4px;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {width: 100%; padding: 8px 12px; border-radius: 9px; margin: 1px 0;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {background: rgba(79,70,229,0.08);}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {background: rgba(79,70,229,0.16); font-weight: 600;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {display: none;}
section[data-testid="stSidebar"] .veri-bilgi {font-size: 0.76rem; opacity: 0.6; border-top: 1px solid rgba(128,128,128,0.2);
    margin-top: 2rem; padding-top: 0.9rem; line-height: 1.5;}
</style>
"""

LOGO_SVG = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" '
            'stroke-linecap="round"><path d="M5 20V11M12 20V5M19 20v-6"/></svg>')

# Tüm grafiklerde ortak düzen; renkler ve yazı rengi Streamlit temasından gelir
GRAFIK_DUZENI = dict(
    font=dict(family="Inter, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    separators=",.", title=None,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
)

KUTU_SAYAC = count()  # her kutuya benzersiz anahtar verir (betik her çalıştığında sıfırlanır)


# ---------------- Görünüm yardımcıları ----------------

def baslik_bandi(baslik, aciklama, ust="MÜŞTERİ SEGMENTASYONU"):
    st.markdown(f"<div class='hero'><div class='ust'>{ust}</div><div class='baslik'>{baslik}</div>"
                f"<div class='aciklama'>{aciklama}</div></div>", unsafe_allow_html=True)


def kart_html(etiket, deger, alt="", renk=ANA_RENK, serit="sol", satirlar=None, aksiyon="", yukseklik=0):
    """Tüm kartların (KPI ve segment) HTML'i. serit: renkli kenarın yeri ('sol' veya 'ust')."""
    kenar = f"border-{'left' if serit == 'sol' else 'top'}: 4px solid {renk}; min-height: {yukseklik}px;"
    icerik = "".join(f"<div class='satir'><span>{a}</span><b>{b}</b></div>" for a, b in (satirlar or []))
    alt = f"<div class='alt'>{alt}</div>" if alt else ""
    aksiyon = f"<div class='aksiyon'>{aksiyon}</div>" if aksiyon else ""
    return (f"<div class='kart' style='{kenar}'><div class='etiket'>{etiket}</div>"
            f"<div class='deger'>{deger}</div>{alt}{icerik}{aksiyon}</div>")


def kart_satiri(kartlar):
    """Kartları yan yana dizer. kartlar: kart_html'e gidecek sözlüklerin listesi."""
    for kolon, kart in zip(st.columns(len(kartlar)), kartlar):
        kolon.markdown(kart_html(yukseklik=125, **kart), unsafe_allow_html=True)


def kutu(baslik=None, aciklama=None):
    """Tablo ve grafikleri içine alan kart. 'with kutu(...):' şeklinde kullanılır."""
    kap = st.container(border=True, key=f"kutu_{next(KUTU_SAYAC)}")
    if baslik:
        alt = f"<div class='kutu-aciklama'>{aciklama}</div>" if aciklama else ""
        kap.markdown(f"<div class='kutu-baslik'>{baslik}</div>{alt}", unsafe_allow_html=True)
    return kap


def not_kutusu(baslik, maddeler, renk=ANA_RENK):
    """Öne çıkan bulgular ve önerilen aksiyonlar için renkli not kutusu."""
    icerik = "<ul>" + "".join(f"<li>{m}</li>" for m in maddeler) + "</ul>" if len(maddeler) > 1 else maddeler[0]
    st.markdown(f"<div class='not-kutusu' style='border-color:{renk}55;background:{renk}14;'>"
                f"<div class='not-baslik' style='color:{renk}'>{baslik}</div>{icerik}</div>",
                unsafe_allow_html=True)


def rozet(metin, renk):
    return f"<span class='rozet' style='background:{renk}'>{metin}</span>"


def grafik_goster(grafik, yukseklik=360, yan_bosluk=10):
    """Grafiği ortak düzenle gösterir: şeffaf zemin, az ızgara, Türkçe sayı biçimi."""
    grafik.update_layout(**GRAFIK_DUZENI, height=yukseklik, margin=dict(l=yan_bosluk, r=yan_bosluk, t=30, b=10))
    grafik.update_xaxes(showgrid=False)
    grafik.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    st.plotly_chart(grafik, width="stretch")


def dagilim_grafigi(degerler, sira, renkler, yukseklik=320):
    sayilar = degerler.value_counts().reindex(sira, fill_value=0)
    grafik = px.bar(x=sira, y=sayilar.values, color=sira, color_discrete_map=renkler,
                    text_auto=",d", labels={"x": "", "y": "Müşteri"})
    grafik.update_layout(showlegend=False)
    grafik.update_traces(textposition="outside", cliponaxis=False)
    grafik.update_xaxes(tickangle=0, tickvals=sira, ticktext=[s.replace("/", "/<br>") for s in sira])
    grafik_goster(grafik, yukseklik)


def bosluk():
    st.markdown("<div class='bosluk'></div>", unsafe_allow_html=True)


# ---------------- Biçim ve hesap yardımcıları ----------------

def sayi(x, ondalik=0):
    """Sayıyı Türkçe biçimde yazar: 1234.5 -> 1.234,5"""
    metin = f"{x:,.{ondalik}f}"
    return metin.replace(",", "X").replace(".", ",").replace("X", ".")


def para(x, ondalik=0):
    return "£" + sayi(x, ondalik)


def yuzde(x, ondalik=1):
    return "%" + sayi(x, ondalik)


def risk_belirle(olasilik):
    return pd.cut(olasilik, bins=[0, 0.4, 0.7, 1], labels=RISKLER, include_lowest=True).astype(str)


def csv_indir(tablo, dosya_adi, etiket):
    st.download_button(etiket, tablo.to_csv(index=False).encode("utf-8-sig"),
                       file_name=dosya_adi, mime="text/csv", icon=":material/download:")


def renk_saydam(hex_renk, saydamlik=0.4):
    r, g, b = (int(hex_renk[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{saydamlik})"


def segment_ozeti(musteriler):
    """Segment başına müşteri sayısı, ciro, ortalama churn olasılığı, oran ve ciro payı."""
    ozet = musteriler.groupby("Segment").agg(
        Musteri=("CustomerID", "count"), Ciro=("Monetary", "sum"), Risk=("ChurnOlasiligi", "mean"),
    ).reindex(SEGMENTLER)
    ozet["Oran"] = ozet["Musteri"] / ozet["Musteri"].sum() * 100
    ozet["CiroPayi"] = ozet["Ciro"] / ozet["Ciro"].sum() * 100
    return ozet


def oncelikli_musteriler(musteriler):
    """Harcaması üst %25'te olup riski Orta veya Yüksek olan müşteriler."""
    esik = musteriler["Monetary"].quantile(0.75)
    secim = musteriler[(musteriler["Monetary"] >= esik) & musteriler["Risk"].isin(["Orta", "Yüksek"])].copy()
    secim["BeklenenKayip"] = (secim["Monetary"] * secim["ChurnOlasiligi"]).round(2)
    secim["Oneri"] = ONCELIKLI_AKSIYON
    return secim.sort_values("BeklenenKayip", ascending=False), esik


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

def genel_bakis(musteriler, gecis):
    baslik_bandi("Genel Bakış",
                 "Bu panel, bir çevrimiçi mağazanın bir yıllık satış kayıtlarını inceleyerek müşterileri alışveriş "
                 "davranışlarına göre 4 segmente ayırır. Her segmentin ciroya katkısını, alışverişi bırakma riskini "
                 "ve önerilen pazarlama aksiyonunu tek ekranda gösterir.")
    with st.expander("Nasıl kullanılır?", icon=":material/help:"):
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
    bosluk()

    toplam_ciro = musteriler["Monetary"].sum()
    yuksek = musteriler[musteriler["Risk"] == "Yüksek"]
    kart_satiri([
        dict(etiket="Toplam müşteri", deger=sayi(len(musteriler)), alt="Aralık 2010 – Aralık 2011"),
        dict(etiket="Toplam ciro", deger=para(toplam_ciro / 1e6, 2) + " Mn", alt="Temizlenmiş satışlar"),
        dict(etiket="Müşteri başına ciro", deger=para(toplam_ciro / len(musteriler)), alt="Ortalama"),
        dict(etiket="Yüksek riskli müşteri", deger=sayi(len(yuksek)), renk=RISK_RENK["Yüksek"],
             alt=f"Müşterilerin {yuzde(len(yuksek) / len(musteriler) * 100)} kadarı"),
    ])

    ozet = segment_ozeti(musteriler)
    vip = ozet.loc["VIP"]
    oncelikli, _ = oncelikli_musteriler(musteriler)
    yuksek_uykuda = (yuksek["Segment"] == "Uykuda").sum()
    not_kutusu("Öne çıkan bulgular", [
        f"<b>VIP müşteriler</b> müşterilerin yalnızca {yuzde(vip['Oran'])} kadarı, ama cironun "
        f"<b>{yuzde(vip['CiroPayi'])}</b> kadarını sağlıyor.",
        f"<b>{sayi(len(oncelikli))} değerli müşteri</b> (harcaması üst %25'te) orta veya yüksek risk altında; "
        f"toplam harcamaları {para(oncelikli['Monetary'].sum())}.",
        f"Yüksek riskli {sayi(len(yuksek))} müşterinin <b>{sayi(yuksek_uykuda)}</b> tanesi Uykuda segmentinde.",
        f"Uykudaki müşterilerin <b>{yuzde(gecis.loc['Uykuda', 'Alışveriş yok'])}</b> kadarı bir sonraki dönemde "
        f"hiç alışveriş yapmadı; geri kazanım için erken davranmak gerekir.",
    ])

    st.subheader("Segmentler")
    st.caption("Önem sırasına göre. Ortalama churn riski, segmentteki müşterilerin churn olasılıklarının ortalamasıdır.")
    kolonlar = st.columns(4)
    for kolon, seg in zip(kolonlar, SEGMENTLER):
        bilgi = ozet.loc[seg]
        kolon.markdown(kart_html(
            ETIKET[seg], sayi(bilgi["Musteri"]), alt="müşteri", renk=SEGMENT_RENK[seg], serit="ust",
            satirlar=[("Müşteri oranı", yuzde(bilgi["Oran"])), ("Ciro payı", yuzde(bilgi["CiroPayi"])),
                      ("Ort. churn riski", yuzde(bilgi["Risk"] * 100))],
            aksiyon=KISA_AKSIYON[seg], yukseklik=320), unsafe_allow_html=True)

    sol, sag = st.columns(2)
    with sol, kutu("Müşteri oranı", "Segmentlerin müşteri sayısı içindeki payı"):
        grafik = px.pie(ozet.reset_index(), names="Segment", values="Musteri", color="Segment",
                        color_discrete_map=SEGMENT_RENK, hole=0.55)
        grafik.update_traces(sort=False, textinfo="percent")
        grafik_goster(grafik)
    with sag, kutu("Müşteri oranı ve ciro payı (%)", "Değer birkaç segmentte yoğunlaşıyor"):
        tablo = ozet.reset_index().melt(id_vars="Segment", value_vars=["Oran", "CiroPayi"],
                                        var_name="Ölçü", value_name="Yüzde")
        tablo["Ölçü"] = tablo["Ölçü"].map({"Oran": "Müşteri oranı", "CiroPayi": "Ciro payı"})
        grafik = px.bar(tablo, x="Segment", y="Yüzde", color="Ölçü", barmode="group", text_auto=".1f",
                        color_discrete_sequence=["#a5b4fc", ANA_RENK])
        grafik.update_layout(yaxis_title="%", xaxis_title="")
        grafik_goster(grafik)


def segmentler_sayfasi(musteriler):
    baslik_bandi("Segmentler", "Her segmentin tipik müşterisi nasıl alışveriş yapıyor ve ona hangi aksiyon uygun?")

    profil = musteriler.groupby("Segment").agg(
        Musteri=("CustomerID", "count"),
        Recency=("Recency", "median"),
        Frequency=("Frequency", "median"),
        Monetary=("Monetary", "median"),
    ).reindex(SEGMENTLER)
    profil["Oran"] = (profil["Musteri"] / profil["Musteri"].sum() * 100).round(1)

    sekme1, sekme2 = st.tabs(["Segment profilleri", "Pazarlama önerileri"])
    with sekme1:
        with kutu("Medyan RFM değerleri", "Recency: son alışverişten bu yana geçen gün · Frequency: sipariş "
                                         "sayısı · Monetary: toplam harcama"):
            tablo = profil.rename(columns={"Musteri": "Müşteri Sayısı", "Recency": "Recency (gün)",
                                           "Monetary": "Monetary (£)", "Oran": "Müşteri Oranı (%)"})
            st.dataframe(tablo.round(1), width="stretch")
        bosluk()
        with kutu("Toplam harcama dağılımı", "£, logaritmik ölçek · kutu: müşterilerin ortadaki yarısı"):
            grafik = px.box(musteriler, x="Segment", y="Monetary", color="Segment", log_y=True,
                            category_orders={"Segment": SEGMENTLER}, color_discrete_map=SEGMENT_RENK)
            grafik.update_layout(showlegend=False, xaxis_title="", yaxis_title="Monetary (£)")
            grafik.update_yaxes(dtick=1)  # log eksende yalnızca 10'un katları
            grafik_goster(grafik, 400)
    with sekme2:
        kolonlar = st.columns(2) + st.columns(2)
        for kolon, seg in zip(kolonlar, SEGMENTLER):
            kolon.markdown(kart_html(ETIKET[seg], f"{sayi(profil.loc[seg, 'Musteri'])} müşteri",
                                     renk=SEGMENT_RENK[seg], serit="ust", aksiyon=ONERI[seg], yukseklik=150),
                           unsafe_allow_html=True)


def gecis_sayfasi(gecis, gecisler):
    baslik_bandi("Segment Geçişi", "1. dönemde (Aralık 2010 – Mayıs 2011) alışveriş yapan müşteriler, "
                                   "2. dönemde (Haziran – Aralık 2011) hangi segmente geçti?")

    kart_satiri([
        dict(etiket="İncelenen müşteri", deger=sayi(len(gecisler)), alt="1. dönemde alışveriş yapanlar"),
        dict(etiket="VIP'te kalan", deger=yuzde(gecis.loc["VIP", "VIP"]), renk=SEGMENT_RENK["VIP"],
             alt="1. dönemdeki VIP'ler içinde"),
        dict(etiket="Sadık → VIP", deger=yuzde(gecis.loc["Sadık", "VIP"]), renk=SEGMENT_RENK["Sadık"],
             alt="Üst segmente yükselen sadıklar"),
        dict(etiket="Uykuda → alışveriş yok", deger=yuzde(gecis.loc["Uykuda", "Alışveriş yok"]),
             renk=RISK_RENK["Yüksek"], alt="Geri dönmeyen uykudakiler"),
    ])

    sekme1, sekme2 = st.tabs(["Müşteri akışı", "Geçiş oranları"])
    with sekme1, kutu("Müşteri akışı: 1. dönem → 2. dönem", "Çizgi kalınlığı, o yoldan geçen müşteri sayısıdır"):
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
            textfont=dict(size=13, weight=600, shadow="none"),
        ))
        grafik_goster(sankey, 480)
    with sekme2, kutu("Geçiş oranları (%)", "Her satır 1. dönemdeki bir segmenttir; satır toplamı %100"):
        grafik = px.imshow(gecis, text_auto=".1f", aspect="auto",
                           color_continuous_scale=[[0, "#eef2ff"], [1, "#3730a3"]],
                           labels={"x": "2. Dönem", "y": "1. Dönem", "color": "%"})
        grafik.update_yaxes(showgrid=False)
        grafik_goster(grafik, 400)

    bosluk()
    not_kutusu("Ne anlama geliyor?", [
        f"<b>VIP müşteriler yerinde duruyor:</b> 1. dönemdeki VIP'lerin {yuzde(gecis.loc['VIP', 'VIP'])} kadarı "
        "2. dönemde de VIP. Bu grup güvenilir, ama korunmalı.",
        f"<b>Sadık müşteriler yükselmeye hazır:</b> Sadıkların {yuzde(gecis.loc['Sadık', 'VIP'])} kadarı VIP'e çıktı. "
        "Sadık segmente yapılan kampanyalar en çok getiriyi sağlayabilir.",
        f"<b>Uykudaki müşterilerin neredeyse yarısı geri gelmiyor:</b> Uykudakilerin "
        f"{yuzde(gecis.loc['Uykuda', 'Alışveriş yok'])} kadarı 2. dönemde hiç alışveriş yapmadı.",
        "2. dönemde ilk kez gelen 1.618 yeni müşteri 1. dönemde olmadığı için bu grafiklerde yer almaz.",
    ])


def churn_sayfasi(musteriler):
    baslik_bandi("Churn Riski", "Random Forest modeli her müşteriye bir alışverişi bırakma olasılığı verir. "
                                "Düşük: 0,4'ün altı · Orta: 0,4 – 0,7 · Yüksek: 0,7'nin üstü.")

    sayilar = musteriler["Risk"].value_counts().reindex(RISKLER, fill_value=0)
    kart_satiri([dict(etiket=f"{risk} risk", deger=sayi(sayilar[risk]), renk=RISK_RENK[risk],
                      alt=f"Müşterilerin {yuzde(sayilar[risk] / len(musteriler) * 100)} kadarı")
                 for risk in RISKLER])

    sekme1, sekme2 = st.tabs(["Risk dağılımı", "Müşteri listesi"])
    with sekme1:
        sol, sag = st.columns([2, 3])
        with sol, kutu("Risk seviyelerine göre müşteri sayısı"):
            dagilim_grafigi(musteriler["Risk"], RISKLER, RISK_RENK)
        with sag, kutu("Segment × risk", "Üstte satır yüzdesi, altta müşteri sayısı"):
            tablo = pd.crosstab(musteriler["Segment"], musteriler["Risk"], normalize="index") * 100
            st.dataframe(tablo.reindex(index=SEGMENTLER, columns=RISKLER, fill_value=0).round(1), width="stretch")
            tablo = pd.crosstab(musteriler["Segment"], musteriler["Risk"], margins=True, margins_name="Toplam")
            st.dataframe(tablo.reindex(index=SEGMENTLER + ["Toplam"], columns=RISKLER + ["Toplam"], fill_value=0),
                         width="stretch")
    with sekme2, kutu("Müşteri listesi", "Segment ve risk seviyesine göre filtreleyin; liste churn olasılığına göre sıralıdır"):
        f1, f2 = st.columns(2)
        secili_segment = f1.multiselect("Segment", SEGMENTLER, default=SEGMENTLER)
        secili_risk = f2.multiselect("Risk seviyesi", RISKLER, default=["Yüksek"])

        liste = musteriler[musteriler["Segment"].isin(secili_segment) & musteriler["Risk"].isin(secili_risk)]
        liste = liste.sort_values("ChurnOlasiligi", ascending=False)
        liste = liste.assign(CustomerID=liste["CustomerID"].astype(int), OrtSepet=liste["OrtSepet"].round(2))
        st.caption(f"{sayi(len(liste))} müşteri listeleniyor.")
        st.dataframe(liste, hide_index=True, width="stretch")
        csv_indir(liste, "churn_listesi.csv", "Listeyi CSV olarak indir")


def oncelikli_sayfasi(musteriler):
    baslik_bandi("Öncelikli Müşteriler", "Şirkete çok kazandıran ama kaybedilme riski taşıyan müşteriler: "
                                         "ilk ulaşılması gereken kişiler.")
    secim, esik = oncelikli_musteriler(musteriler)

    not_kutusu("Ölçüt", [
        f"Toplam harcaması müşterilerin <b>en yüksek %25'inde</b> olan (Monetary ≥ <b>{para(esik, 2)}</b>) "
        "<b>ve</b> churn riski <b>Orta</b> veya <b>Yüksek</b> olan müşteriler. <b>Beklenen kayıp</b> = Monetary × "
        "churn olasılığı; geçmiş harcamaya dayanan kaba bir önem ölçüsüdür ve listeyi sıralamak için kullanılır."])

    kart_satiri([
        dict(etiket="Öncelikli müşteri", deger=sayi(len(secim)), alt="Ölçüte uyan müşteriler"),
        dict(etiket="Toplam harcamaları", deger=para(secim["Monetary"].sum()), alt="Geçmiş bir yıl"),
        dict(etiket="Ciro içindeki payı", deger=yuzde(secim["Monetary"].sum() / musteriler["Monetary"].sum() * 100),
             alt="Toplam ciroya göre"),
        dict(etiket="Toplam beklenen kayıp", deger=para(secim["BeklenenKayip"].sum()), renk=RISK_RENK["Yüksek"],
             alt="Harcama × churn olasılığı"),
    ])
    not_kutusu("Önerilen aksiyon", [ONCELIKLI_AKSIYON], renk=RISK_RENK["Düşük"])

    sol, sag = st.columns([2, 3])
    with sol, kutu("Segmentlere göre dağılım"):
        dagilim_grafigi(secim["Segment"], SEGMENTLER, SEGMENT_RENK, 390)
    with sag, kutu("Öncelikli müşteri listesi", "Beklenen kayba göre sıralı"):
        liste = secim[["CustomerID", "Segment", "Risk", "ChurnOlasiligi", "Recency", "Frequency",
                       "Monetary", "BeklenenKayip", "Oneri"]]
        liste = liste.assign(CustomerID=liste["CustomerID"].astype(int))
        st.dataframe(liste, hide_index=True, width="stretch", height=330)
        csv_indir(liste, "oncelikli_musteriler.csv", "Öncelikli müşterileri CSV olarak indir")


def risk_gostergesi(olasilik):
    grafik = go.Figure(go.Indicator(
        mode="gauge+number", value=olasilik * 100,
        number={"prefix": "%", "valueformat": ".1f", "font": {"size": 34}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": ANA_RENK, "thickness": 0.3},
               "steps": [{"range": [0, 40], "color": renk_saydam(RISK_RENK["Düşük"], 0.55)},
                         {"range": [40, 70], "color": renk_saydam(RISK_RENK["Orta"], 0.55)},
                         {"range": [70, 100], "color": renk_saydam(RISK_RENK["Yüksek"], 0.55)}]},
    ))
    grafik_goster(grafik, 240, yan_bosluk=40)


def sorgu_sayfasi(musteriler):
    baslik_bandi("Müşteri Sorgula", "Bir müşteri numarası girin; segmentini, riskini ve önerilen aksiyonu görün.")
    with kutu():
        girilen = st.text_input("Müşteri numarası (CustomerID)", placeholder="ör. 12347")
        st.caption(f"Örnek numaralar: {', '.join(str(int(i)) for i in musteriler['CustomerID'].head(3))}")
    if not girilen:
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
    bosluk()
    st.markdown(f"### Müşteri {int(secilen['CustomerID'])}")
    st.markdown(rozet(ETIKET[seg], SEGMENT_RENK[seg]) + rozet(secilen["Risk"] + " risk", RISK_RENK[secilen["Risk"]]),
                unsafe_allow_html=True)
    bosluk()

    sol, sag = st.columns([2, 3])
    with sol, kutu("Churn olasılığı", "Yeşil: düşük · sarı: orta · kırmızı: yüksek risk"):
        risk_gostergesi(secilen["ChurnOlasiligi"])
    with sag, kutu("Tüm müşterilere göre konumu"):
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

    kart_satiri([
        dict(etiket="Ortalama sepet", deger=para(secilen["OrtSepet"], 2), alt="Sipariş başına harcama"),
        dict(etiket="Farklı ürün", deger=sayi(secilen["UrunCesidi"]), alt="Satın aldığı ürün çeşidi"),
        dict(etiket="Müşteri süresi", deger=f"{sayi(secilen['IlkAlisveris'])} gün", alt="İlk alışverişten bu yana"),
    ])
    not_kutusu("Önerilen aksiyon", [ONERI[seg]], renk=SEGMENT_RENK[seg])


def yeni_veri_sayfasi():
    baslik_bandi("Yeni Veri Analizi", "Online Retail formatındaki kendi satış dosyanızı yükleyin; uygulama veriyi "
                                      "temizler, müşterileri segmentlere ayırır ve churn riskini hesaplar.")

    with kutu("Dosya yükle", "Gerekli sütunlar: " + ", ".join(GEREKLI_SUTUNLAR)):
        ornek = VERI / "ornek_islemler.csv"
        if ornek.exists():
            st.download_button("Örnek dosyayı indir", ornek.read_bytes(), file_name="ornek_islemler.csv",
                               mime="text/csv", icon=":material/description:")
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

    if sonuc is None:
        st.dataframe(rapor, hide_index=True, width="stretch")
        st.error("Temizleme sonrası analiz edilecek kayıt kalmadı.")
        return

    bosluk()
    kart_satiri([
        dict(etiket="Müşteri sayısı", deger=sayi(len(sonuc)), alt=f"{sayi(rapor['Kalan Satır'].iloc[-1])} temiz satırdan"),
        dict(etiket="Toplam ciro", deger=para(sonuc["Monetary"].sum()), alt="Temizlenmiş satışlar"),
        dict(etiket="Yüksek riskli müşteri", deger=sayi((sonuc["Risk"] == "Yüksek").sum()), renk=RISK_RENK["Yüksek"],
             alt="Churn olasılığı 0,7'nin üstünde"),
    ])

    sekme1, sekme2, sekme3 = st.tabs(["Özet", "Veri temizleme", "Müşteri sonuçları"])
    with sekme1:
        sol, sag = st.columns(2)
        with sol, kutu("Segment dağılımı"):
            dagilim_grafigi(sonuc["Segment"], SEGMENTLER, SEGMENT_RENK)
        with sag, kutu("Risk dağılımı"):
            dagilim_grafigi(sonuc["Risk"], RISKLER, RISK_RENK)
    with sekme2, kutu("Temizleme adımları", "Notebook 01–02'deki kurallar sırayla uygulandı"):
        st.dataframe(rapor, hide_index=True, width="stretch")
    with sekme3, kutu("Müşteri sonuçları", "Gün hesapları, dosyadaki en son tarihin bir gün sonrasına göre yapılır"):
        sonuc = sonuc.assign(CustomerID=sonuc["CustomerID"].astype(int), OrtSepet=sonuc["OrtSepet"].round(2),
                             Monetary=sonuc["Monetary"].round(2))
        st.dataframe(sonuc, hide_index=True, width="stretch")
        csv_indir(sonuc, "yeni_veri_sonuclari.csv", "Sonuçları CSV olarak indir")


def yontem_sayfasi():
    baslik_bandi("Yöntem", "Kullanılan veri, temizleme adımları, kümeleme ve churn modeli.")
    sekmeler = st.tabs(["Veri ve temizleme", "RFM ve kümeleme", "Segment geçişi", "Churn modeli"])
    with sekmeler[0], kutu():
        st.markdown("""
UCI Machine Learning Repository'deki **Online Retail** veri seti: İngiltere merkezli bir çevrimiçi perakende
firmasının Aralık 2010 – Aralık 2011 arasındaki işlem kayıtları. Tutarlar **sterlin (£)** cinsindendir.

| Adım | Çıkarılan | Kalan Satır |
|---|:---:|:---:|
| Başlangıç (ham veri) | — | 541.909 |
| Müşteri numarası boş kayıtlar | 135.080 | 406.829 |
| İptal/iade faturaları + sıfır/negatif adet–fiyat | 8.945 | 397.884 |
| Ürün dışı kodlar (POST, M, C2, DOT, BANK CHARGES, PADS) | 1.547 | 396.337 |
| Aynı gün tamamı iptal edilen çok büyük siparişlerin asıl satırları | 4 | 396.333 |
| Birebir yinelenen (kopya) satırlar | 5.187 | 391.146 |
| **Analize hazır veri** | — | **391.146 satır / 4.333 müşteri** |
""")
    with sekmeler[1], kutu():
        st.markdown("""
- **Recency:** Son alışverişten bu yana geçen gün
- **Frequency:** Benzersiz sipariş (fatura) sayısı
- **Monetary:** Toplam harcama (adet × birim fiyat)

Çarpık dağılımlar `log1p` ile düzeltilmiş, ardından `StandardScaler` ile ölçeklenmiştir.

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
""")
    with sekmeler[2], kutu():
        st.markdown("""
Veri iki döneme bölündü: 1. dönem Aralık 2010 – Mayıs 2011 (2.715 müşteri), 2. dönem Haziran – Aralık 2011
(3.540 müşteri). Ölçekleyici ve K-Means 1. dönemde eğitilip aynı model 2. döneme uygulandı; böylece iki dönemin
segmentleri karşılaştırılabilir oldu.
""")
    with sekmeler[3], kutu():
        st.markdown("""
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

st.markdown(CSS, unsafe_allow_html=True)
musteriler, gecis, gecisler = verileri_yukle()

SAYFALAR = {
    ":material/dashboard: Genel Bakış": lambda: genel_bakis(musteriler, gecis),
    ":material/groups: Segmentler": lambda: segmentler_sayfasi(musteriler),
    ":material/swap_horiz: Segment Geçişi": lambda: gecis_sayfasi(gecis, gecisler),
    ":material/warning: Churn Riski": lambda: churn_sayfasi(musteriler),
    ":material/flag: Öncelikli Müşteriler": lambda: oncelikli_sayfasi(musteriler),
    ":material/person_search: Müşteri Sorgula": lambda: sorgu_sayfasi(musteriler),
    ":material/upload_file: Yeni Veri Analizi": yeni_veri_sayfasi,
    ":material/menu_book: Yöntem": yontem_sayfasi,
}

with st.sidebar:
    st.markdown(f"<div class='logo'><div class='logo-ikon'>{LOGO_SVG}</div><div>"
                "<div class='logo-ad'>RFM Paneli</div><div class='logo-alt'>Müşteri Segmentasyonu</div></div></div>"
                "<div class='menu-baslik'>MENÜ</div>", unsafe_allow_html=True)
    sayfa = st.radio("Sayfa", list(SAYFALAR), label_visibility="collapsed")
    st.markdown(f"<div class='veri-bilgi'>Veri: UCI Online Retail · {sayi(len(musteriler))} müşteri<br>"
                "Aralık 2010 – Aralık 2011</div>", unsafe_allow_html=True)

SAYFALAR[sayfa]()
