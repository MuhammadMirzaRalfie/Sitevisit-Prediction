# =====================================================
# SiteVisit Prediction Dashboard (Streamlit)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier

# =====================================================
# 1️⃣ Setup Streamlit Layout
# =====================================================
st.set_page_config(page_title="SiteVisit Prediction", layout="wide")
st.title("SiteVisit Prediction Dashboard")
st.markdown("""
Dashboard prediksi kebutuhan **Site Visit** berdasarkan data order pelanggan.
---
""")

# =====================================================
# Load Model dan Encoder
# =====================================================
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)

    try:
        model_path = os.path.join(base_path, "XGboost_sitevisit.pkl")
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.write("Isi direktori saat ini:", os.listdir(base_path))
        st.stop()

    try:
        enc_path = os.path.join(base_path, "encoder.pkl")
        encoders = joblib.load(enc_path)
    except Exception as e:
        st.error(f"Gagal memuat encoder: {e}")
        st.write("Isi direktori saat ini:", os.listdir(base_path))
        st.stop()

    return model, encoders

model, encoders = load_assets()

# =====================================================
#  Fungsi Encoding
# =====================================================
def encode_data(df, encoders):
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if col in encoders:
            le = encoders[col]
            # Jika encoder adalah LabelEncoder-like
            if hasattr(le, "classes_") and hasattr(le, "transform"):
                classes = set([str(x) for x in le.classes_])

                def safe_transform(val):
                    s = str(val)
                    if s in classes:
                        try:
                            return int(le.transform([s])[0])
                        except Exception:
                            return -1
                    return -1

                df_encoded[col] = df_encoded[col].map(safe_transform)
            # Jika encoder adalah mapping dict
            elif isinstance(le, dict):
                df_encoded[col] = df_encoded[col].map(lambda v: le.get(v, -1))
            else:
                # fallback: coba transform jika tersedia, else -1
                try:
                    df_encoded[col] = df_encoded[col].map(lambda v: int(le.transform([str(v)])[0]))
                except Exception:
                    df_encoded[col] = -1

    # Isi NaN dengan -1 dan coba konversi numeric untuk kolom numerik
    df_encoded = df_encoded.fillna(-1)
    for c in df_encoded.columns:
        try:
            df_encoded[c] = pd.to_numeric(df_encoded[c])
        except Exception:
            pass

    return df_encoded


def validate_and_align(df, encoders, model):
    """
    Pastikan dataframe memiliki kolom yang diharapkan oleh model/encoder.
    - Isi kolom yang hilang dengan -1
    - Laporkan kolom ekstra
    - Urutkan kolom sesuai model.feature_names_in_ jika tersedia, atau keys(encoders)
    """
    df_checked = df.copy()

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = list(encoders.keys())

    missing = [c for c in expected if c not in df_checked.columns]
    extra = [c for c in df_checked.columns if c not in expected]

    for col in missing:
        df_checked[col] = -1

    # Reorder columns to expected (if possible)
    try:
        df_checked = df_checked[expected]
    except Exception:
        # jika gagal, biarkan apa adanya
        pass

    return df_checked, missing, extra, expected

# =====================================================
# 4️⃣ Tabs (Upload CSV vs Manual Input)
# =====================================================
tab1, tab2 = st.tabs(["📂 Upload Dataset", " Input Manual"])

# =====================================================
# 5️⃣ TAB 1: Batch Prediction (Upload CSV)
# =====================================================
with tab1:
    st.subheader("📁 Upload Dataset untuk Prediksi ")

    uploaded_file = st.file_uploader("Upload file CSV (fitur sesuai template):", type=["csv"])

    if uploaded_file:
        # Baca data
        data = pd.read_csv(uploaded_file)
        st.write("### 📋 Data Awal", data.head())

        # Validasi dan align kolom sebelum encode
        aligned, missing_cols, extra_cols, expected = validate_and_align(data, encoders, model)
        if missing_cols:
            st.warning(f"Kolom hilang ditambahkan dengan default -1: {missing_cols}")
        if extra_cols:
            st.info(f"Ada kolom ekstra yang akan diabaikan untuk prediksi: {extra_cols}")

        # Encode dan Prediksi
        encoded = encode_data(aligned, encoders)
        preds = model.predict(encoded)
        probs = model.predict_proba(encoded)[:, 1]

        # Tambahkan hasil prediksi ke dataframe
        data["Predicted_SiteVisit"] = preds
        data["Probability"] = probs
        st.success("✅ Prediksi selesai!")

        # Pastikan kolom yang dibutuhkan ada
        if "Predicted_SiteVisit" not in data.columns or "Probability" not in data.columns:
            st.error("Hasil prediksi tidak memiliki kolom yang diharapkan.")
        else:
            # UI filter & sort
            st.write("###  Hasil Prediksi")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                filter_opt = st.selectbox("Filter Prediksi", ["Semua", "Needs Site Visit", "No Site Visit"])
            with col2:
                sort_opt = st.selectbox("Sort by Probability", ["None", "Probability (desc)", "Probability (asc)"])
            with col3:
                top_n = st.number_input("Tampilkan baris teratas (0 = semua)", min_value=0, value=20, step=5)

            display = data.copy()
            if filter_opt == "Needs Site Visit":
                display = display[display["Predicted_SiteVisit"] == 1]
            elif filter_opt == "No Site Visit":
                display = display[display["Predicted_SiteVisit"] == 0]

            if sort_opt == "Probability (desc)":
                display = display.sort_values("Probability", ascending=False)
            elif sort_opt == "Probability (asc)":
                display = display.sort_values("Probability", ascending=True)

            if top_n > 0:
                display_show = display.head(int(top_n))
            else:
                display_show = display

            # Tampilkan dataframe yang dapat di-scroll
            st.dataframe(display_show.reset_index(drop=True))

            # Download hasil (sesuai filter/sort)
            csv = display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Hasil Prediksi (filtered)",
                data=csv,
                file_name="sitevisit_predictions_filtered.csv",
                mime="text/csv"
            )

            # Statistik ringkas berdasarkan tampilan saat ini
            st.write("### Ringkasan Prediksi")
            # Pastikan ada nilai sebelum membuat chart
            if not display["Predicted_SiteVisit"].empty:
                st.bar_chart(display["Predicted_SiteVisit"].value_counts())
            else:
                st.info("Tidak ada baris untuk ditampilkan pada ringkasan.")

# =====================================================
# TAB 2: Manual Input Prediction
# =====================================================
with tab2:
    st.subheader(" Prediksi Manual")

    # Input field sesuai fitur
    ORDERTYPE = st.selectbox("Order Type", 
        ['New Installation', 'Addon', 'Relocation', 'Change Plan', 'Change Service',
         'Transfer Of Ownership', 'Unknown', 'Change Status', 'Cancellation'])

    PRODFAMILYNAME = st.text_input("Product Family Name (contoh: LA_NET)")
    FPA = st.selectbox("FPA", ['No', 'Yes', 'Unknown'])
    NEXT_PROCESS = st.selectbox("Next Process", 
        ['Unknown', 'closing', 'report_validation', 'aktivasi', 'task_dispatch', 
         'cancel', 'installasi', 'close', 'jm_approval', 'service_termination',
         'internet_termination', 'internet_setting', 'suspend', 'release_suspend', 
         'order_validation', 'result_verification', 'dismantle'])

    ISDS_ORDER_JAR = st.selectbox("ISDS Order JAR", ['PSB', 'MUTASI', 'CABUT'])
    MEDIAACCESS = st.selectbox("Media Access", 
        ['ETHERNET', 'VSAT', 'FO', 'RADIOLINK', 'Unknown', 'WIRELINE', 'WIRELESS',
         'BWA', 'BWA 3', 'SDL', 'SKDP', 'WLL', 'SL', 'BWA 2', 'HX50', 'wifi', 'FPA'])
    ORDERSUBTYPE = st.selectbox("Order Subtype", 
        ['New Installation', 'Subscribe', 'Relocation', 'Upgrade/Downgrade',
         'Change Access Media', 'Change Service', 'Change Backhaul Reference', 
         'Unknown', 'Terminate', 'Cancellation', 'Unsubscribe', 
         'Suspend Temporary', 'Release Suspension'])

    SOURCESITELAT = st.number_input("Source Site Latitude", value=6.239)
    SOURCESITELONG = st.number_input("Source Site Longitude", value=106.82)
    PREVIOUSTASKNAME = st.text_input("Previous Task Name", "Unknown")
    DESTINATIONPOPSITELAT = st.number_input("Destination POP Latitude", value=-6.239)
    DESTINATIONPOPSITELONG = st.number_input("Destination POP Longitude", value=106.823)
    KAMSUBCATEGORY = st.selectbox("KAM Subcategory", 
        ['Partner', 'Wholesale 2', 'SKA 2C', 'Undefined', 'Wholesale 1',
         'Small National Account', 'SKA 1B', 'National Account', 'New SKA 2', 
         'SKA 1A', 'SKA 2A', 'Big National Account', 'SKA 1 Regional', 'SKA 2B', 
         'KA Pusat', 'SKA 2 Regional', 'KA Regional', 'Small Regular', 'SKA 1', 
         'Key Partner', 'Regular Partner', 'Regular', 'EA 3', 'Big Regular', 
         'EA Regional', 'EA 2', 'Unknown', 'SKA 2', 'New EA', 'SKA Regional', 
         'SKA 1C', 'EA 1', 'Strategic Partner'])

    # Buat dataframe dari input
    input_data = pd.DataFrame([{
        'ORDERTYPE': ORDERTYPE,
        'PRODFAMILYNAME': PRODFAMILYNAME,
        'FPA': FPA,
        'NEXT_PROCESS': NEXT_PROCESS,
        'ISDS_ORDER_JAR': ISDS_ORDER_JAR,
        'MEDIAACCESS': MEDIAACCESS,
        'ORDERSUBTYPE': ORDERSUBTYPE,
        'SOURCESITELAT': SOURCESITELAT,
        'SOURCESITELONG': SOURCESITELONG,
        'PREVIOUSTASKNAME': PREVIOUSTASKNAME,
        'DESTINATIONPOPSITELAT': DESTINATIONPOPSITELAT,
        'DESTINATIONPOPSITELONG': DESTINATIONPOPSITELONG,
        'KAMSUBCATEGORY': KAMSUBCATEGORY
    }])

    if st.button("Prediksi"):
        # Validasi/align kolom input manual
        aligned_input, missing_cols_in, extra_cols_in, expected_in = validate_and_align(input_data, encoders, model)
        if missing_cols_in:
            st.warning(f"Kolom input ditambahkan dengan default -1: {missing_cols_in}")
        if extra_cols_in:
            st.info(f"Ada kolom ekstra pada input yang diabaikan: {extra_cols_in}")

        encoded_input = encode_data(aligned_input, encoders)
        pred = model.predict(encoded_input)[0]
        prob = model.predict_proba(encoded_input)[0][1]

        st.write("###  Hasil Prediksi")
        if pred == 1:
            st.success(f"✅ Order ini **MEMBUTUHKAN Site Visit** (Probabilitas: {prob:.2%})")
        else:
            st.warning(f"⚙️ Order ini **TIDAK membutuhkan Site Visit** (Probabilitas: {prob:.2%})")

        st.metric("Probabilitas Site Visit", f"{prob:.2%}")
