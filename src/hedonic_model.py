"""
Hedonik Fiyat Modeli: Konut Fiyatlarının Ekonometrik Analizi
================================================================
Ames Housing veri seti kullanılarak, konut özelliklerinin (yaşam alanı,
kalite, yaş, garaj, mahalle vb.) satış fiyatı üzerindeki etkisini
log-lineer bir hedonik regresyon modeliyle tahmin eder.

Veri kaynağı:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
(train.csv dosyasını indirip data/ klasörüne koyun)

Kullanım:
    python src/hedonic_model.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------------------------------------------
# 1. VERİ YÜKLEME VE TEMİZLEME
# ----------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Veri seti yüklendi: {df.shape[0]} gözlem, {df.shape[1]} değişken")
    return df


def prepare_data(df):
    """Hedonik modelde kullanılacak değişkenleri seç ve temizle."""

    cols = [
        "SalePrice", "GrLivArea", "LotArea", "OverallQual", "OverallCond",
        "YearBuilt", "TotalBsmtSF", "GarageCars", "FullBath", "BedroomAbvGr",
        "Neighborhood", "CentralAir",
    ]
    data = df[cols].copy()

    # Eksik değerleri düşür (bu değişkenlerde genelde çok az eksik olur)
    data = data.dropna()

    # Aykırı değerleri kırp (üst ve alt %0.5 - hedonik modellerde standart pratik)
    for col in ["SalePrice", "GrLivArea", "LotArea"]:
        low, high = data[col].quantile([0.005, 0.995])
        data = data[(data[col] >= low) & (data[col] <= high)]

    # Ev yaşı türetilmiş değişken
    data["HouseAge"] = 2010 - data["YearBuilt"]  # veri setinin son satış yılı ~2010

    # Kukla (dummy) değişkenler
    data["CentralAir"] = (data["CentralAir"] == "Y").astype(int)

    # En sık görülen 10 mahalleyi tut, diğerlerini "Other" olarak grupla
    top_neighborhoods = data["Neighborhood"].value_counts().nlargest(10).index
    data["Neighborhood"] = data["Neighborhood"].where(
        data["Neighborhood"].isin(top_neighborhoods), "Other"
    )

    # Bağımlı değişkeni logaritmik dönüştür (hedonik modellerde standart)
    data["log_price"] = np.log(data["SalePrice"])

    print(f"Temizlik sonrası: {data.shape[0]} gözlem kaldı")
    return data


# ----------------------------------------------------------------
# 2. MODEL TAHMİNİ
# ----------------------------------------------------------------

def fit_hedonic_model(data):
    """
    Log-lineer hedonik fiyat modeli:
    log(Fiyat) = b0 + b1*YaşamAlanı + b2*ArsaAlanı + b3*Kalite + ... + e

    Log-lineer form sayesinde katsayılar yaklaşık semi-elastikiyet olarak
    yorumlanır: b1 = 0.01 --> ilgili değişkende 1 birimlik artış fiyatı
    yaklaşık %1 artırır.
    """
    formula = (
        "log_price ~ GrLivArea + LotArea + OverallQual + OverallCond + "
        "HouseAge + TotalBsmtSF + GarageCars + FullBath + BedroomAbvGr + "
        "CentralAir + C(Neighborhood)"
    )
    model = smf.ols(formula=formula, data=data).fit(cov_type="HC3")  # heteroskedastisiteye dayanıklı hatalar
    return model


# ----------------------------------------------------------------
# 3. TANI TESTLERİ (DIAGNOSTICS)
# ----------------------------------------------------------------

def calculate_rmse(model, data):
    """
    RMSE (Root Mean Square Error) hesapla.
    
    Üç ayrı ölçüm:
    1. RMSE (log-scale): log(fiyat) tahminindeki hata
    2. RMSE (dollar): back-transformed tahminlerdeki dolar cinsinden ortalama hata
    3. MAPE (%): Mean Absolute Percentage Error - yüzde cinsinden ortalama mutlak hata
    """
    
    # Log-scale tahminler ve gerçek değerler
    y_pred_log = model.fittedvalues
    y_actual_log = data["log_price"]
    
    # Log-scale RMSE
    rmse_log = np.sqrt(np.mean((y_actual_log - y_pred_log) ** 2))
    
    # Dollar-scale (back-transformed) RMSE
    y_pred_dollar = np.exp(y_pred_log)
    y_actual_dollar = np.exp(y_actual_log)
    rmse_dollar = np.sqrt(np.mean((y_actual_dollar - y_pred_dollar) ** 2))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_actual_dollar - y_pred_dollar) / y_actual_dollar)) * 100
    
    # Yaklaşık yüzde hata (log-scale RMSE'den türetilmiş)
    # exp(RMSE_log) - 1 ≈ ortalama yüzde sapma
    approx_percentage_error = (np.exp(rmse_log) - 1) * 100
    
    return {
        "rmse_log": rmse_log,
        "rmse_dollar": rmse_dollar,
        "mape": mape,
        "approx_percentage_error": approx_percentage_error,
    }


def run_diagnostics(model, data):
    results = {}

    # Breusch-Pagan değişen varyans testi
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    results["breusch_pagan_pvalue"] = bp_test[1]

    # Jarque-Bera normallik testi
    jb_test = jarque_bera(model.resid)
    results["jarque_bera_pvalue"] = jb_test[1]

    # VIF (çoklu doğrusal bağlantı) - sadece sayısal değişkenler için
    numeric_vars = ["GrLivArea", "LotArea", "OverallQual", "OverallCond",
                     "HouseAge", "TotalBsmtSF", "GarageCars", "FullBath", "BedroomAbvGr"]
    X_vif = sm.add_constant(data[numeric_vars])
    vif_data = pd.DataFrame()
    vif_data["variable"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    results["vif"] = vif_data
    
    # RMSE hesaplamaları
    rmse_results = calculate_rmse(model, data)
    results["rmse_metrics"] = rmse_results

    return results


# ----------------------------------------------------------------
# 4. GÖRSELLEŞTİRME
# ----------------------------------------------------------------

def make_plots(model, data):
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gerçek vs tahmin
    axes[0].scatter(np.exp(model.fittedvalues), np.exp(data["log_price"]), alpha=0.3, s=15)
    lims = [data["SalePrice"].min(), data["SalePrice"].max()]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Tahmin Edilen Fiyat ($)")
    axes[0].set_ylabel("Gerçek Fiyat ($)")
    axes[0].set_title("Gerçek vs. Tahmin Edilen Fiyat")

    # Artık (residual) dağılımı
    axes[1].scatter(model.fittedvalues, model.resid, alpha=0.3, s=15)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Tahmin Edilen log(Fiyat)")
    axes[1].set_ylabel("Artık (Residual)")
    axes[1].set_title("Artık Dağılımı (Değişen Varyans Kontrolü)")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "model_diagnostics.png")
    plt.savefig(out_path, dpi=150)
    print(f"Grafik kaydedildi: {out_path}")


# ----------------------------------------------------------------
# 5. ANA AKIŞ
# ----------------------------------------------------------------

def main():
    df = load_data()
    data = prepare_data(df)
    model = fit_hedonic_model(data)

    print("\n" + "=" * 70)
    print("HEDONİK FİYAT MODELİ SONUÇLARI")
    print("=" * 70)
    print(model.summary())

    # Sonuçları txt dosyasına kaydet
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
        f.write(model.summary().as_text())

    diagnostics = run_diagnostics(model, data)
    print("\n" + "=" * 70)
    print("TANI TESTLERİ")
    print("=" * 70)
    print(f"Breusch-Pagan (değişen varyans) p-değeri: {diagnostics['breusch_pagan_pvalue']:.4f}")
    print("  -> p < 0.05 ise değişen varyans şüphesi var (HC3 zaten buna karşı dayanıklı)")
    print(f"Jarque-Bera (normallik) p-değeri: {diagnostics['jarque_bera_pvalue']:.4f}")
    print("\nVIF (Çoklu Doğrusal Bağlantı) Tablosu:")
    print(diagnostics["vif"].to_string(index=False))
    print("  -> VIF > 10 olan değişkenler ciddi çoklu doğrusal bağlantı işareti taşır")
    
    # RMSE sonuçları
    rmse_metrics = diagnostics["rmse_metrics"]
    print("\n" + "=" * 70)
    print("RMSE VE TAHMIN HATA METRİKLERİ")
    print("=" * 70)
    print(f"RMSE (log-scale):               {rmse_metrics['rmse_log']:.6f}")
    print(f"  -> Modelin log(fiyat) tahminindeki standart sapma")
    print(f"\nRMSE (dollar):                  ${rmse_metrics['rmse_dollar']:,.2f}")
    print(f"  -> Ortalama tahmin hatası (dolar cinsinden)")
    print(f"\nMAPE (Mean Absolute %Error):    {rmse_metrics['mape']:.2f}%")
    print(f"  -> Fiyat tahminindeki ortalama yüzde mutlak hata")
    print(f"\nYaklaşık Yüzde Hata:             {rmse_metrics['approx_percentage_error']:.2f}%")
    print(f"  -> Tipik tahmin sapması (log-scale'ten türetilmiş)")

    make_plots(model, data)

    print("\nTamamlandı. Sonuçlar outputs/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
