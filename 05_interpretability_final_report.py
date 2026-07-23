from pathlib import Path
# Dosya ve klasör yollarını güvenli şekilde yönetmek için kullanılır.
import warnings
# Gereksiz uyarıları kontrol etmek için kullanılır.
warnings.filterwarnings("ignore")
# Eğitim sırasında çıkan uzun uyarılar kapatılır; kritik hatalar yine görünür.

import os
# Çalışma ortamını ve Colab/local ayrımını kontrol etmek için kullanılır.
import shutil
# Çıktı dosyalarını GitHub klasörüne veya Google Drive yedeğine kopyalamak için kullanılır.
import numpy as np
# Sayısal matris, vektör ve metrik hesaplamaları için kullanılır.
import pandas as pd
# CSV okuma, tablo düzenleme ve sonuç kaydetme için kullanılır.
import matplotlib.pyplot as plt
# Veri dağılımı, model performansı ve yorumlanabilirlik grafiklerini çizmek için kullanılır.
import joblib
# Eğitilmiş modelleri ve pipeline nesnelerini dosyaya kaydetmek için kullanılır.

from sklearn.model_selection import train_test_split
# Veriyi train/test olarak bölmek için kullanılır.
from sklearn.impute import SimpleImputer
# Mordred descriptorlarında oluşabilecek eksik değerleri median ile doldurmak için kullanılır.
from sklearn.pipeline import Pipeline
# Ön işlem ve modeli tek nesne halinde birleştirmek için kullanılır.
from sklearn.preprocessing import StandardScaler
# Ölçekleme gereken modellerde descriptorları standartlaştırmak için kullanılır.
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
# Ağaç tabanlı regression modelleri ve ensemble yapıları için kullanılır.
from sklearn.linear_model import RidgeCV, LinearRegression
# Basit lineer regression ve stacking final estimator için kullanılır.
from sklearn.neighbors import KNeighborsRegressor
# Descriptor uzayında komşuluk tabanlı regression karşılaştırması için kullanılır.
from sklearn.svm import SVR
# Ölçeklenmiş descriptorlar üzerinde kernel regression alternatifi için kullanılır.
from sklearn.inspection import permutation_importance
# Model yorumlanabilirliği için permutation importance hesaplamakta kullanılır.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Regression performansını ölçmek için MAE, RMSE ve R2 hesaplamada kullanılır.

try:
    from IPython.display import display
    # Notebook ortamında tabloları daha okunabilir göstermek için kullanılır.
except Exception:
    display = None
    # Terminal ortamında display yoksa None olarak bırakılır.

try:
    from rdkit import Chem, DataStructs
    # SMILES okuma ve Tanimoto similarity için RDKit kullanılır.
    from rdkit.Chem import rdFingerprintGenerator
    # Tanimoto filtreleme için Morgan fingerprint hesaplamakta kullanılır.
except Exception as e:
    raise ImportError(
        "RDKit gerekli. Localde conda rdkit ortamını aktive edin. Örnek: conda activate rdkit_env. "
        "Colab'da notebookun ilk paket hücresi rdkit-pypi kurmayı dener."
    ) from e
    # RDKit olmadan SMILES ve Tanimoto işlemleri yapılamaz.

RANDOM_STATE = 42
# Train/test split ve model eğitimlerinde tekrarlanabilirlik için sabit seed kullanılır.
TEST_SIZE = 0.20
# Test set oranı %20 olarak belirlenir.
GITHUB_REPO_BASE = "https://raw.githubusercontent.com/MOL-FEST/MOL-FET_regression/main"
# Notebooklar GitHub'a yüklendikten sonra ham veri ve checkpoint dosyalarını buradan okuyabilir.
EXPORT_ROOT = Path("MOL_FET_regression_github")
# Local çalıştırmada GitHub'a yüklenecek klasörün adı.
if Path.cwd().name == "MOL_FET_regression_github":
    EXPORT_ROOT = Path(".")
    # Script doğrudan GitHub klasörünün içinden çalışıyorsa iç içe klasör oluşturulmaz.
OUTPUT_ROOT = EXPORT_ROOT / "molfet_regression_outputs"
# Bütün ara çıktılar bu ana klasör altında tutulur.
DRIVE_OUTPUT_ROOT = Path("/content/drive/MyDrive/MOL_FET_regression_outputs")
# Colab'da Google Drive yedeği istenirse kullanılacak ana klasör.
USE_GOOGLE_DRIVE_BACKUP = False
# True yapılırsa çıktıların Google Drive'a da kopyalanması denenir.
TARGET_COLUMN_CANDIDATES = ["pStandard_mean", "pStandard", "pIC50", "target", "Target", "y"]
# Regression target kolonu için sırayla denenecek olası kolon adları.
SMILES_COLUMN_CANDIDATES = ["parent_smiles", "canonical_smiles", "smiles", "SMILES", "QSAR-Ready SMILES"]
# SMILES kolonu için sırayla denenecek olası kolon adları.
ID_COLUMN_CANDIDATES = ["parent_chembl_id", "molecule_chembl_id", "chembl_id", "ID", "id"]
# Molekül ID kolonu için sırayla denenecek olası kolon adları.
TRAINING_DATA_LOCAL = Path("training_data.csv")
# Localde aranacak ham veri dosyası.
TRAINING_DATA_GITHUB = f"{GITHUB_REPO_BASE}/training_data.csv"
# Localde yoksa okunacak GitHub raw training_data.csv linki.
FEATURE_FILE = OUTPUT_ROOT / "01_mordred_feature_store" / "mordred_2d_features.csv"
# Localde üretilecek veya okunacak 2D Mordred feature dosyası.
FEATURE_FILE_GITHUB = f"{GITHUB_REPO_BASE}/molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv"
# Colab'da checkpoint olarak okunabilecek GitHub raw Mordred feature dosyası.


def ensure_dir(path):
    """Klasör yoksa oluşturur ve Path nesnesi olarak döndürür."""
    path = Path(path)
    # Girdi Path formatına çevrilir.
    path.mkdir(parents=True, exist_ok=True)
    # Klasör ve üst klasörler yoksa oluşturulur.
    return path
    # Oluşturulan klasör döndürülür.


def note(title, message=""):
    """Terminal/Colab çıktısında ders notu gibi okunabilir başlık üretir."""
    print("\n" + "=" * 100)
    # Başlık üst çizgisi basılır.
    print(title)
    # Başlık metni basılır.
    print("=" * 100)
    # Başlık alt çizgisi basılır.
    if message:
        print(message)
        # Açıklama varsa ekrana yazdırılır.


def show_table(df, n=20, title=None):
    """Tabloyu Colab'da display, terminalde text olarak gösterir."""
    if title:
        print("\n" + title)
        # Tablo başlığı yazdırılır.
    if display is not None:
        display(df.head(n))
        # Notebook ortamında tablo görsel olarak gösterilir.
    else:
        print(df.head(n).to_string(index=False))
        # Terminalde tablo metin olarak gösterilir.


def save_csv(df, path):
    """DataFrame'i CSV olarak kaydeder."""
    path = Path(path)
    # Kayıt yolu Path nesnesine çevrilir.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Üst klasör yoksa oluşturulur.
    df.to_csv(path, index=False)
    # CSV dosyası index yazılmadan kaydedilir.
    print(f"[Kaydedildi] {path}")
    # Kaydedilen dosya yolu ekrana yazdırılır.
    backup_to_drive(path)
    # Google Drive yedeği açıksa dosya Drive'a kopyalanır.


def backup_to_drive(path):
    """İstenirse çıktıyı Google Drive yedek klasörüne kopyalar."""
    if not USE_GOOGLE_DRIVE_BACKUP:
        return
        # Drive yedeği kapalıysa işlem yapılmaz.
    path = Path(path)
    # Dosya yolu Path nesnesine çevrilir.
    if not path.exists():
        return
        # Dosya yoksa kopyalama yapılmaz.
    try:
        relative = path.relative_to(OUTPUT_ROOT)
        # Çıktı klasörüne göre göreli yol hesaplanır.
        drive_path = DRIVE_OUTPUT_ROOT / relative
        # Drive içinde aynı göreli yapı oluşturulur.
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        # Drive üst klasörü yoksa oluşturulur.
        shutil.copy2(path, drive_path)
        # Dosya Drive'a kopyalanır.
        print(f"[Drive yedeği] {drive_path}")
        # Drive yedek yolu yazdırılır.
    except Exception as e:
        print(f"[Drive yedeği atlandı] {e}")
        # Drive bağlı değilse veya hata varsa pipeline durdurulmaz.


def detect_column(df, candidates, label):
    """Aday kolon listesinden veri içinde bulunan ilk kolonu döndürür."""
    for col in candidates:
        # Aday kolonlar sırayla gezilir.
        if col in df.columns:
            return col
            # İlk bulunan kolon döndürülür.
    raise ValueError(f"{label} kolonu bulunamadı. Adaylar: {candidates}. Mevcut kolonlar: {list(df.columns)}")
    # Hiçbiri bulunamazsa açık hata verilir.


def read_training_data():
    """training_data.csv dosyasını localden, localde yoksa GitHub raw linkinden okur."""
    if TRAINING_DATA_LOCAL.exists():
        source = str(TRAINING_DATA_LOCAL)
        # Local dosya varsa kaynak localdir.
        df = pd.read_csv(TRAINING_DATA_LOCAL)
        # CSV localden okunur.
    else:
        source = TRAINING_DATA_GITHUB
        # Local dosya yoksa GitHub raw kaynak kullanılır.
        df = pd.read_csv(TRAINING_DATA_GITHUB)
        # CSV GitHub raw linkinden okunur.
    target_col = detect_column(df, TARGET_COLUMN_CANDIDATES, "Target")
    # Regression hedef kolonu belirlenir.
    smiles_col = detect_column(df, SMILES_COLUMN_CANDIDATES, "SMILES")
    # SMILES kolonu belirlenir.
    try:
        id_col = detect_column(df, ID_COLUMN_CANDIDATES, "ID")
        # ID kolonu varsa belirlenir.
    except ValueError:
        id_col = None
        # ID kolonu yoksa None kalır.
    note(
        "Ham veri okundu",
        f"Kaynak: {source}\nSatır sayısı: {df.shape[0]}\nKolon sayısı: {df.shape[1]}\nTarget: {target_col}\nSMILES: {smiles_col}\nID: {id_col}"
    )
    # Veri özeti ekrana basılır.
    return df, target_col, smiles_col, id_col, source
    # Veri ve kolon adları döndürülür.


def read_feature_store():
    """2D Mordred feature store dosyasını localden, yoksa GitHub raw linkinden okur."""
    if FEATURE_FILE.exists():
        source = str(FEATURE_FILE)
        # Local feature dosyası varsa local kaynak kullanılır.
        df = pd.read_csv(FEATURE_FILE)
        # Feature CSV localden okunur.
    else:
        source = FEATURE_FILE_GITHUB
        # Local feature yoksa GitHub checkpoint kaynak kullanılır.
        df = pd.read_csv(FEATURE_FILE_GITHUB)
        # Feature CSV GitHub raw linkinden okunur.
    note("2D Mordred feature store okundu", f"Kaynak: {source}\nSatır: {df.shape[0]}\nKolon: {df.shape[1]}")
    # Feature store özeti yazdırılır.
    return df, source
    # Feature tablosu ve kaynak bilgisi döndürülür.


def mordred_feature_columns(df):
    """Sadece 2D Mordred descriptor kolonlarını seçer."""
    cols = [c for c in df.columns if c.startswith("Mordred_")]
    # Mordred_ prefixli kolonlar model feature olarak seçilir.
    if not cols:
        raise ValueError("Mordred_ ile başlayan descriptor kolonu bulunamadı. Önce 01 scriptini çalıştırın.")
        # Feature yoksa açık hata verilir.
    return cols
    # Descriptor kolonları döndürülür.


def prepare_xy(feature_df, feature_cols=None):
    """Feature tablosundan X, y, ID ve SMILES alanlarını hazırlar."""
    if feature_cols is None:
        feature_cols = mordred_feature_columns(feature_df)
        # Feature listesi verilmezse tüm 2D Mordred descriptorları kullanılır.
    X = feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Descriptor matrisi sayısal hale getirilir; inf değerler NaN yapılır.
    y = pd.to_numeric(feature_df["Target"], errors="coerce")
    # Target sayısal hale getirilir.
    valid = y.notna()
    # Target eksik olmayan satırlar seçilir.
    X = X.loc[valid].reset_index(drop=True)
    # Feature matrisi geçerli satırlara göre filtrelenir.
    y = y.loc[valid].reset_index(drop=True).to_numpy(dtype=float)
    # Target vektörü numpy float array yapılır.
    meta = feature_df.loc[valid, [c for c in ["MoleculeID", "SMILES", "Target"] if c in feature_df.columns]].reset_index(drop=True)
    # ID, SMILES ve target gibi açıklayıcı kolonlar saklanır.
    return X, y, meta, feature_cols
    # Model matrisi, hedef vektör, meta tablo ve feature adları döndürülür.


def fixed_train_test_split(X, y, meta):
    """Tüm regression scriptlerinde aynı train/test bölmesini oluşturur."""
    idx = np.arange(len(y))
    # Satır indeksleri oluşturulur.
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Regression için sabit random split yapılır.
    X_train = X.iloc[train_idx].reset_index(drop=True)
    # Train feature matrisi alınır.
    X_test = X.iloc[test_idx].reset_index(drop=True)
    # Test feature matrisi alınır.
    y_train = y[train_idx]
    # Train target vektörü alınır.
    y_test = y[test_idx]
    # Test target vektörü alınır.
    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    # Train meta bilgileri alınır.
    meta_test = meta.iloc[test_idx].reset_index(drop=True)
    # Test meta bilgileri alınır.
    return X_train, X_test, y_train, y_test, meta_train, meta_test
    # Splitlenmiş veriler döndürülür.


def regression_metrics(y_true, y_pred):
    """Regression performans metriklerini hesaplar."""
    mae = mean_absolute_error(y_true, y_pred)
    # Ortalama mutlak hata hesaplanır.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # RMSE hesaplanır; sklearn sürüm bağımlılığı olmaması için karekök elle alınır.
    r2 = r2_score(y_true, y_pred)
    # R2 skoru hesaplanır.
    mbe = float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))
    # Mean bias error hesaplanır; pozitif değer overprediction anlamına gelir.
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MBE": mbe}
    # Metrik sözlüğü döndürülür.


def make_rf_pipeline(n_estimators=500, random_state=RANDOM_STATE):
    """2D Mordred descriptorları için Random Forest regression pipeline kurar."""
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # Eksik Mordred descriptor değerleri train median ile doldurulur.
        ("model", RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, max_features="sqrt")),
        # Random Forest regression modeli kurulur.
    ])
    # İmputer ve model tek pipeline içinde tutulur.
    return model
    # Pipeline döndürülür.


def save_predictions(meta_test, y_test, y_pred, path):
    """Test tahminlerini CSV olarak kaydeder."""
    pred_df = meta_test.copy()
    # Test meta bilgileri kopyalanır.
    pred_df["y_true"] = y_test
    # Gerçek değerler eklenir.
    pred_df["y_pred"] = y_pred
    # Model tahminleri eklenir.
    pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
    # Hata değeri eklenir.
    pred_df["abs_error"] = pred_df["error"].abs()
    # Mutlak hata değeri eklenir.
    save_csv(pred_df, path)
    # Tahmin tablosu kaydedilir.
    return pred_df
    # Tahmin tablosu döndürülür.


def plot_target_distribution(y, out_path, title="Target dağılımı"):
    """Regression target dağılımını histogram olarak çizer."""
    plt.figure(figsize=(7, 4))
    # Grafik boyutu belirlenir.
    plt.hist(y, bins=30)
    # Target değerlerinin histogramı çizilir.
    plt.xlabel("Target")
    # X ekseni etiketi yazılır.
    plt.ylabel("Molekül sayısı")
    # Y ekseni etiketi yazılır.
    plt.title(title)
    # Grafik başlığı yazılır.
    plt.tight_layout()
    # Grafik yerleşimi düzenlenir.
    out_path = Path(out_path)
    # Çıktı yolu Path nesnesine çevrilir.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Çıktı klasörü oluşturulur.
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # Grafik PNG olarak kaydedilir.
    plt.show()
    # Grafik ekranda gösterilir.
    backup_to_drive(out_path)
    # Google Drive yedeği açıksa grafik Drive'a kopyalanır.


def plot_predicted_vs_actual(y_true, y_pred, out_path, title="Predicted vs actual"):
    """Gerçek ve tahmin edilen target değerlerini scatter plot ile gösterir."""
    plt.figure(figsize=(5, 5))
    # Kare grafik alanı oluşturulur.
    plt.scatter(y_true, y_pred, alpha=0.7)
    # Gerçek-tahmin noktaları çizilir.
    low = min(np.min(y_true), np.min(y_pred))
    # Diagonal çizgi için minimum değer alınır.
    high = max(np.max(y_true), np.max(y_pred))
    # Diagonal çizgi için maksimum değer alınır.
    plt.plot([low, high], [low, high], linestyle="--")
    # İdeal tahmin çizgisi çizilir.
    plt.xlabel("Gerçek değer")
    # X ekseni etiketi yazılır.
    plt.ylabel("Tahmin")
    # Y ekseni etiketi yazılır.
    plt.title(title)
    # Grafik başlığı yazılır.
    plt.tight_layout()
    # Grafik yerleşimi düzenlenir.
    out_path = Path(out_path)
    # Çıktı yolu Path nesnesine çevrilir.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Çıktı klasörü oluşturulur.
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # Grafik PNG olarak kaydedilir.
    plt.show()
    # Grafik ekranda gösterilir.
    backup_to_drive(out_path)
    # Google Drive yedeği açıksa grafik Drive'a kopyalanır.


def make_interpretability_rf(X_train, y_train):
    """Yorumlanabilirlik için RF modelini impute edilmiş Mordred descriptorlarıyla eğitir."""
    imputer = SimpleImputer(strategy="median")
    # Eksik descriptor değerleri için median imputer kurulur.
    X_train_i = imputer.fit_transform(X_train)
    # Train descriptorları median ile doldurulur.
    model = RandomForestRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1, max_features="sqrt")
    # Yorumlanabilirlik için Random Forest regressor kurulur.
    model.fit(X_train_i, y_train)
    # Model impute edilmiş train verisi üzerinde eğitilir.
    return imputer, model, X_train_i
    # İmputer, model ve impute train matrisi döndürülür.


def read_optional_csv(path):
    """CSV varsa okur, yoksa None döndürür."""
    path = Path(path)
    # Dosya yolu Path nesnesine çevrilir.
    if path.exists():
        return pd.read_csv(path)
        # Dosya varsa okunur.
    return None
    # Dosya yoksa None döndürülür.


def main():
    """05 numaralı interpretability ve final rapor adımını çalıştırır."""
    lesson_out = ensure_dir(OUTPUT_ROOT / "05_interpretability_final_report")
    # Bu adımın çıktı klasörü oluşturulur.
    feature_df, source = read_feature_store()
    # 2D Mordred feature store okunur.
    X, y, meta, feature_cols = prepare_xy(feature_df)
    # Model matrisi hazırlanır.
    X_train, X_test, y_train, y_test, meta_train, meta_test = fixed_train_test_split(X, y, meta)
    # Sabit train/test bölmesi yapılır.
    imputer, rf_model, X_train_i = make_interpretability_rf(X_train, y_train)
    # Yorumlanabilirlik için RF modeli eğitilir.
    X_test_i = imputer.transform(X_test)
    # Test descriptorları aynı imputer ile dönüştürülür.
    y_pred = rf_model.predict(X_test_i)
    # Test tahminleri alınır.
    metrics = regression_metrics(y_test, y_pred)
    # Regression metrikleri hesaplanır.
    metrics.update({"Model": "Interpretability_RF", "FeatureSet": "Mordred2D", "FeatureCount": len(feature_cols)})
    # Metrik tablosuna model bilgileri eklenir.
    metrics_df = pd.DataFrame([metrics])
    # Metrik sözlüğü tabloya çevrilir.
    save_csv(metrics_df, lesson_out / "interpretability_rf_metrics.csv")
    # Yorumlanabilirlik RF metrikleri kaydedilir.
    show_table(metrics_df, title="Interpretability RF metrikleri")
    # Metrikler ekranda gösterilir.

    pred_df = save_predictions(meta_test, y_test, y_pred, lesson_out / "interpretability_rf_predictions.csv")
    # Test tahminleri kaydedilir.
    plot_predicted_vs_actual(y_test, y_pred, lesson_out / "interpretability_rf_predicted_vs_actual.png", "Interpretability RF predicted vs actual")
    # Predicted vs actual grafiği çizilir.

    note("Permutation importance hesaplanıyor", "Bu analiz descriptor değerleri bozulduğunda model hatasının ne kadar değiştiğini ölçer.")
    # Permutation importance açıklaması yazdırılır.
    perm = permutation_importance(rf_model, X_test_i, y_test, scoring="neg_mean_absolute_error", n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
    # Test set üzerinde permutation importance hesaplanır.
    perm_df = pd.DataFrame({"Feature": feature_cols, "ImportanceMean_MAEDecrease": perm.importances_mean, "ImportanceStd": perm.importances_std})
    # Importance sonuçları tabloya dönüştürülür.
    perm_df = perm_df.sort_values("ImportanceMean_MAEDecrease", ascending=False)
    # En önemli descriptorlar üste alınır.
    save_csv(perm_df, lesson_out / "permutation_importance_top_features.csv")
    # Permutation importance tablosu kaydedilir.
    show_table(perm_df, n=25, title="Top 25 permutation importance descriptor")
    # Top featurelar ekranda gösterilir.

    top_perm = perm_df.head(25).iloc[::-1]
    # Grafik için top 25 feature ters sıralanır.
    plt.figure(figsize=(8, 7))
    # Grafik boyutu belirlenir.
    plt.barh(top_perm["Feature"], top_perm["ImportanceMean_MAEDecrease"])
    # Permutation importance bar chart çizilir.
    plt.xlabel("MAE decrease when descriptor is useful")
    # X ekseni etiketi yazılır.
    plt.title("Top 25 2D Mordred descriptor — permutation importance")
    # Grafik başlığı yazılır.
    plt.tight_layout()
    # Yerleşim düzenlenir.
    plt.savefig(lesson_out / "permutation_importance_top25.png", dpi=300, bbox_inches="tight")
    # Grafik kaydedilir.
    plt.show()
    # Grafik ekranda gösterilir.

    try:
        import shap
        # SHAP paketi çağrılır.
        sample_n = min(250, X_test_i.shape[0])
        # SHAP için maksimum 250 test örneği kullanılır.
        rng = np.random.RandomState(RANDOM_STATE)
        # Tekrarlanabilir örnekleme için RNG kurulur.
        sample_idx = rng.choice(np.arange(X_test_i.shape[0]), size=sample_n, replace=False)
        # SHAP hesaplanacak test örnekleri seçilir.
        explainer = shap.TreeExplainer(rf_model)
        # Random Forest için TreeExplainer kurulur.
        shap_values = explainer.shap_values(X_test_i[sample_idx])
        # SHAP değerleri hesaplanır.
        shap.summary_plot(shap_values, pd.DataFrame(X_test_i[sample_idx], columns=feature_cols), show=False, max_display=25)
        # SHAP beeswarm grafiği oluşturulur.
        plt.tight_layout()
        # Yerleşim düzenlenir.
        plt.savefig(lesson_out / "shap_beeswarm_top25.png", dpi=300, bbox_inches="tight")
        # SHAP beeswarm kaydedilir.
        plt.show()
        # Grafik ekranda gösterilir.
        shap.summary_plot(shap_values, pd.DataFrame(X_test_i[sample_idx], columns=feature_cols), plot_type="bar", show=False, max_display=25)
        # SHAP bar grafiği oluşturulur.
        plt.tight_layout()
        # Yerleşim düzenlenir.
        plt.savefig(lesson_out / "shap_bar_top25.png", dpi=300, bbox_inches="tight")
        # SHAP bar grafiği kaydedilir.
        plt.show()
        # Grafik ekranda gösterilir.
        shap_abs = np.abs(shap_values).mean(axis=0)
        # Ortalama mutlak SHAP değeri hesaplanır.
        shap_df = pd.DataFrame({"Feature": feature_cols, "MeanAbsSHAP": shap_abs}).sort_values("MeanAbsSHAP", ascending=False)
        # SHAP feature tablosu oluşturulur.
        save_csv(shap_df, lesson_out / "shap_mean_abs_importance.csv")
        # SHAP importance tablosu kaydedilir.
    except Exception as e:
        print(f"[SHAP atlandı] {e}")
        # SHAP çalışmazsa pipeline durmaz.

    try:
        from lime.lime_tabular import LimeTabularExplainer
        # LIME tabular explainer çağrılır.
        explainer = LimeTabularExplainer(X_train_i, feature_names=feature_cols, mode="regression", random_state=RANDOM_STATE)
        # Regression modu için LIME explainer kurulur.
        low_error_idx = int(pred_df["abs_error"].idxmin())
        # En düşük hata örneği seçilir.
        high_error_idx = int(pred_df["abs_error"].idxmax())
        # En yüksek hata örneği seçilir.
        for label, idx in [("low_error", low_error_idx), ("high_error", high_error_idx)]:
            # İki lokal açıklama sırayla üretilir.
            exp = explainer.explain_instance(X_test_i[idx], rf_model.predict, num_features=15)
            # LIME lokal açıklaması hesaplanır.
            html_path = lesson_out / f"lime_{label}_explanation.html"
            # HTML çıktı yolu hazırlanır.
            exp.save_to_file(str(html_path))
            # LIME açıklaması HTML olarak kaydedilir.
            print(f"[Kaydedildi] {html_path}")
            # Kayıt yolu yazdırılır.
    except Exception as e:
        print(f"[LIME atlandı] {e}")
        # LIME çalışmazsa pipeline durmaz.

    residual_df = pred_df.copy()
    # Residual analizi için tahmin tablosu kopyalanır.
    plt.figure(figsize=(7, 4))
    # Residual grafiği için alan oluşturulur.
    plt.scatter(residual_df["y_pred"], residual_df["error"], alpha=0.7)
    # Tahmin değerine karşı hata çizilir.
    plt.axhline(0, linestyle="--")
    # Sıfır hata çizgisi eklenir.
    plt.xlabel("Predicted target")
    # X ekseni etiketi yazılır.
    plt.ylabel("Error")
    # Y ekseni etiketi yazılır.
    plt.title("Residual plot")
    # Grafik başlığı yazılır.
    plt.tight_layout()
    # Yerleşim düzenlenir.
    plt.savefig(lesson_out / "residual_plot.png", dpi=300, bbox_inches="tight")
    # Residual grafiği kaydedilir.
    plt.show()
    # Grafik ekranda gösterilir.

    baseline_metrics = read_optional_csv(OUTPUT_ROOT / "02_rf_mordred_baseline" / "rf_mordred_baseline_metrics.csv")
    # Baseline metrikleri varsa okunur.
    tanimoto_metrics = read_optional_csv(OUTPUT_ROOT / "03_tanimoto_similarity_filtration" / "tanimoto_filtration_metrics.csv")
    # Tanimoto filtration metrikleri varsa okunur.
    lazy_table = read_optional_csv(OUTPUT_ROOT / "04_lazypredict_regression" / "lazypredict_regression_model_table.csv")
    # LazyPredict tablosu varsa okunur.

    final_summary_rows = []
    # Final özet satırları burada tutulur.
    if baseline_metrics is not None:
        row = baseline_metrics.iloc[0].to_dict()
        # Baseline ilk satırı alınır.
        row["Section"] = "02_RF_Mordred_Baseline"
        # Bölüm adı eklenir.
        final_summary_rows.append(row)
        # Baseline özete eklenir.
    final_row = metrics_df.iloc[0].to_dict()
    # Interpretability RF satırı alınır.
    final_row["Section"] = "05_Interpretability_RF"
    # Bölüm adı eklenir.
    final_summary_rows.append(final_row)
    # Interpretability RF özete eklenir.
    if lazy_table is not None and not lazy_table.empty:
        top_lazy = lazy_table.sort_values("R2", ascending=False).iloc[0].to_dict()
        # LazyPredict tablosundaki en yüksek R2'li model rapor için alınır.
        top_lazy["Section"] = "04_LazyPredict_Top_Row_ReportOnly"
        # Bu satırın model seçimi değil rapor amaçlı olduğu belirtilir.
        final_summary_rows.append(top_lazy)
        # LazyPredict top satırı özete eklenir.

    final_summary_df = pd.DataFrame(final_summary_rows)
    # Final özet tablo oluşturulur.
    save_csv(final_summary_df, lesson_out / "final_regression_summary.csv")
    # Final özet tablo kaydedilir.
    show_table(final_summary_df, n=10, title="Final regression summary")
    # Final özet ekranda gösterilir.

    if tanimoto_metrics is not None:
        show_table(tanimoto_metrics, n=20, title="Tanimoto AD/filtration checkpoint özeti")
        # Tanimoto AD tablosu ekranda gösterilir.

    joblib.dump({"imputer": imputer, "model": rf_model, "feature_cols": feature_cols}, lesson_out / "interpretability_rf_model_bundle.joblib")
    # Model, imputer ve feature listesi tek bundle olarak kaydedilir.
    print(f"[Kaydedildi] {lesson_out / 'interpretability_rf_model_bundle.joblib'}")
    # Bundle kayıt yolu yazdırılır.

    note("05 tamamlandı", "Regression v3 akışı tamamlandı: model featureları sadece 2D Mordred descriptorlarıdır.")
    # Final mesajı yazdırılır.


if __name__ == "__main__":
    main()
    # Script doğrudan çalıştırıldığında 05 adımı başlatılır.
