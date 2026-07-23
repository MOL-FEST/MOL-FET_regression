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


try:
    from mordred import Calculator, descriptors
    # Mordred 2D descriptor hesaplamak için Calculator ve descriptor koleksiyonu çağrılır.
except Exception as e:
    raise ImportError("Mordred gerekli. Localde: pip install mordred. Colab'da ilk hücre bunu kurar.") from e
    # Mordred yoksa açık hata verilir.


def canonicalize_smiles(smiles):
    """SMILES stringini RDKit ile canonical SMILES formatına çevirir."""
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    # SMILES geçerliyse RDKit Mol nesnesi oluşturulur.
    if mol is None:
        return None, None
        # Geçersiz SMILES için None döndürülür.
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # Molekül canonical isomeric SMILES formatına çevrilir.
    return mol, canonical
    # Mol nesnesi ve canonical SMILES döndürülür.


def main():
    """01 numaralı feature store üretim adımını çalıştırır."""
    lesson_out = ensure_dir(OUTPUT_ROOT / "01_mordred_feature_store")
    # Bu adımın çıktı klasörü oluşturulur.
    df, target_col, smiles_col, id_col, source = read_training_data()
    # Ham veri ve kolon adları okunur.

    target = pd.to_numeric(df[target_col], errors="coerce")
    # Target sayısal formata çevrilir.
    mask = target.notna() & df[smiles_col].notna()
    # Target ve SMILES eksik olmayan satırlar seçilir.
    cleaned = df.loc[mask].copy().reset_index(drop=True)
    # Kullanılabilir satırlar yeni tabloya alınır.
    cleaned["Target"] = pd.to_numeric(cleaned[target_col], errors="coerce")
    # Standart Target kolonu oluşturulur.

    note("Target ve SMILES temizliği", f"Önceki satır: {len(df)}\nTemiz satır: {len(cleaned)}\nÇıkarılan satır: {len(df)-len(cleaned)}")
    # Temizlik özeti yazdırılır.
    plot_target_distribution(cleaned["Target"].to_numpy(), lesson_out / "target_distribution.png", "Regression target dağılımı")
    # Target dağılımı grafiği çizilir.

    molecules = []
    # Geçerli RDKit molekülleri burada tutulur.
    canonical_smiles = []
    # Canonical SMILES değerleri burada tutulur.
    valid_rows = []
    # Geçerli molekül satır indeksleri burada tutulur.

    for i, smi in enumerate(cleaned[smiles_col]):
        # Her SMILES değeri sırayla işlenir.
        mol, can = canonicalize_smiles(smi)
        # SMILES canonicalize edilir.
        if mol is not None:
            molecules.append(mol)
            # Geçerli molekül listeye eklenir.
            canonical_smiles.append(can)
            # Canonical SMILES listeye eklenir.
            valid_rows.append(i)
            # Geçerli satır indeksi saklanır.

    valid_df = cleaned.iloc[valid_rows].copy().reset_index(drop=True)
    # Geçerli SMILES satırları alınır.
    valid_df["SMILES"] = canonical_smiles
    # Canonical SMILES standart SMILES kolonuna yazılır.
    if id_col is not None:
        valid_df["MoleculeID"] = valid_df[id_col].astype(str)
        # ID kolonu varsa MoleculeID olarak kullanılır.
    else:
        valid_df["MoleculeID"] = [f"MOL_{i:05d}" for i in range(len(valid_df))]
        # ID kolonu yoksa otomatik ID üretilir.

    note("SMILES geçerlilik kontrolü", f"Geçerli molekül: {len(valid_df)}\nGeçersiz/okunamayan SMILES: {len(cleaned)-len(valid_df)}")
    # SMILES kontrol özeti yazdırılır.

    calc = Calculator(descriptors, ignore_3D=True)
    # Sadece 2D Mordred descriptor hesaplayacak calculator oluşturulur.
    note("2D Mordred descriptor hesaplanıyor", "Bu adım veri sayısına göre birkaç dakika sürebilir. Fingerprint feature üretilmez.")
    # Kullanıcıya işlem süresi ve descriptor türü açıklanır.
    mordred_df = calc.pandas(molecules)
    # 2D Mordred descriptor tablosu hesaplanır.
    mordred_df.columns = ["Mordred_" + str(c).replace(" ", "_") for c in mordred_df.columns]
    # Descriptor kolonlarına Mordred_ prefixi eklenir.
    mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")
    # Mordred hata objeleri ve sayısal olmayan değerler NaN yapılır.
    mordred_df = mordred_df.replace([np.inf, -np.inf], np.nan)
    # Sonsuz değerler NaN yapılır.

    missing_fraction = mordred_df.isna().mean()
    # Her descriptor için eksik oranı hesaplanır.
    non_empty_cols = missing_fraction[missing_fraction < 1.0].index.tolist()
    # Tamamı eksik olmayan descriptorlar tutulur.
    mordred_df = mordred_df[non_empty_cols]
    # Tamamen boş descriptor kolonları çıkarılır; bu feature selection değil, teknik temizliktir.

    nunique = mordred_df.nunique(dropna=True)
    # Her descriptor için benzersiz değer sayısı hesaplanır.
    non_constant_cols = nunique[nunique > 1].index.tolist()
    # Sabit olmayan descriptorlar tutulur.
    mordred_df = mordred_df[non_constant_cols]
    # Sabit descriptor kolonları çıkarılır; modeller için bilgi taşımazlar.

    feature_store = pd.concat(
        [valid_df[["MoleculeID", "SMILES", "Target"]].reset_index(drop=True), mordred_df.reset_index(drop=True)],
        axis=1,
    )
    # Meta kolonlar ve Mordred descriptorlar tek feature store tablosunda birleştirilir.

    descriptor_cols = mordred_feature_columns(feature_store)
    # Üretilen Mordred descriptor kolonları alınır.
    summary = pd.DataFrame([
        {
            "RawRows": len(df),
            "CleanRows": len(cleaned),
            "ValidMolecules": len(valid_df),
            "Mordred2DDescriptorCount": len(descriptor_cols),
            "TargetMean": feature_store["Target"].mean(),
            "TargetStd": feature_store["Target"].std(),
            "TargetMin": feature_store["Target"].min(),
            "TargetMax": feature_store["Target"].max(),
        }
    ])
    # Feature store özeti tabloya dönüştürülür.
    show_table(summary, title="Feature store özeti")
    # Özet tablo ekranda gösterilir.

    missing_report = pd.DataFrame({"Feature": descriptor_cols, "MissingFraction": feature_store[descriptor_cols].isna().mean().values})
    # Descriptor eksik oranları raporlanır.
    missing_report = missing_report.sort_values("MissingFraction", ascending=False)
    # En çok eksik içeren descriptorlar üste alınır.
    show_table(missing_report, n=15, title="En yüksek eksik oranlı 2D Mordred descriptorlar")
    # Eksik oranları ekranda gösterilir.

    save_csv(feature_store, lesson_out / "mordred_2d_features.csv")
    # Ana 2D Mordred feature store CSV olarak kaydedilir.
    save_csv(summary, lesson_out / "feature_store_index.csv")
    # Feature store özeti CSV olarak kaydedilir.
    save_csv(missing_report, lesson_out / "mordred_missingness_report.csv")
    # Eksik descriptor raporu CSV olarak kaydedilir.

    shutil.copy2(TRAINING_DATA_LOCAL if TRAINING_DATA_LOCAL.exists() else Path("training_data.csv"), EXPORT_ROOT / "training_data.csv") if TRAINING_DATA_LOCAL.exists() else None
    # Local training_data.csv varsa GitHub klasörüne kopyalanır.

    note("01 tamamlandı", f"Feature store: {lesson_out / 'mordred_2d_features.csv'}\nDescriptor sayısı: {len(descriptor_cols)}")
    # Final durum mesajı yazdırılır.


if __name__ == "__main__":
    main()
    # Script doğrudan çalıştırıldığında 01 adımı başlatılır.
