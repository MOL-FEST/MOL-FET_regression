# MOL-FET Regression Workshop

Bu depo, bir **ChEMBL Target ID** üzerinden ligand verisi toplamadan başlayıp 2D Mordred descriptor üretimi, LazyPredict regresyon karşılaştırması, final ağaç modeli, applicability domain ve SHAP yorumlanabilirlik analizine kadar ilerleyen beş notebookluk bir workshop akışı içerir.

Bu güncel sürümde model girdisi olarak yalnızca **2D Mordred descriptorları** kullanılır. Morgan, MACCS, Avalon veya başka fingerprint türleri model feature olarak kullanılmaz.

## Güncel notebook sırası

```text
00_chembl_target_data_store.ipynb
01_data_cleaning.ipynb
02_feature_generation.ipynb
03_lazypredict.ipynb
04_SHAP_interpretability.ipynb
```

## Genel veri akışı

```text
ChEMBL Target ID
        ↓
IC50 ve single-protein assay verisi
        ↓
Parent SMILES ve parent-dedup regression CSV
        ↓
Tuz / karşı iyon temizliği
        ↓
Canonical SMILES ve duplicate birleştirme
        ↓
Lipinski filtresi
        ↓
2D Mordred feature store
        ↓
LazyPredict regression benchmark
        ↓
Final ağaç modeli ve 5-fold CV
        ↓
PCA-leverage applicability domain
        ↓
Williams plot ve SHAP yorumlanabilirlik
```

## Depo

```text
https://github.com/MOL-FEST/MOL-FET_regression
```

## Çalışma mantığı

Notebooklar Google Colab için hazırlanmıştır.

- Üretilen dosyalar Google Drive içindeki ortak workshop klasörüne kaydedilir.
- Sonraki notebookun ihtiyaç duyduğu ana CSV, ilgili notebook çalıştırıldıktan sonra GitHub deposuna yüklenir.
- Sonraki notebook girdiyi GitHub RAW bağlantısından okur.
- GitHub dosya konumu tek bir sabit yola bağlı değildir; notebooklar repo ağacını tarayarak uygun CSV dosyasını bulmaya çalışır.
- Mevcut GitHub repo yapısındaki aşağıdaki feature store yolu da desteklenir:

```text
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

Ortak Google Drive çıktı klasörü:

```text
/content/drive/MyDrive/MOL_FET_regression_workshop/
```

## Önerilen GitHub klasör yapısı

```text
MOL-FET_regression/
├── notebooks/
│   ├── 00_chembl_target_data_store.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_generation.ipynb
│   ├── 03_lazypredict.ipynb
│   └── 04_SHAP_interpretability.ipynb
│
├── data/
│   ├── CHEMBL206_IC50_single_protein_format_CLEAN_parent_dedup.csv
│   ├── CHEMBL206_IC50_parent_dedup_Lipinski_filtered.csv
│   └── CHEMBL206_Mordred2D_model_ready.csv
│
├── molfet_regression_outputs/
│   └── 01_mordred_feature_store/
│       └── mordred_2d_features.csv
│
└── README.md
```

`TARGET_ID` değiştirildiğinde dosya adlarındaki `CHEMBL206` bölümü yeni target kimliğiyle oluşturulur.

---

# Notebook 00 — ChEMBL Target Data Store

Dosya:

```text
00_chembl_target_data_store.ipynb
```

## Amaç

Bir ChEMBL Target ID alır ve hedefe ait regression verisini ChEMBL REST API üzerinden oluşturur.

## Temel işlemler

- ChEMBL target metadata kontrolü
- IC50 aktivitelerinin sayfalı API çağrılarıyla indirilmesi
- Single protein format assay seçimi
- Kesin `=` ilişkili ölçümlerin tutulması
- IC50 değerlerinden `pStandard` hesaplanması
- Ligand SMILES bilgilerinin alınması
- Parent ChEMBL ID ve parent SMILES oluşturulması
- Tuzlardan arındırılmış canonical parent yapı üretimi
- Parent yapı bazında duplicate aggregation
- Medyan, ortalama, standart sapma ve ölçüm sayılarının kaydedilmesi

## Google Drive çıktıları

```text
<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
<TARGET_ID>_IC50_target_smiles.csv
<TARGET_ID>_target_metadata.csv
```

## GitHub'a yüklenecek ana çıktı

```text
data/<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
```

Bu dosya Notebook 01 tarafından GitHub üzerinden okunur.

---

# Notebook 01 — Data Cleaning

Dosya:

```text
01_data_cleaning.ipynb
```

## Amaç

Notebook 00 tarafından oluşturulan ana regression CSV dosyasını GitHub deposundan bulur, kimyasal temizleme işlemlerini uygular ve Lipinski filtreli veri setini oluşturur.

## GitHub arama davranışı

Notebook aşağıdaki yolları ve repo içindeki benzer dosya adlarını kontrol eder:

```text
data/<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
training_data.csv
data/training_data.csv
```

Bir URL 404 döndürürse notebook ilk hatada durmaz; diğer aday yolları denemeye devam eder.

## Temel işlemler

- Target ve SMILES sütun kontrolü
- Eksik target ve SMILES kayıtlarının ayrılması
- RDKit ile SMILES doğrulama
- Tuz ve karşı iyonların çıkarılması
- Canonical parent SMILES üretimi
- Aynı canonical yapıya ait duplicate kayıtların medyan target ile birleştirilmesi
- Ölçüm aralığı ve aktivite çatışma bayrağı
- Molekül ağırlığı
- MolLogP
- H-bağı verici sayısı
- H-bağı alıcı sayısı
- Strict Lipinski Rule-of-Five filtresi

## Google Drive çıktıları

```text
<TARGET_ID>_IC50_parent_dedup_Lipinski_filtered.csv
<TARGET_ID>_IC50_parent_dedup_with_Lipinski.csv
<TARGET_ID>_invalid_smiles.csv
<TARGET_ID>_Lipinski_rejected.csv
<TARGET_ID>_cleaning_report.csv
```

## GitHub'a yüklenecek ana çıktı

```text
data/<TARGET_ID>_IC50_parent_dedup_Lipinski_filtered.csv
```

Bu dosya Notebook 02 tarafından GitHub üzerinden okunur.

---

# Notebook 02 — 2D Mordred Feature Generation

Dosya:

```text
02_feature_generation.ipynb
```

## Amaç

Lipinski filtreli regression CSV dosyasını GitHub deposundan okur ve yalnızca **2D Mordred descriptorlarından** oluşan model-ready feature store üretir.

## Temel işlemler

- GitHub repo ağacında uygun regression CSV arama
- Target, SMILES ve molekül ID sütunlarının otomatik seçimi
- Canonical SMILES kontrolü
- Duplicate yapıların birleştirilmesi
- Target dağılımının raporlanması
- RDKit molekül nesnelerinin oluşturulması
- `ignore_3D=True` ile Mordred 2D descriptor hesabı
- Sayısal olmayan Mordred hata nesnelerinin `NaN` yapılması
- Sonsuz değerlerin `NaN` yapılması
- Yüksek eksik oranlı descriptorların çıkarılması
- Kalan eksik değerlerin medyanla doldurulması
- Sabit descriptorların çıkarılması
- Mutlak Pearson korelasyonu yüksek descriptorların çıkarılması
- Model-ready feature store ve feature manifest oluşturulması

## Varsayılan feature filtreleri

```text
Maksimum eksik değer oranı: 0.20
Korelasyon eşiği: 0.95
3D descriptor: kullanılmaz
```

## Google Drive çıktıları

```text
<TARGET_ID>_Mordred2D_model_ready.csv
<TARGET_ID>_Mordred2D_raw_numeric.csv
<TARGET_ID>_Mordred2D_manifest.json
<TARGET_ID>_Mordred2D_QC_summary.csv
<TARGET_ID>_Mordred2D_invalid_rows.csv
<TARGET_ID>_target_distribution.png
```

## GitHub'a yüklenecek ana çıktı

Tercih edilen target-özel dosya:

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
```

Mevcut repo yapısında desteklenen alternatif feature store:

```text
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

Notebook 03 ve Notebook 04, bu iki yapıdan uygun olanı otomatik bulmaya çalışır.

---

# Notebook 03 — LazyPredict Regression

Dosya:

```text
03_lazypredict.ipynb
```

## Amaç

GitHub reposundaki 2D Mordred feature store dosyasını okur, çok sayıda regression model ailesini LazyPredict ile karşılaştırır ve SHAP uyumlu bir final ağaç modeli oluşturur.

LazyPredict burada otomatik model seçiminin son kararı olarak değil, farklı model ailelerinin aynı veri üzerindeki davranışını ders içinde karşılaştırmak için kullanılır.

## Desteklenen GitHub feature store yapıları

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
<TARGET_ID>_Mordred2D_model_ready.csv
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

## Eksik değer yönetimi

- `NaN`, `+inf` ve `-inf` değerleri güvenli biçimde işlenir.
- Tamamen boş descriptorlar çıkarılır.
- Eksik oranı `%20` üzerinde olan descriptorlar çıkarılır.
- Sabit descriptorlar çıkarılır.
- Median imputer yalnızca `X_train` üzerinde fit edilir.
- Aynı imputer `X_test` üzerinde yalnızca `transform` için kullanılır.
- Böylece test bilgisinin eğitim ön işleme aşamasına sızması engellenir.

## Modelleme akışı

- Tekrar üretilebilir `%80 / %20` train-test split
- LazyPredict regression benchmark
- LazyPredict cache kontrolü
- Geçerli cache varsa pahalı benchmarkın yeniden çalıştırılmaması
- SHAP uyumlu final ağaç modeli seçimi
- Desteklenen final model aileleri:
  - ExtraTreesRegressor
  - RandomForestRegressor
  - GradientBoostingRegressor
- Train ve test metrikleri
- Eğitim setinde 5-fold cross-validation
- Gerçek ve tahmin edilen target grafiği

## Regression metrikleri

```text
R²   = açıklanan varyans oranı
RMSE = büyük hataları daha fazla cezalandıran hata metriği
MAE  = ortalama mutlak hata
```

## Google Drive çıktıları

```text
<TARGET_ID>_LazyPredict_results.csv
<TARGET_ID>_LazyPredict_test_predictions.csv
<TARGET_ID>_LazyPredict_cache_metadata.json
<TARGET_ID>_feature_imputer.joblib
<TARGET_ID>_final_tree_model.joblib
<TARGET_ID>_final_model_bundle.json
<TARGET_ID>_train_predictions.csv
<TARGET_ID>_test_predictions.csv
<TARGET_ID>_final_model_metrics.csv
<TARGET_ID>_final_model_5CV.csv
<TARGET_ID>_actual_vs_predicted.png
```

## Model bundle

`<TARGET_ID>_final_model_bundle.json` dosyası aşağıdaki bilgileri taşır:

- Feature isimleri ve sırası
- Target sütunu
- Train ve test indeksleri
- Random seed
- Test oranı
- Final model adı ve parametreleri
- Train ve test metrikleri
- İmputer bilgisi
- Çıkarılan eksik ve sabit featurelar

Bu bilgiler Notebook 04 tarafından kullanılabilir.

---

# Notebook 04 — Application Domain ve SHAP Interpretability

Dosya:

```text
04_SHAP_interpretability.ipynb
```

## Amaç

GitHub feature store dosyasını kullanarak final modelin applicability domain ve yorumlanabilirlik analizlerini üretir.

## Girdi çözümleme

Feature store şu kaynaklardan otomatik aranır:

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
<TARGET_ID>_Mordred2D_model_ready.csv
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

Model bundle ve joblib model:

1. GitHub deposunda aranır.
2. Bulunamazsa Google Drive kontrol edilir.
3. Uyumlu bir model bulunamazsa aynı temizlenmiş Mordred feature seti üzerinde tekrar üretilebilir bir ağaç modeli yalnızca bir kez eğitilir.

## Feature ön işleme

- Target satırlarının doğrulanması
- Tamamen boş descriptorların çıkarılması
- Yüksek eksik oranlı descriptorların çıkarılması
- Sabit descriptorların çıkarılması
- Median imputerın yalnızca eğitim setinde fit edilmesi
- Aynı dönüşümün test setine uygulanması

## Application Domain

Application domain analizi PCA-leverage yaklaşımıyla yapılır.

Temel eşik:

```text
h* = 3(k + 1) / n_train
```

Burada:

```text
k       = kullanılan PCA bileşen sayısı
n_train = eğitim molekülü sayısı
```

AD-in koşulu:

```text
leverage ≤ h*
ve
|standartlaştırılmış artık| ≤ 3
```

Üretilen başlıca AD çıktıları:

```text
<TARGET_ID>_train_application_domain.csv
<TARGET_ID>_test_application_domain.csv
<TARGET_ID>_Williams_plot.png
```

## SHAP analizleri

Notebook aşağıdaki yorumlanabilirlik çıktılarını üretir:

- Ortalama mutlak SHAP önem tablosu
- SHAP beeswarm
- SHAP bar grafiği
- En yüksek test hatasına sahip molekül için lokal waterfall
- En önemli descriptor için SHAP dependence grafiği

Üretilen SHAP çıktıları:

```text
<TARGET_ID>_SHAP_feature_importance.csv
<TARGET_ID>_SHAP_beeswarm.png
<TARGET_ID>_SHAP_bar.png
<TARGET_ID>_SHAP_local_waterfall.png
<TARGET_ID>_SHAP_top_feature_dependence.png
```

Ek çıktılar:

```text
<TARGET_ID>_final_tree_model.joblib
<TARGET_ID>_feature_imputer.joblib
<TARGET_ID>_final_model_bundle.json
<TARGET_ID>_AD_SHAP_summary.json
```

---

# Colab kullanım sırası

## 1. Notebook 00

```text
00_chembl_target_data_store.ipynb
```

`TARGET_ID` değerini ayarlayın ve bütün hücreleri yukarıdan aşağıya çalıştırın.

Ana çıktıyı GitHub'a yükleyin:

```text
data/<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
```

## 2. Notebook 01

```text
01_data_cleaning.ipynb
```

Notebook ana CSV'yi GitHub'dan bulup temizler.

Ana çıktıyı GitHub'a yükleyin:

```text
data/<TARGET_ID>_IC50_parent_dedup_Lipinski_filtered.csv
```

## 3. Notebook 02

```text
02_feature_generation.ipynb
```

Notebook Lipinski filtreli veriyi GitHub'dan okur ve 2D Mordred feature store üretir.

Ana çıktıyı GitHub'a yükleyin:

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
```

## 4. Notebook 03

```text
03_lazypredict.ipynb
```

Notebook Mordred feature store dosyasını GitHub'dan okur, LazyPredict benchmark ve final modelleme yapar.

Model ve raporlar Google Drive'a kaydedilir.

## 5. Notebook 04

```text
04_SHAP_interpretability.ipynb
```

Notebook feature store, model bundle ve modeli GitHub veya Drive kaynaklarından çözer; applicability domain ve SHAP analizlerini tamamlar.

---

# Paketler

Notebooklar gerekli paketleri Colab içinde kontrol eder ve eksik olanları kurar.

Başlıca bağımlılıklar:

```text
numpy
pandas
requests
tqdm
rdkit
mordredcommunity[full]==2.0.7
scikit-learn
lazypredict==0.3.0
shap
matplotlib
joblib
```

Eski `rdkit-pypi` paket adı kullanılmaz. Güncel paket adı:

```text
rdkit
```

---

# Bilimsel kapsam

Bu workshop sürümünde:

- Model feature olarak yalnızca 2D Mordred descriptorları kullanılır.
- Fingerprint feature kullanılmaz.
- Ayrı bir Tanimoto similarity notebooku bulunmaz.
- Ayrı bir RF baseline notebooku bulunmaz.
- LazyPredict model ailelerini karşılaştırmak için kullanılır.
- Final model yorumlanabilirlik ve application-domain analizine uygun bir ağaç modelidir.
- Eksik değer imputasyonu yalnızca eğitim setinde fit edilir.
- PCA, scaler ve leverage referans uzayı yalnızca eğitim setinden oluşturulur.
- Test seti model seçimi veya ön işleme fit aşamasında kullanılmaz.

---

# Ders planı önerisi

```text
Ders 1:
00 — ChEMBL Target ID ve ligand verisinin oluşturulması

Ders 2:
01 — SMILES temizliği, tuz çıkarma, duplicate ve Lipinski

Ders 3:
02 — 2D Mordred feature üretimi ve feature kalite kontrolü

Ders 4:
03 — LazyPredict, final model, holdout ve 5-fold CV

Ders 5:
04 — Applicability domain, Williams plot ve SHAP yorumlanabilirlik
```

---

# GitHub yükleme örneği

```bash
git clone https://github.com/MOL-FEST/MOL-FET_regression.git
cd MOL-FET_regression
```

Güncellenen notebook ve CSV dosyalarını ilgili klasörlere kopyaladıktan sonra:

```bash
git add .
git commit -m "Update MOL-FET regression workshop notebooks"
git push origin main
```

---

# Önemli notlar

- Notebookları yukarıdan aşağıya sırasıyla çalıştırın.
- `TARGET_ID` değerinin bütün notebooklarda aynı olduğundan emin olun.
- Bir notebookun ana CSV çıktısını GitHub'a yüklemeden sonraki notebooku çalıştırmayın.
- Google Drive çıktı klasörünü değiştirecekseniz bütün notebooklarda aynı yolu kullanın.
- GitHub'daki dosya adı değişirse notebookun repo ağacı arama kurallarına uygun olduğundan emin olun.
- Feature store içindeki target, SMILES ve Mordred descriptor sütunlarının doğru seçildiğini checkpoint çıktılarından kontrol edin.
- LazyPredict cache yalnızca veri, feature listesi, split ve ayarlar tamamen aynıysa yeniden kullanılır.
