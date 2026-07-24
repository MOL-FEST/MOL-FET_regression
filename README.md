# MOL-FET Regression Workshop

Bu depo, bir **ChEMBL Target ID** Ã¼zerinden ligand verisi toplamadan baÅŸlayÄ±p 2D Mordred descriptor Ã¼retimi, LazyPredict regresyon karÅŸÄ±laÅŸtÄ±rmasÄ±, final aÄŸaÃ§ modeli, applicability domain ve SHAP yorumlanabilirlik analizine kadar ilerleyen beÅŸ notebookluk bir workshop akÄ±ÅŸÄ± iÃ§erir.

Bu gÃ¼ncel sÃ¼rÃ¼mde model girdisi olarak yalnÄ±zca **2D Mordred descriptorlarÄ±** kullanÄ±lÄ±r. Morgan, MACCS, Avalon veya baÅŸka fingerprint tÃ¼rleri model feature olarak kullanÄ±lmaz.

## GÃ¼ncel notebook sÄ±rasÄ±

```text
00_chembl_target_data_store.ipynb
01_data_cleaning.ipynb
02_feature_generation.ipynb
03_lazypredict.ipynb
04_SHAP_interpretability.ipynb
```

## Genel veri akÄ±ÅŸÄ±

```text
ChEMBL Target ID
        â†“
IC50 ve single-protein assay verisi
        â†“
Parent SMILES ve parent-dedup regression CSV
        â†“
Tuz / karÅŸÄ± iyon temizliÄŸi
        â†“
Canonical SMILES ve duplicate birleÅŸtirme
        â†“
Lipinski filtresi
        â†“
2D Mordred feature store
        â†“
LazyPredict regression benchmark
        â†“
Final aÄŸaÃ§ modeli ve 5-fold CV
        â†“
PCA-leverage applicability domain
        â†“
Williams plot ve SHAP yorumlanabilirlik
```

## Depo

```text
https://github.com/MOL-FEST/MOL-FET_regression
```

## Ã‡alÄ±ÅŸma mantÄ±ÄŸÄ±

Notebooklar Google Colab iÃ§in hazÄ±rlanmÄ±ÅŸtÄ±r.

- Ãœretilen dosyalar Google Drive iÃ§indeki ortak workshop klasÃ¶rÃ¼ne kaydedilir.
- Sonraki notebookun ihtiyaÃ§ duyduÄŸu ana CSV, ilgili notebook Ã§alÄ±ÅŸtÄ±rÄ±ldÄ±ktan sonra GitHub deposuna yÃ¼klenir.
- Sonraki notebook girdiyi GitHub RAW baÄŸlantÄ±sÄ±ndan okur.
- GitHub dosya konumu tek bir sabit yola baÄŸlÄ± deÄŸildir; notebooklar repo aÄŸacÄ±nÄ± tarayarak uygun CSV dosyasÄ±nÄ± bulmaya Ã§alÄ±ÅŸÄ±r.
- Mevcut GitHub repo yapÄ±sÄ±ndaki aÅŸaÄŸÄ±daki feature store yolu da desteklenir:

```text
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

Ortak Google Drive Ã§Ä±ktÄ± klasÃ¶rÃ¼:

```text
/content/drive/MyDrive/MOL_FET_regression_workshop/
```

## Ã–nerilen GitHub klasÃ¶r yapÄ±sÄ±

```text
MOL-FET_regression/
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 00_chembl_target_data_store.ipynb
â”‚   â”œâ”€â”€ 01_data_cleaning.ipynb
â”‚   â”œâ”€â”€ 02_feature_generation.ipynb
â”‚   â”œâ”€â”€ 03_lazypredict.ipynb
â”‚   â””â”€â”€ 04_SHAP_interpretability.ipynb
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ CHEMBL206_IC50_single_protein_format_CLEAN_parent_dedup.csv
â”‚   â”œâ”€â”€ CHEMBL206_IC50_parent_dedup_Lipinski_filtered.csv
â”‚   â””â”€â”€ CHEMBL206_Mordred2D_model_ready.csv
â”‚
â”œâ”€â”€ molfet_regression_outputs/
â”‚   â””â”€â”€ 01_mordred_feature_store/
â”‚       â””â”€â”€ mordred_2d_features.csv
â”‚
â””â”€â”€ README.md
```

`TARGET_ID` deÄŸiÅŸtirildiÄŸinde dosya adlarÄ±ndaki `CHEMBL206` bÃ¶lÃ¼mÃ¼ yeni target kimliÄŸiyle oluÅŸturulur.

---

# Notebook 00 â€” ChEMBL Target Data Store

Dosya:

```text
00_chembl_target_data_store.ipynb
```

## AmaÃ§

Bir ChEMBL Target ID alÄ±r ve hedefe ait regression verisini ChEMBL REST API Ã¼zerinden oluÅŸturur.

## Temel iÅŸlemler

- ChEMBL target metadata kontrolÃ¼
- IC50 aktivitelerinin sayfalÄ± API Ã§aÄŸrÄ±larÄ±yla indirilmesi
- Single protein format assay seÃ§imi
- Kesin `=` iliÅŸkili Ã¶lÃ§Ã¼mlerin tutulmasÄ±
- IC50 deÄŸerlerinden `pStandard` hesaplanmasÄ±
- Ligand SMILES bilgilerinin alÄ±nmasÄ±
- Parent ChEMBL ID ve parent SMILES oluÅŸturulmasÄ±
- Tuzlardan arÄ±ndÄ±rÄ±lmÄ±ÅŸ canonical parent yapÄ± Ã¼retimi
- Parent yapÄ± bazÄ±nda duplicate aggregation
- Medyan, ortalama, standart sapma ve Ã¶lÃ§Ã¼m sayÄ±larÄ±nÄ±n kaydedilmesi

## Google Drive Ã§Ä±ktÄ±larÄ±

```text
<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
<TARGET_ID>_IC50_target_smiles.csv
<TARGET_ID>_target_metadata.csv
```

## GitHub'a yÃ¼klenecek ana Ã§Ä±ktÄ±

```text
data/<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
```

Bu dosya Notebook 01 tarafÄ±ndan GitHub Ã¼zerinden okunur.

---

# Notebook 01 â€” Data Cleaning

Dosya:

```text
01_data_cleaning.ipynb
```

## AmaÃ§

Notebook 00 tarafÄ±ndan oluÅŸturulan ana regression CSV dosyasÄ±nÄ± GitHub deposundan bulur, kimyasal temizleme iÅŸlemlerini uygular ve Lipinski filtreli veri setini oluÅŸturur.

## GitHub arama davranÄ±ÅŸÄ±

Notebook aÅŸaÄŸÄ±daki yollarÄ± ve repo iÃ§indeki benzer dosya adlarÄ±nÄ± kontrol eder:

```text
data/<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
training_data.csv
data/training_data.csv
```

Bir URL 404 dÃ¶ndÃ¼rÃ¼rse notebook ilk hatada durmaz; diÄŸer aday yollarÄ± denemeye devam eder.

## Temel iÅŸlemler

- Target ve SMILES sÃ¼tun kontrolÃ¼
- Eksik target ve SMILES kayÄ±tlarÄ±nÄ±n ayrÄ±lmasÄ±
- RDKit ile SMILES doÄŸrulama
- Tuz ve karÅŸÄ± iyonlarÄ±n Ã§Ä±karÄ±lmasÄ±
- Canonical parent SMILES Ã¼retimi
- AynÄ± canonical yapÄ±ya ait duplicate kayÄ±tlarÄ±n medyan target ile birleÅŸtirilmesi
- Ã–lÃ§Ã¼m aralÄ±ÄŸÄ± ve aktivite Ã§atÄ±ÅŸma bayraÄŸÄ±
- MolekÃ¼l aÄŸÄ±rlÄ±ÄŸÄ±
- MolLogP
- H-baÄŸÄ± verici sayÄ±sÄ±
- H-baÄŸÄ± alÄ±cÄ± sayÄ±sÄ±
- Strict Lipinski Rule-of-Five filtresi

## Google Drive Ã§Ä±ktÄ±larÄ±

```text
<TARGET_ID>_IC50_parent_dedup_Lipinski_filtered.csv
<TARGET_ID>_IC50_parent_dedup_with_Lipinski.csv
<TARGET_ID>_invalid_smiles.csv
<TARGET_ID>_Lipinski_rejected.csv
<TARGET_ID>_cleaning_report.csv
```

## GitHub'a yÃ¼klenecek ana Ã§Ä±ktÄ±

```text
data/<TARGET_ID>_IC50_parent_dedup_Lipinski_filtered.csv
```

Bu dosya Notebook 02 tarafÄ±ndan GitHub Ã¼zerinden okunur.

---

# Notebook 02 â€” 2D Mordred Feature Generation

Dosya:

```text
02_feature_generation.ipynb
```

## AmaÃ§

Lipinski filtreli regression CSV dosyasÄ±nÄ± GitHub deposundan okur ve yalnÄ±zca **2D Mordred descriptorlarÄ±ndan** oluÅŸan model-ready feature store Ã¼retir.

## Temel iÅŸlemler

- GitHub repo aÄŸacÄ±nda uygun regression CSV arama
- Target, SMILES ve molekÃ¼l ID sÃ¼tunlarÄ±nÄ±n otomatik seÃ§imi
- Canonical SMILES kontrolÃ¼
- Duplicate yapÄ±larÄ±n birleÅŸtirilmesi
- Target daÄŸÄ±lÄ±mÄ±nÄ±n raporlanmasÄ±
- RDKit molekÃ¼l nesnelerinin oluÅŸturulmasÄ±
- `ignore_3D=True` ile Mordred 2D descriptor hesabÄ±
- SayÄ±sal olmayan Mordred hata nesnelerinin `NaN` yapÄ±lmasÄ±
- Sonsuz deÄŸerlerin `NaN` yapÄ±lmasÄ±
- YÃ¼ksek eksik oranlÄ± descriptorlarÄ±n Ã§Ä±karÄ±lmasÄ±
- Kalan eksik deÄŸerlerin medyanla doldurulmasÄ±
- Sabit descriptorlarÄ±n Ã§Ä±karÄ±lmasÄ±
- Mutlak Pearson korelasyonu yÃ¼ksek descriptorlarÄ±n Ã§Ä±karÄ±lmasÄ±
- Model-ready feature store ve feature manifest oluÅŸturulmasÄ±

## VarsayÄ±lan feature filtreleri

```text
Maksimum eksik deÄŸer oranÄ±: 0.20
Korelasyon eÅŸiÄŸi: 0.95
3D descriptor: kullanÄ±lmaz
```

## Google Drive Ã§Ä±ktÄ±larÄ±

```text
<TARGET_ID>_Mordred2D_model_ready.csv
<TARGET_ID>_Mordred2D_raw_numeric.csv
<TARGET_ID>_Mordred2D_manifest.json
<TARGET_ID>_Mordred2D_QC_summary.csv
<TARGET_ID>_Mordred2D_invalid_rows.csv
<TARGET_ID>_target_distribution.png
```

## GitHub'a yÃ¼klenecek ana Ã§Ä±ktÄ±

Tercih edilen target-Ã¶zel dosya:

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
```

Mevcut repo yapÄ±sÄ±nda desteklenen alternatif feature store:

```text
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

Notebook 03 ve Notebook 04, bu iki yapÄ±dan uygun olanÄ± otomatik bulmaya Ã§alÄ±ÅŸÄ±r.

---

# Notebook 03 â€” LazyPredict Regression

Dosya:

```text
03_lazypredict.ipynb
```

## AmaÃ§

GitHub reposundaki 2D Mordred feature store dosyasÄ±nÄ± okur, Ã§ok sayÄ±da regression model ailesini LazyPredict ile karÅŸÄ±laÅŸtÄ±rÄ±r ve SHAP uyumlu bir final aÄŸaÃ§ modeli oluÅŸturur.

LazyPredict burada otomatik model seÃ§iminin son kararÄ± olarak deÄŸil, farklÄ± model ailelerinin aynÄ± veri Ã¼zerindeki davranÄ±ÅŸÄ±nÄ± ders iÃ§inde karÅŸÄ±laÅŸtÄ±rmak iÃ§in kullanÄ±lÄ±r.

## Desteklenen GitHub feature store yapÄ±larÄ±

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
<TARGET_ID>_Mordred2D_model_ready.csv
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

## Eksik deÄŸer yÃ¶netimi

- `NaN`, `+inf` ve `-inf` deÄŸerleri gÃ¼venli biÃ§imde iÅŸlenir.
- Tamamen boÅŸ descriptorlar Ã§Ä±karÄ±lÄ±r.
- Eksik oranÄ± `%20` Ã¼zerinde olan descriptorlar Ã§Ä±karÄ±lÄ±r.
- Sabit descriptorlar Ã§Ä±karÄ±lÄ±r.
- Median imputer yalnÄ±zca `X_train` Ã¼zerinde fit edilir.
- AynÄ± imputer `X_test` Ã¼zerinde yalnÄ±zca `transform` iÃ§in kullanÄ±lÄ±r.
- BÃ¶ylece test bilgisinin eÄŸitim Ã¶n iÅŸleme aÅŸamasÄ±na sÄ±zmasÄ± engellenir.

## Modelleme akÄ±ÅŸÄ±

- Tekrar Ã¼retilebilir `%80 / %20` train-test split
- LazyPredict regression benchmark
- LazyPredict cache kontrolÃ¼
- GeÃ§erli cache varsa pahalÄ± benchmarkÄ±n yeniden Ã§alÄ±ÅŸtÄ±rÄ±lmamasÄ±
- SHAP uyumlu final aÄŸaÃ§ modeli seÃ§imi
- Desteklenen final model aileleri:
  - ExtraTreesRegressor
  - RandomForestRegressor
  - GradientBoostingRegressor
- Train ve test metrikleri
- EÄŸitim setinde 5-fold cross-validation
- GerÃ§ek ve tahmin edilen target grafiÄŸi

## Regression metrikleri

```text
RÂ²   = aÃ§Ä±klanan varyans oranÄ±
RMSE = bÃ¼yÃ¼k hatalarÄ± daha fazla cezalandÄ±ran hata metriÄŸi
MAE  = ortalama mutlak hata
```

## Google Drive Ã§Ä±ktÄ±larÄ±

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

`<TARGET_ID>_final_model_bundle.json` dosyasÄ± aÅŸaÄŸÄ±daki bilgileri taÅŸÄ±r:

- Feature isimleri ve sÄ±rasÄ±
- Target sÃ¼tunu
- Train ve test indeksleri
- Random seed
- Test oranÄ±
- Final model adÄ± ve parametreleri
- Train ve test metrikleri
- Ä°mputer bilgisi
- Ã‡Ä±karÄ±lan eksik ve sabit featurelar

Bu bilgiler Notebook 04 tarafÄ±ndan kullanÄ±labilir.

---

# Notebook 04 â€” Application Domain ve SHAP Interpretability

Dosya:

```text
04_SHAP_interpretability.ipynb
```

## AmaÃ§

GitHub feature store dosyasÄ±nÄ± kullanarak final modelin applicability domain ve yorumlanabilirlik analizlerini Ã¼retir.

## Girdi Ã§Ã¶zÃ¼mleme

Feature store ÅŸu kaynaklardan otomatik aranÄ±r:

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
<TARGET_ID>_Mordred2D_model_ready.csv
molfet_regression_outputs/01_mordred_feature_store/mordred_2d_features.csv
```

Model bundle ve joblib model:

1. GitHub deposunda aranÄ±r.
2. Bulunamazsa Google Drive kontrol edilir.
3. Uyumlu bir model bulunamazsa aynÄ± temizlenmiÅŸ Mordred feature seti Ã¼zerinde tekrar Ã¼retilebilir bir aÄŸaÃ§ modeli yalnÄ±zca bir kez eÄŸitilir.

## Feature Ã¶n iÅŸleme

- Target satÄ±rlarÄ±nÄ±n doÄŸrulanmasÄ±
- Tamamen boÅŸ descriptorlarÄ±n Ã§Ä±karÄ±lmasÄ±
- YÃ¼ksek eksik oranlÄ± descriptorlarÄ±n Ã§Ä±karÄ±lmasÄ±
- Sabit descriptorlarÄ±n Ã§Ä±karÄ±lmasÄ±
- Median imputerÄ±n yalnÄ±zca eÄŸitim setinde fit edilmesi
- AynÄ± dÃ¶nÃ¼ÅŸÃ¼mÃ¼n test setine uygulanmasÄ±

## Application Domain

Application domain analizi PCA-leverage yaklaÅŸÄ±mÄ±yla yapÄ±lÄ±r.

Temel eÅŸik:

```text
h* = 3(k + 1) / n_train
```

Burada:

```text
k       = kullanÄ±lan PCA bileÅŸen sayÄ±sÄ±
n_train = eÄŸitim molekÃ¼lÃ¼ sayÄ±sÄ±
```

AD-in koÅŸulu:

```text
leverage â‰¤ h*
ve
|standartlaÅŸtÄ±rÄ±lmÄ±ÅŸ artÄ±k| â‰¤ 3
```

Ãœretilen baÅŸlÄ±ca AD Ã§Ä±ktÄ±larÄ±:

```text
<TARGET_ID>_train_application_domain.csv
<TARGET_ID>_test_application_domain.csv
<TARGET_ID>_Williams_plot.png
```

## SHAP analizleri

Notebook aÅŸaÄŸÄ±daki yorumlanabilirlik Ã§Ä±ktÄ±larÄ±nÄ± Ã¼retir:

- Ortalama mutlak SHAP Ã¶nem tablosu
- SHAP beeswarm
- SHAP bar grafiÄŸi
- En yÃ¼ksek test hatasÄ±na sahip molekÃ¼l iÃ§in lokal waterfall
- En Ã¶nemli descriptor iÃ§in SHAP dependence grafiÄŸi

Ãœretilen SHAP Ã§Ä±ktÄ±larÄ±:

```text
<TARGET_ID>_SHAP_feature_importance.csv
<TARGET_ID>_SHAP_beeswarm.png
<TARGET_ID>_SHAP_bar.png
<TARGET_ID>_SHAP_local_waterfall.png
<TARGET_ID>_SHAP_top_feature_dependence.png
```

Ek Ã§Ä±ktÄ±lar:

```text
<TARGET_ID>_final_tree_model.joblib
<TARGET_ID>_feature_imputer.joblib
<TARGET_ID>_final_model_bundle.json
<TARGET_ID>_AD_SHAP_summary.json
```

---

# Colab kullanÄ±m sÄ±rasÄ±

## 1. Notebook 00

```text
00_chembl_target_data_store.ipynb
```

`TARGET_ID` deÄŸerini ayarlayÄ±n ve bÃ¼tÃ¼n hÃ¼creleri yukarÄ±dan aÅŸaÄŸÄ±ya Ã§alÄ±ÅŸtÄ±rÄ±n.

Ana Ã§Ä±ktÄ±yÄ± GitHub'a yÃ¼kleyin:

```text
data/<TARGET_ID>_IC50_single_protein_format_CLEAN_parent_dedup.csv
```

## 2. Notebook 01

```text
01_data_cleaning.ipynb
```

Notebook ana CSV'yi GitHub'dan bulup temizler.

Ana Ã§Ä±ktÄ±yÄ± GitHub'a yÃ¼kleyin:

```text
data/<TARGET_ID>_IC50_parent_dedup_Lipinski_filtered.csv
```

## 3. Notebook 02

```text
02_feature_generation.ipynb
```

Notebook Lipinski filtreli veriyi GitHub'dan okur ve 2D Mordred feature store Ã¼retir.

Ana Ã§Ä±ktÄ±yÄ± GitHub'a yÃ¼kleyin:

```text
data/<TARGET_ID>_Mordred2D_model_ready.csv
```

## 4. Notebook 03

```text
03_lazypredict.ipynb
```

Notebook Mordred feature store dosyasÄ±nÄ± GitHub'dan okur, LazyPredict benchmark ve final modelleme yapar.

Model ve raporlar Google Drive'a kaydedilir.

## 5. Notebook 04

```text
04_SHAP_interpretability.ipynb
```

Notebook feature store, model bundle ve modeli GitHub veya Drive kaynaklarÄ±ndan Ã§Ã¶zer; applicability domain ve SHAP analizlerini tamamlar.

---

# Paketler

Notebooklar gerekli paketleri Colab iÃ§inde kontrol eder ve eksik olanlarÄ± kurar.

BaÅŸlÄ±ca baÄŸÄ±mlÄ±lÄ±klar:

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

Eski `rdkit-pypi` paket adÄ± kullanÄ±lmaz. GÃ¼ncel paket adÄ±:

```text
rdkit
```

---

# Bilimsel kapsam

Bu workshop sÃ¼rÃ¼mÃ¼nde:

- Model feature olarak yalnÄ±zca 2D Mordred descriptorlarÄ± kullanÄ±lÄ±r.
- Fingerprint feature kullanÄ±lmaz.
- AyrÄ± bir Tanimoto similarity notebooku bulunmaz.
- AyrÄ± bir RF baseline notebooku bulunmaz.
- LazyPredict model ailelerini karÅŸÄ±laÅŸtÄ±rmak iÃ§in kullanÄ±lÄ±r.
- Final model yorumlanabilirlik ve application-domain analizine uygun bir aÄŸaÃ§ modelidir.
- Eksik deÄŸer imputasyonu yalnÄ±zca eÄŸitim setinde fit edilir.
- PCA, scaler ve leverage referans uzayÄ± yalnÄ±zca eÄŸitim setinden oluÅŸturulur.
- Test seti model seÃ§imi veya Ã¶n iÅŸleme fit aÅŸamasÄ±nda kullanÄ±lmaz.

---

# Ders planÄ± Ã¶nerisi

```text
Ders 1:
00 â€” ChEMBL Target ID ve ligand verisinin oluÅŸturulmasÄ±

Ders 2:
01 â€” SMILES temizliÄŸi, tuz Ã§Ä±karma, duplicate ve Lipinski

Ders 3:
02 â€” 2D Mordred feature Ã¼retimi ve feature kalite kontrolÃ¼

Ders 4:
03 â€” LazyPredict, final model, holdout ve 5-fold CV

Ders 5:
04 â€” Applicability domain, Williams plot ve SHAP yorumlanabilirlik
```

---

# GitHub yÃ¼kleme Ã¶rneÄŸi

```bash
git clone https://github.com/MOL-FEST/MOL-FET_regression.git
cd MOL-FET_regression
```

GÃ¼ncellenen notebook ve CSV dosyalarÄ±nÄ± ilgili klasÃ¶rlere kopyaladÄ±ktan sonra:

```bash
git add .
git commit -m "Update MOL-FET regression workshop notebooks"
git push origin main
```

---

# Ã–nemli notlar

- NotebooklarÄ± yukarÄ±dan aÅŸaÄŸÄ±ya sÄ±rasÄ±yla Ã§alÄ±ÅŸtÄ±rÄ±n.
- `TARGET_ID` deÄŸerinin bÃ¼tÃ¼n notebooklarda aynÄ± olduÄŸundan emin olun.
- Bir notebookun ana CSV Ã§Ä±ktÄ±sÄ±nÄ± GitHub'a yÃ¼klemeden sonraki notebooku Ã§alÄ±ÅŸtÄ±rmayÄ±n.
- Google Drive Ã§Ä±ktÄ± klasÃ¶rÃ¼nÃ¼ deÄŸiÅŸtirecekseniz bÃ¼tÃ¼n notebooklarda aynÄ± yolu kullanÄ±n.
- GitHub'daki dosya adÄ± deÄŸiÅŸirse notebookun repo aÄŸacÄ± arama kurallarÄ±na uygun olduÄŸundan emin olun.
- Feature store iÃ§indeki target, SMILES ve Mordred descriptor sÃ¼tunlarÄ±nÄ±n doÄŸru seÃ§ildiÄŸini checkpoint Ã§Ä±ktÄ±larÄ±ndan kontrol edin.
- LazyPredict cache yalnÄ±zca veri, feature listesi, split ve ayarlar tamamen aynÄ±ysa yeniden kullanÄ±lÄ±r.
