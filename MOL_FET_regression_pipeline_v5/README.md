# MOL-FET Regression Pipeline v5

Bu paket, beş notebookun arka arkaya çalıştığı sade bir regression pipeline'ıdır.

## Eğitim amaçlı v5 düzenlemesi

Bu sürüm, Python ve kemoinformatik kodlamaya yeni başlayan katılımcılar için
hazırlanmıştır.

- Her kod hücresinden önce, hücrenin amacı ve pipeline içindeki yeri açıklanır.
- Her çalıştırılabilir Python satırının hemen üzerinde kısa bir `#` açıklaması bulunur.
- Açıklamalar kodun davranışını değiştirmez.
- Pipeline görev dağılımı, Drive öncelikli veri akışı, GitHub fallback yapısı,
  filtrelenmemiş temiz verinin kullanılması, `|r| > 0.80` feature filtresi ve
  PCA kullanılmaması v4 ile aynı şekilde korunur.

## Temel ilkeler

- Her bilimsel işlem yalnızca bir notebookta yapılır.
- Notebook 00 yalnızca ChEMBL verisi toplar.
- Notebook 01 yalnızca temizleme, duplicate ve Lipinski analizi yapar.
- Notebook 02 yalnızca 2D Mordred feature üretir ve feature filtreler.
- Notebook 03 yalnızca modelleme yapar.
- Notebook 04 yalnızca applicability domain ve SHAP analizi yapar.
- PCA kullanılmaz.
- Notebooklar girdiyi önce Google Drive'dan okur.
- Drive girdisi bulunamazsa GitHub RAW bağlantısı denenir.
- Pipeline modellemede Lipinski filtreli dosyayı kullanmaz.
- Notebook 02, temiz fakat Lipinski ile filtrelenmemiş CSV'yi kullanır.

## Notebook sırası

```text
00_chembl_target_data_store.ipynb
01_data_cleaning.ipynb
02_feature_generation.ipynb
03_lazypredict.ipynb
04_SHAP_interpretability.ipynb
```

## Google Drive klasörü

```text
/content/drive/MyDrive/MOL_FET_regression_pipeline/
```

Notebook 00 aynı klasöre `pipeline_config.json` yazar. Sonraki notebooklar
Target ID ve dosya adlarını bu config dosyasından alır.

## Veri akışı

```text
ChEMBL IC50 activity-level raw CSV
        ↓
Canonical parent SMILES + duplicate aggregation
        ↓
Clean unfiltered CSV + ayrı Lipinski raporları
        ↓
2D Mordred descriptorları
        ↓
Missing > 20%, constant ve |r| > 0.80 filtreleri
        ↓
Train-only median imputer
        ↓
LazyPredict + final tree model + 5-fold CV
        ↓
Mahalanobis AD + SHAP
```

## Notebook 00

Çıktılar:

```text
<TARGET_ID>_IC50_single_protein_raw.csv
<TARGET_ID>_target_metadata.csv
pipeline_config.json
```

Notebook 00 tuz çıkarma, duplicate veya Lipinski işlemi yapmaz.

## Notebook 01

Girdi:

```text
<TARGET_ID>_IC50_single_protein_raw.csv
```

Çıktılar:

```text
<TARGET_ID>_clean_unfiltered.csv
<TARGET_ID>_Lipinski_filtered.csv
<TARGET_ID>_Lipinski_rejected.csv
<TARGET_ID>_invalid_records.csv
<TARGET_ID>_cleaning_report.csv
```

Sonraki notebookun kullandığı dosya:

```text
<TARGET_ID>_clean_unfiltered.csv
```

Lipinski filtreli dosya yalnızca raporlama ve karşılaştırma içindir.

## Notebook 02

Girdi:

```text
<TARGET_ID>_clean_unfiltered.csv
```

Feature kuralları:

```text
2D Mordred only
missing ratio > 0.20 → çıkar
bütün değerler aynı → çıkar
|Pearson r| > 0.80 → çiftlerden birini çıkar
```

Çıktılar:

```text
<TARGET_ID>_Mordred2D_features.csv
<TARGET_ID>_Mordred2D_manifest.json
<TARGET_ID>_Mordred2D_raw.csv
```

Eksik kalan hücreler burada doldurulmaz.

## Notebook 03

Girdi:

```text
<TARGET_ID>_Mordred2D_features.csv
```

İşlemler:

```text
80/20 train-test split
median imputer yalnızca train setinde fit
LazyPredict benchmark
final tree model
train/test metrics
5-fold CV
```

Ana çıktılar:

```text
<TARGET_ID>_final_tree_model.joblib
<TARGET_ID>_feature_imputer.joblib
<TARGET_ID>_model_bundle.json
<TARGET_ID>_LazyPredict_results.csv
<TARGET_ID>_model_metrics.csv
<TARGET_ID>_model_5CV.csv
```

## Notebook 04

Notebook 03 tarafından kaydedilen model, imputer ve bundle dosyalarını yükler.
Modeli yeniden eğitmez ve feature filtrelerini yeniden uygulamaz.

PCA yerine doğrudan standardize Mordred uzayında:

```text
Ledoit-Wolf Mahalanobis distance
train distance 95th percentile threshold
|standardized residual| ≤ 3
```

kullanılır.

Ana çıktılar:

```text
<TARGET_ID>_train_AD.csv
<TARGET_ID>_test_AD.csv
<TARGET_ID>_AD_distance_residual.png
<TARGET_ID>_SHAP_importance.csv
<TARGET_ID>_SHAP_beeswarm.png
<TARGET_ID>_SHAP_bar.png
<TARGET_ID>_SHAP_waterfall.png
<TARGET_ID>_SHAP_dependence.png
<TARGET_ID>_AD_SHAP_summary.json
```

## GitHub fallback yolları

CSV dosyaları için:

```text
data/<dosya_adı>
```

Model artifactları için:

```text
artifacts/<dosya_adı>
```

Önerilen repo yapısı:

```text
MOL-FET_regression/
├── notebooks/
│   ├── 00_chembl_target_data_store.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_generation.ipynb
│   ├── 03_lazypredict.ipynb
│   └── 04_SHAP_interpretability.ipynb
├── data/
│   ├── <TARGET_ID>_IC50_single_protein_raw.csv
│   ├── <TARGET_ID>_clean_unfiltered.csv
│   └── <TARGET_ID>_Mordred2D_features.csv
└── artifacts/
    ├── <TARGET_ID>_final_tree_model.joblib
    ├── <TARGET_ID>_feature_imputer.joblib
    └── <TARGET_ID>_model_bundle.json
```

## Paketler

```text
numpy
pandas
requests
rdkit
mordredcommunity[full]==2.0.7
scikit-learn
lazypredict==0.2.16
joblib
matplotlib
shap
```
