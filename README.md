# MOL-FET Regression Pipeline v3 — 2D Mordred ve Interpretability Odaklı Akış

Bu paket, regression problemi için classification paketindeki ders formatına benzer şekilde hazırlanmıştır; ancak amaç ve akış farklıdır.

Bu sürümde model feature olarak **yalnızca 2D Mordred descriptorları** kullanılır. Morgan, MACCS, Avalon veya başka fingerprint kolonları model girdisi olarak kullanılmaz.

Tanimoto similarity yalnızca applicability-domain / filtration analizi için kullanılır. Yani fingerprint, model feature değildir; sadece test molekülünün train setine ne kadar benzediğini ölçmek için kullanılır.

## Ana farklar

```text
Classification akışı:
- Feature ablation
- Feature selection
- Çok model seçimi
- Sampling/tuning/ensemble ile performans artırma

Regression v3 akışı:
- 2D Mordred descriptor üretimi
- RF + Mordred baseline
- Tanimoto similarity based filtration / applicability domain
- LazyPredict ile automated regression benchmark
- Interpretability: permutation importance, SHAP, LIME, residual analizi
```

Bu regression akışında manuel feature selection yapılmaz. Manuel model seçimi de ana amaç değildir. LazyPredict sonucu model ailesi davranışını görmek için üretilir.

## Dosyalar

```text
01_regression_mordred_feature_store.py / .ipynb
02_rf_mordred_regression_baseline.py / .ipynb
03_tanimoto_similarity_filtration_ad.py / .ipynb
04_lazypredict_mordred_regression.py / .ipynb
05_interpretability_final_report.py / .ipynb
run_all_regression.py
training_data.csv
requirements_regression.txt
environment.yml
setup_local_env.sh
```

## Local çalışma

Önerilen yöntem, RDKit içeren conda ortamında çalışmaktır.

```bash
conda activate rdkit_env
python run_all_regression.py
```

Sadece paket kontrolü:

```bash
python run_all_regression.py --check-only
```

Eksik Python paketlerini otomatik kurmak istemezseniz:

```bash
python run_all_regression.py --no-install
```

## GitHub'a yüklenecek klasör

`run_all_regression.py` çalışınca şu klasör oluşur:

```text
MOL_FET_regression_github/
```

Bu klasörün içine girip GitHub'a yükleyebilirsiniz:

```bash
cd MOL_FET_regression_github
git init
git add .
git commit -m "Add MOL-FET regression pipeline v3"
git branch -M main
git remote add origin https://github.com/MOL-FEST/MOL-FET_regression.git
git push -u origin main
```

## Colab mantığı

Notebooklar önce local dosyayı okumaya çalışır. Dosya yoksa GitHub raw linkinden okur:

```text
https://raw.githubusercontent.com/MOL-FEST/MOL-FET_regression/main/training_data.csv
https://raw.githubusercontent.com/MOL-FEST/MOL-FET_regression/main/molfet_regression_outputs/...
```

Bu nedenle localde pipeline'ı bir kez çalıştırıp `MOL_FET_regression_github/` klasörünü GitHub'a attıktan sonra notebooklar Colab'da checkpoint dosyalarını GitHub'dan okuyabilir.

## Output klasörleri

```text
molfet_regression_outputs/01_mordred_feature_store/
molfet_regression_outputs/02_rf_mordred_baseline/
molfet_regression_outputs/03_tanimoto_similarity_filtration/
molfet_regression_outputs/04_lazypredict_regression/
molfet_regression_outputs/05_interpretability_final_report/
```

## 5 notebook özeti

### 01 — 2D Mordred feature store

Ham veriyi okur, target/SMILES temizliği yapar, sadece 2D Mordred descriptor üretir ve `mordred_2d_features.csv` dosyasını kaydeder.

### 02 — RF + Mordred baseline

2D Mordred feature store dosyasını okur, RandomForestRegressor baseline modelini eğitir, predicted-vs-actual ve residual grafikleri üretir.

### 03 — Tanimoto similarity based filtration

RF modeli yine Mordred descriptorlarıyla eğitilir. Tanimoto similarity yalnızca test moleküllerinin train setine benzerliğini ölçmek için kullanılır. Farklı similarity eşiklerinde coverage, MAE, RMSE ve R2 raporlanır.

### 04 — LazyPredict regression

2D Mordred descriptorları üzerinde LazyPredict ile automated regression benchmark üretir. Bu adım model seçimi için değil, model ailelerini tartışmak için vardır.

### 05 — Interpretability final report

RF + Mordred modelini yorumlanabilirlik için tekrar eğitir. Permutation importance, SHAP, LIME, residual analizi ve final özet tabloları üretir.

## Regression metrikleri

```text
MAE  = ortalama mutlak hata
RMSE = büyük hataları daha fazla cezalandıran hata metriği
R2   = açıklanan varyans oranı
MBE  = mean bias error; pozitif değer overprediction eğilimini gösterir
```

## Ders anlatım önerisi

```text
Ders 1: 01 + 02
Ders 2: 03
Ders 3: 04
Ders 4: 05 interpretability çıktıları
Ders 5: genel değerlendirme, hata analizi ve applicability-domain yorumu
```
