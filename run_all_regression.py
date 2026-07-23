#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
# Dosya ve klasör yollarını yönetmek için kullanılır.
import sys
# Mevcut Python yorumlayıcısını ve komut satırı argümanlarını okumak için kullanılır.
import subprocess
# Scriptleri sırayla çalıştırmak ve eksik paketleri kurmak için kullanılır.
import importlib.util
# Paket kurulu mu kontrol etmek için kullanılır.
import shutil
# Dosyaları GitHub'a yüklenecek klasöre kopyalamak için kullanılır.

EXPORT_ROOT = Path("MOL_FET_regression_github")
# GitHub'a yüklenecek ana klasör.
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
# Klasör yoksa oluşturulur.

SCRIPTS = [
    "01_regression_mordred_feature_store.py",
    "02_rf_mordred_regression_baseline.py",
    "03_tanimoto_similarity_filtration_ad.py",
    "04_lazypredict_mordred_regression.py",
    "05_interpretability_final_report.py",
]
# Localde sırayla çalıştırılacak regression scriptleri.

COPY_PATTERNS = ["*.py", "*.ipynb", "*.md", "*.txt", "*.yml", "training_data.csv"]
# Çalışma bitince GitHub klasörüne kopyalanacak dosya tipleri.

REQUIRED_IMPORTS = [
    ("rdkit", None, "RDKit conda ortamında kurulmalı. Örnek: conda activate rdkit_env"),
    ("mordred", "mordred", "2D Mordred descriptor üretimi için gerekli."),
    ("sklearn", "scikit-learn", "Modelleme ve metrikler için gerekli."),
    ("pandas", "pandas", "CSV ve tablo işlemleri için gerekli."),
    ("numpy", "numpy", "Sayısal işlemler için gerekli."),
    ("matplotlib", "matplotlib", "Grafikler için gerekli."),
    ("joblib", "joblib", "Model kaydetmek için gerekli."),
    ("lazypredict", "lazypredict", "LazyPredict benchmark için gerekli."),
    ("shap", "shap", "SHAP yorumlanabilirlik için gerekli."),
    ("lime", "lime", "LIME açıklamaları için gerekli."),
]
# Paket kontrol listesi. RDKit pip ile zorlanmaz; conda ortamı önerilir.


def package_exists(import_name):
    """Paket import edilebilir mi kontrol eder."""
    return importlib.util.find_spec(import_name) is not None
    # Paket bulunursa True döner.


def install_package(pip_name):
    """Eksik paketi pip ile kurar."""
    print(f"[INSTALL] {pip_name}")
    # Kurulacak paket adı yazdırılır.
    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
    # Paket mevcut Python ortamına kurulur.


def check_and_install_packages(install_missing=True):
    """RDKit dışındaki eksik paketleri isteğe bağlı kurar."""
    missing = []
    # Eksik paketler burada tutulur.
    for import_name, pip_name, message in REQUIRED_IMPORTS:
        # Paketler sırayla kontrol edilir.
        if package_exists(import_name):
            print(f"[OK] {import_name}")
            # Paket varsa OK yazılır.
            continue
        if pip_name is None:
            missing.append((import_name, message))
            # RDKit gibi pip ile kurulmayacak paketler missing listesine alınır.
            continue
        if install_missing:
            install_package(pip_name)
            # Eksik paket pip ile kurulur.
        else:
            missing.append((import_name, message))
            # Kurulum kapalıysa eksik listesine eklenir.
    if missing:
        print("\nEksik kritik paketler:")
        # Eksik paket başlığı yazılır.
        for name, msg in missing:
            print(f"- {name}: {msg}")
            # Eksik paket açıklaması yazılır.
        raise SystemExit("Eksik paketler tamamlanmadan pipeline çalıştırılamaz.")
        # Kritik paket eksikse çalışma durdurulur.


def copy_project_files_to_export():
    """Script, notebook ve README dosyalarını GitHub klasörüne kopyalar."""
    for pattern in COPY_PATTERNS:
        # Her kopyalama paterni sırayla gezilir.
        for path in Path(".").glob(pattern):
            # Mevcut klasörde paterne uyan dosyalar bulunur.
            if path.name == EXPORT_ROOT.name:
                continue
                # Export klasörü dosya gibi ele alınmaz.
            if path.is_file():
                shutil.copy2(path, EXPORT_ROOT / path.name)
                # Dosya GitHub klasörünün köküne kopyalanır.
    print(f"[Kopyalama tamamlandı] {EXPORT_ROOT.resolve()}")
    # Kopyalama tamamlandı mesajı yazılır.


def main():
    """Bütün regression pipeline scriptlerini sırayla çalıştırır."""
    install_missing = "--no-install" not in sys.argv
    # --no-install verilirse eksik paketler otomatik kurulmaz.
    check_only = "--check-only" in sys.argv
    # --check-only verilirse sadece paket kontrolü yapılır.
    check_and_install_packages(install_missing=install_missing)
    # Paket kontrolü yapılır.
    if check_only:
        print("Paket kontrolü tamamlandı. --check-only nedeniyle modelleme başlatılmadı.")
        # Check-only modunda bilgi verilir.
        return
        # Çalışma bitirilir.

    copy_project_files_to_export()
    # Önce kod ve notebook dosyaları export klasörüne kopyalanır.

    for script in SCRIPTS:
        # Scriptler sırayla çalıştırılır.
        print("\n" + "=" * 100)
        print(f"ÇALIŞTIRILIYOR: {script}")
        print("=" * 100)
        subprocess.check_call([sys.executable, script])
        # İlgili script mevcut Python ortamında çalıştırılır.

    copy_project_files_to_export()
    # Çalışma sonrası güncel dosyalar tekrar export klasörüne kopyalanır.
    print("\n" + "=" * 100)
    print("TÜM REGRESSION PIPELINE TAMAMLANDI")
    print("=" * 100)
    print(f"GitHub'a yüklenecek klasör: {EXPORT_ROOT.resolve()}")
    print("Bu klasörün içine girip GitHub'a push edebilirsiniz.")
    # Final yönerge yazdırılır.


if __name__ == "__main__":
    main()
    # Script doğrudan çalıştırılırsa ana fonksiyon başlatılır.
