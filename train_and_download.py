#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import requests
import shutil
import glob
import logging

# POD_ID задаётся вручную
POD_ID = "2rym4rcubd26lu"  # Замените на реальный POD_ID

# Получаем API-ключ RunPod из переменной окружения
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("❌ API-ключ не найден! Передайте его через export RUNPOD_API_KEY=...")
    sys.exit(1)

print(f"✅ Используется POD_ID: {POD_ID}")

# Локальная структура папок (папка Ready)
BASE_LOCAL_DIR = r"C:\Users\Kroha\Documents\trading-models\Ready"
NEURAL_NETWORKS_DIR = os.path.join(BASE_LOCAL_DIR, "neural_networks")
ENSEMBLE_MODELS_DIR = os.path.join(BASE_LOCAL_DIR, "ensemble_models")
os.makedirs(NEURAL_NETWORKS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_MODELS_DIR, exist_ok=True)

# Пути, где во время обучения модели сохраняются (например, в OUTPUT_DIR)
MODELS_DIR = os.path.join(BASE_LOCAL_DIR, "models", "neural_networks")
OUTPUT_DIR = os.path.join(BASE_LOCAL_DIR, "output", "neural_networks")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Зависимости (оставьте, как есть)
REQUIRED_PACKAGES = [
    "numpy", "pandas", "matplotlib", "scipy", "tensorflow[and-cuda]==2.12.0", "tensorflow-addons",
    "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "lightgbm", "joblib",
    "ta", "pandas-ta", "python-binance", "filterpy", "requests"
]

def install_packages():
    print("✅ Устанавливаем зависимости...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy==1.23.5"], check=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "tensorflow"], check=True)
    except subprocess.CalledProcessError:
        print("⚠ Ошибка при установке TensorFlow, устанавливаем tensorflow-cpu...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "tensorflow-cpu"], check=True)
    for package in REQUIRED_PACKAGES:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", package], check=True)
        except subprocess.CalledProcessError:
            print(f"⚠ Ошибка установки пакета: {package}")

def check_gpu():
    print("\n🔍 Проверяем доступность GPU...")
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if output.returncode == 0:
            print("✅ GPU доступен!")
            print(output.stdout)
        else:
            print("⚠ GPU не найден, обучение будет идти на CPU.")
    except FileNotFoundError:
        print("⚠ nvidia-smi не найден, вероятно, GPU недоступен.")

check_gpu()

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

# Модельная карта: имя файла модели -> имя модели (для создания подпапок)
MODELS = {
    "market_condition_classifier.py": "Market_Classifier",
    "bullish_neural_network.py": "Neural_Bullish",
    "bullish_ensemble.py": "Ensemble_Bullish",
    "flat_neural_network.py": "Neural_Flat",
    "flat_ensemble.py": "Ensemble_Flat",
    "bearish_neural_network.py": "Neural_Bearish",
    "bearish_ensemble.py": "Ensemble_Bearish"
}

def train_models():
    print("\n🚀 Запускаем обучение моделей...")
    # Для каждой модели копируем файл из OUTPUT_DIR в соответствующую папку локального сохранения
    for model_file, model_name in MODELS.items():
        if "ensemble" in model_name.lower():
            parent_dir = ENSEMBLE_MODELS_DIR
        else:
            parent_dir = NEURAL_NETWORKS_DIR
        model_folder = os.path.join(parent_dir, model_name)
        os.makedirs(model_folder, exist_ok=True)
        # Предполагается, что обученная модель сохраняется в OUTPUT_DIR с именем {model_name}.h5
        trained_model_path = os.path.join(OUTPUT_DIR, f"{model_name}.h5")
        local_model_path = os.path.join(model_folder, f"{model_name}.h5")
        if os.path.exists(trained_model_path):
            print(f"📥 Копирование модели {model_name} в {local_model_path}...")
            try:
                subprocess.run(["cp", trained_model_path, local_model_path], check=True)
                print(f"✅ Модель {model_name} успешно сохранена в {local_model_path}!")
            except subprocess.CalledProcessError:
                print(f"⚠ Ошибка при копировании модели: {model_file}")
        else:
            print(f"⚠ Файл модели не найден: {trained_model_path}")

# Запускаем копирование моделей для локального сохранения
train_models()

# Сохранение моделей на диске в RunPod (для временного хранения)
RUNPOD_SAVE_DIR = "/workspace/saved_models"
os.makedirs(RUNPOD_SAVE_DIR, exist_ok=True)
print("\n🚀 Сохраняем модели на диске в RunPod...")
for model_file, model_name in MODELS.items():
    src_file = os.path.join(OUTPUT_DIR, f"{model_name}.h5")
    if os.path.exists(src_file):
        dest_file = os.path.join(RUNPOD_SAVE_DIR, f"{model_name}.h5")
        try:
            subprocess.run(["cp", src_file, dest_file], check=True)
            print(f"✅ Модель {model_name} сохранена на RunPod: {dest_file}")
        except subprocess.CalledProcessError:
            print(f"⚠ Ошибка при сохранении модели {model_name} на RunPod.")
    else:
        print(f"⚠ Файл модели не найден для сохранения на RunPod: {src_file}")

# Остановка пода на RunPod
if RUNPOD_API_KEY:
    print("\n🔧 Работаем с RunPod...")
    try:
        subprocess.run(["pip", "install", "runpod"], check=True)
    except subprocess.CalledProcessError:
        print("⚠ Ошибка установки runpod CLI, под не будет остановлен.")
        sys.exit(1)
    try:
        response = requests.get(
            "https://api.runpod.io/v2/pod/list",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        pods = response.json()
        if "pods" in pods and pods["pods"]:
            print(f"✅ Найден под с ID: {POD_ID}")
            requests.post(
                f"https://api.runpod.io/v2/pod/{POD_ID}/stop",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            print(f"✅ Под {POD_ID} остановлен.")
            requests.delete(
                f"https://api.runpod.io/v2/pod/{POD_ID}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            print(f"✅ Под {POD_ID} удалён.")
        else:
            print("⚠ Подов не найдено.")
    except Exception as e:
        print(f"⚠ Ошибка при управлении RunPod: {e}")

print("\n🎉 Обучение завершено, все модели сохранены!")
