#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import requests
import shutil

# -----------------------------------------------------------------------------
# 1. Конфигурация путей и моделей
# -----------------------------------------------------------------------------

# POD_ID и RUNPOD_API_KEY для управления подом (при необходимости)
POD_ID = "2rym4rcubd26lu"  # Замените на реальный POD_ID
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# Локальные каталоги на Windows для сохранения результатов
BASE_LOCAL_DIR = r"C:\Users\Kroha\Documents\trading-models\Ready"
NEURAL_NETWORKS_DIR = os.path.join(BASE_LOCAL_DIR, "neural_networks")
ENSEMBLE_MODELS_DIR = os.path.join(BASE_LOCAL_DIR, "ensemble_models")

# Каталог для сохранения на диске RunPod
RUNPOD_SAVE_DIR = "/workspace/saved_models"

# Словарь с порядком запуска: ключ – имя скрипта, значение – логическое имя модели
# Порядок следования в словаре (Python 3.7+) определяет последовательность обучения
MODELS = {
    "market_condition_classifier.py": "Market_Classifier",
    "bullish_neural_network.py": "Neural_Bullish",
    "bullish_ensemble.py": "Ensemble_Bullish",
    "flat_neural_network.py": "Neural_Flat",
    "flat_ensemble.py": "Ensemble_Flat",
    "bearish_neural_network.py": "Neural_Bearish",
    "bearish_ensemble.py": "Ensemble_Bearish"
}

# Для каждого логического имени задаём ожидаемые файлы и каталоги с результатами,
# которые будут копироваться после обучения.
# Измените пути, если ваши скрипты сохраняют результаты в иных локациях.
EXPECTED_RESULTS = {
    "Market_Classifier": {
        "files": ["market_condition_classifier.h5", "scaler.pkl"],
        "dirs": ["checkpoints/market_condition_classifier"]
    },
    "Neural_Bullish": {
        "files": ["bullish_nn_model.h5"],
        "dirs": ["checkpoints/bullish_neural_network"]
    },
    "Ensemble_Bullish": {
        "files": [os.path.join("models", "bullish_stacked_ensemble_model.pkl")],
        "dirs": ["checkpoints/bullish"]
    },
    "Neural_Flat": {
        "files": ["flat_nn_model.h5"],
        "dirs": ["checkpoints/flat_neural_network"]
    },
    "Ensemble_Flat": {
        "files": [os.path.join("models", "flat_stacked_ensemble_model.pkl")],
        "dirs": ["checkpoints/flat"]
    },
    "Neural_Bearish": {
        "files": ["bearish_nn_model.h5"],
        "dirs": ["checkpoints/bearish_neural_network"]
    },
    "Ensemble_Bearish": {
        "files": [os.path.join("models", "bearish_stacked_ensemble_model.pkl")],
        "dirs": ["checkpoints/bearish"]
    },
}

# Список необходимых зависимостей
REQUIRED_PACKAGES = [
    "numpy", "pandas", "matplotlib", "scipy",
    "tensorflow[and-cuda]==2.12.0", "tensorflow-addons",
    "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "lightgbm",
    "joblib", "ta", "pandas-ta", "python-binance", "filterpy", "requests"
]

# -----------------------------------------------------------------------------
# 2. Утилиты
# -----------------------------------------------------------------------------

def install_packages():
    print("✅ Устанавливаем зависимости...")
    # Обновляем pip и устанавливаем numpy нужной версии
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy==1.23.5"], check=True)
    
    # Пытаемся установить tensorflow (если не получилось, tensorflow-cpu)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "tensorflow"], check=True)
    except subprocess.CalledProcessError:
        print("⚠ Ошибка установки tensorflow, устанавливаем tensorflow-cpu...")
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
        print("⚠ nvidia-smi не найден, возможно, GPU недоступен.")

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def copy_results(model_logical_name):
    """
    Для заданного логического имени модели копирует ожидаемые файлы и каталоги из
    текущей директории в целевые папки (локально и на RunPod).
    """
    expected = EXPECTED_RESULTS.get(model_logical_name, {})
    files = expected.get("files", [])
    dirs = expected.get("dirs", [])
    
    # Определяем целевую локальную папку в зависимости от типа модели (ансамбль или нейросеть)
    if "Ensemble" in model_logical_name:
        local_base = ENSEMBLE_MODELS_DIR
    else:
        local_base = NEURAL_NETWORKS_DIR

    local_target = os.path.join(local_base, model_logical_name)
    ensure_directory(local_target)
    ensure_directory(RUNPOD_SAVE_DIR)
    runpod_target = os.path.join(RUNPOD_SAVE_DIR, model_logical_name)
    ensure_directory(runpod_target)
    
    # Копируем файлы
    for f in files:
        if os.path.exists(f):
            try:
                shutil.copy(f, local_target)
                shutil.copy(f, runpod_target)
                print(f"✅ Файл {f} скопирован в {local_target} и {runpod_target}")
            except Exception as e:
                print(f"⚠ Ошибка копирования файла {f}: {e}")
        else:
            print(f"⚠ Файл не найден: {f}")
    
    # Рекурсивно копируем каталоги
    for d in dirs:
        if os.path.exists(d):
            dest_local = os.path.join(local_target, os.path.basename(d))
            dest_runpod = os.path.join(runpod_target, os.path.basename(d))
            try:
                # Если папка уже существует, удаляем её перед копированием
                if os.path.exists(dest_local):
                    shutil.rmtree(dest_local, onerror=on_rm_error)
                shutil.copytree(d, dest_local)
                if os.path.exists(dest_runpod):
                    shutil.rmtree(dest_runpod, onerror=on_rm_error)
                shutil.copytree(d, dest_runpod)
                print(f"✅ Каталог {d} скопирован в {dest_local} и {dest_runpod}")
            except Exception as e:
                print(f"⚠ Ошибка копирования каталога {d}: {e}")
        else:
            print(f"⚠ Каталог не найден: {d}")

def on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, 0o777)
        func(path)
    except Exception as e:
        print(f"⚠ Ошибка при удалении {path}: {e}")

# -----------------------------------------------------------------------------
# 3. Логика обучения
# -----------------------------------------------------------------------------

def run_training_scripts():
    """
    По очереди запускает каждый обучающий скрипт из MODELS,
    а после завершения его работы копирует ожидаемые результаты.
    """
    for script, model_name in MODELS.items():
        print(f"\n🚀 Запускаем обучение: {script} ({model_name})")
        try:
            subprocess.run(["python", script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠ Ошибка при запуске {script}: {e}")
            continue
        
        # Даем время на завершение записи файлов (если требуется)
        time.sleep(5)
        print(f"📥 Копирование результатов для {model_name}...")
        copy_results(model_name)

# -----------------------------------------------------------------------------
# 4. Управление подом RunPod (опционально)
# -----------------------------------------------------------------------------

def stop_and_remove_pod():
    if not RUNPOD_API_KEY or not POD_ID:
        print("⚠ RUNPOD_API_KEY или POD_ID не заданы, пропускаем остановку пода.")
        return
    
    print("\n🔧 Работаем с RunPod...")
    try:
        subprocess.run(["pip", "install", "runpod"], check=True)
    except subprocess.CalledProcessError:
        print("⚠ Ошибка установки runpod CLI, под не будет остановлен.")
        return
    
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
            print("⚠ Подов не найдено. Возможно, неправильный POD_ID.")
    except Exception as e:
        print(f"⚠ Ошибка при управлении RunPod: {e}")

# -----------------------------------------------------------------------------
# 5. Основной блок
# -----------------------------------------------------------------------------

def main():
    install_packages()
    check_gpu()
    run_training_scripts()
    stop_and_remove_pod()
    print("\n🎉 Все обучения завершены, результаты сохранены локально и на RunPod!")

if __name__ == "__main__":
    main()
