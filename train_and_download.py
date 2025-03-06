#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import shutil
import requests

# -----------------------------------------------------------------------------
# 1. Конфигурация
# -----------------------------------------------------------------------------

# POD_ID и RUNPOD_API_KEY для управления подом (если нужно остановить под после обучения)
POD_ID = "zeqm08zb5eg1ik"  # Замените на реальный POD_ID
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# Абсолютный путь к репозиторию
BASE_REPO_DIR = "/workspace/trading-bot-"

# Жестко прописанные пути к обучающим скриптам
SCRIPTS = [
    ("/workspace/trading-bot-/neural_networks/market_condition_classifier.py", "Market_Classifier"),
    ("/workspace/trading-bot-/neural_networks/bullish_neural_network.py", "Neural_Bullish"),
    ("/workspace/trading-bot-/ensemble_models/bullish_ensemble.py", "Ensemble_Bullish"),
    ("/workspace/trading-bot-/neural_networks/flat_neural_network.py", "Neural_Flat"),
    ("/workspace/trading-bot-/ensemble_models/flat_ensemble.py", "Ensemble_Flat"),
    ("/workspace/trading-bot-/neural_networks/bearish_neural_network.py", "Neural_Bearish"),
    ("/workspace/trading-bot-/ensemble_models/bearish_ensemble.py", "Ensemble_Bearish")
]

# Предполагаемые папки с результатами, которые создают ваши скрипты (измените, если нужно)
EXPECTED_OUTPUT = {
    "Market_Classifier": "/workspace/trading-bot-/neural_networks/output/market_condition_classifier",
    "Neural_Bullish": "/workspace/trading-bot-/neural_networks/output/bullish_neural_network",
    "Ensemble_Bullish": "/workspace/trading-bot-/ensemble_models/output/bullish_ensemble",
    "Neural_Flat": "/workspace/trading-bot-/neural_networks/output/flat_neural_network",
    "Ensemble_Flat": "/workspace/trading-bot-/ensemble_models/output/flat_ensemble",
    "Neural_Bearish": "/workspace/trading-bot-/neural_networks/output/bearish_neural_network",
    "Ensemble_Bearish": "/workspace/trading-bot-/ensemble_models/output/bearish_ensemble"
}

# Жестко прописанные целевые пути для сохранения результатов
LOCAL_SAVE_BASE = r"C:\Users\Kroha\Documents\trading-models\Ready"
# Для нейросетей: LOCAL_SAVE_BASE\neural_networks, для ансамблей: LOCAL_SAVE_BASE\ensemble_models
RUNPOD_SAVE_BASE = "/workspace/saved_models"

# -----------------------------------------------------------------------------
# 2. Утилиты
# -----------------------------------------------------------------------------

def install_packages():
    print("Устанавливаем зависимости...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    if os.path.exists("requirements.txt"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    # Если requirements.txt нет, можно прописать установку вручную:
    # subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]==2.12.0", "numpy", ...], check=True)

def check_gpu():
    print("\nПроверяем доступность GPU...")
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if output.returncode == 0:
            print("GPU доступен:")
            print(output.stdout)
        else:
            print("GPU не найден, обучение будет идти на CPU")
    except Exception as e:
        print("nvidia-smi не найден:", e)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def copy_output(model_name, source_dir):
    """
    Копирует содержимое source_dir в две целевые локации:
    - на локальный диск (Windows) в папку нейросетей или ансамблей
    - на диск RunPod (RUNPOD_SAVE_BASE)
    """
    if "Ensemble" in model_name:
        local_target = os.path.join(LOCAL_SAVE_BASE, "ensemble_models", model_name)
    else:
        local_target = os.path.join(LOCAL_SAVE_BASE, "neural_networks", model_name)
    ensure_directory(local_target)
    
    runpod_target = os.path.join(RUNPOD_SAVE_BASE, model_name)
    ensure_directory(RUNPOD_SAVE_BASE)
    ensure_directory(runpod_target)
    
    if os.path.exists(source_dir):
        try:
            # Копируем всю папку (сопутствующие файлы, логи, чекпоинты и т.п.)
            shutil.copytree(source_dir, local_target, dirs_exist_ok=True)
            shutil.copytree(source_dir, runpod_target, dirs_exist_ok=True)
            print(f"Результаты для {model_name} скопированы в:\n  {local_target}\n  {runpod_target}")
        except Exception as e:
            print(f"Ошибка при копировании результатов для {model_name}: {e}")
    else:
        print(f"Не найдена папка с результатами для {model_name}: {source_dir}")

# -----------------------------------------------------------------------------
# 3. Логика обучения
# -----------------------------------------------------------------------------

def run_training():
    install_packages()
    check_gpu()
    
    # Для каждого скрипта запускаем обучение
    for script_path, model_name in SCRIPTS:
        print(f"\nЗапускаем скрипт обучения: {script_path} ({model_name})")
        try:
            subprocess.run(["python", script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при запуске {script_path}: {e}")
            continue
        
        # Ждем 5 секунд для завершения записи результатов
        time.sleep(5)
        
        # Копируем результаты для данной модели
        source_dir = EXPECTED_OUTPUT.get(model_name)
        if source_dir:
            copy_output(model_name, source_dir)
        else:
            print(f"Для модели {model_name} не задан источник результатов.")

def stop_and_remove_pod():
    if not RUNPOD_API_KEY or not POD_ID:
        print("RUNPOD_API_KEY или POD_ID не заданы, пропускаем остановку пода.")
        return
    print("\nОстанавливаем под RunPod...")
    try:
        subprocess.run(["pip", "install", "runpod"], check=True)
    except subprocess.CalledProcessError:
        print("Ошибка установки runpod CLI, пропускаем остановку пода.")
        return
    try:
        response = requests.get("https://api.runpod.io/v2/pod/list",
                                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"})
        pods = response.json()
        if "pods" in pods and pods["pods"]:
            print(f"Найден под с ID: {POD_ID}")
            requests.post(f"https://api.runpod.io/v2/pod/{POD_ID}/stop",
                          headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"})
            print(f"Под {POD_ID} остановлен.")
            requests.delete(f"https://api.runpod.io/v2/pod/{POD_ID}",
                          headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"})
            print(f"Под {POD_ID} удалён.")
        else:
            print("Подов не найдено. Проверьте POD_ID.")
    except Exception as e:
        print(f"Ошибка при управлении подом RunPod: {e}")

def main():
    run_training()
    stop_and_remove_pod()
    print("\nОбучение завершено. Результаты сохранены локально и на диске RunPod.")

if __name__ == "__main__":
    main()
