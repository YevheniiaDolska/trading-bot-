import os
import shutil

# Укажите пути, куда сохранять результаты локально и на RunPod
LOCAL_SAVE_BASE = r"C:\Users\Kroha\Documents\trading-models\Ready"
RUNPOD_SAVE_BASE = "/workspace/saved_models"

def ensure_directory(path):
    """Создаёт директорию, если она не существует."""
    os.makedirs(path, exist_ok=True)

def copy_output(model_name, source_dir):
    """
    Копирует содержимое source_dir в две целевые локации:
      - локально (для нейросетей – LOCAL_SAVE_BASE\neural_networks или для ансамблей – LOCAL_SAVE_BASE\ensemble_models)
      - на диск RunPod (RUNPOD_SAVE_BASE/model_name)
    """
    # Определяем локальную директорию в зависимости от типа модели
    if "ensemble" in model_name.lower():
        local_target = os.path.join(LOCAL_SAVE_BASE, "ensemble_models", model_name)
    else:
        local_target = os.path.join(LOCAL_SAVE_BASE, "neural_networks", model_name)
    ensure_directory(local_target)
    
    # Директория для сохранения на RunPod
    runpod_target = os.path.join(RUNPOD_SAVE_BASE, model_name)
    ensure_directory(RUNPOD_SAVE_BASE)
    ensure_directory(runpod_target)
    
    if os.path.exists(source_dir):
        try:
            shutil.copytree(source_dir, local_target, dirs_exist_ok=True)
            shutil.copytree(source_dir, runpod_target, dirs_exist_ok=True)
            print(f"Результаты для {model_name} скопированы в:\n  {local_target}\n  {runpod_target}")
        except Exception as e:
            print(f"Ошибка при копировании результатов для {model_name}: {e}")
    else:
        print(f"Источник результатов для {model_name} не найден: {source_dir}")

def save_model_output(model, model_name, source_dir, save_func):
    """
    Универсальная функция сохранения модели.
    
    Параметры:
      model       - обученная модель (например, Keras-модель или scikit-learn объект)
      model_name  - строковое имя модели (например, "Neural_Bullish" или "Ensemble_Bullish")
      source_dir  - директория, куда сохраняется модель внутри скрипта обучения
      save_func   - функция сохранения модели, которую вы передаёте. Например:
                      для Keras: lambda model, path: model.save(path)
                      для scikit-learn: lambda model, path: joblib.dump(model, path)
    
    Функция сначала сохраняет модель в source_dir, а затем вызывает copy_output для копирования результатов.
    """
    ensure_directory(source_dir)
    model_save_path = os.path.join(source_dir, model_name + ".h5")
    try:
        save_func(model, model_save_path)
        print(f"Модель {model_name} сохранена по пути: {model_save_path}")
    except Exception as e:
        print(f"Ошибка при сохранении модели {model_name}: {e}")
    copy_output(model_name, source_dir)
