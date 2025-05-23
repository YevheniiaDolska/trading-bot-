import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.combine import SMOTETomek
from joblib import Parallel, delayed
from binance.client import Client
from datetime import datetime, timedelta
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, AccDistIndexIndicator
from sklearn.utils.class_weight import compute_class_weight
from ta.trend import ADXIndicator
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from filterpy.kalman import KalmanFilter
import requests
import zipfile
from io import BytesIO
import tensorflow as tf
from threading import Lock
from joblib import parallel_backend
import stat
from sklearn.base import clone
from datetime import datetime
import joblib, glob, shutil, logging, dill, time
from datetime import datetime
from utils_output import ensure_directory, copy_output, save_model_output
from sklearn.linear_model import LogisticRegression

  

# Создаем необходимые директории
required_dirs = [
    "/workspace/logs",
    "/workspace/saved_models/bearish",
    "/workspace/checkpoints/bearish",
    "/workspace/data",
    "/workspace/output/bearish_ensemble"
]

for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("/workspace/logs", "debug_log_bearish_ensemble.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # Вывод в консоль с поддержкой юникода
    ]
)

# Универсальная функция для чанкования DataFrame
def apply_in_chunks(df, func, chunk_size=100000):
    """
    Применяет функцию func к DataFrame df по чанкам заданного размера.
    Если df не является DataFrame, возвращает func(df).
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return func(df)
    # Если датасет меньше одного чанка, сразу возвращаем результат
    if len(df) <= chunk_size:
        return func(df)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    processed_chunks = [func(chunk) for chunk in chunks]
    return pd.concat(processed_chunks)


# Имя файла для сохранения модели
market_type = "bearish"

ensemble_model_filename = os.path.join("/workspace/saved_models/bearish", "bearish_stacked_ensemble_model.pkl")

checkpoint_base_dir = os.path.join("/workspace/checkpoints/bearish", market_type)

ensemble_checkpoint_path = os.path.join(checkpoint_base_dir, f"{market_type}_ensemble_checkpoint.pkl")



def initialize_strategy():
    """
    Инициализирует стратегию для GPU, если они доступны.
    Если GPU нет, возвращает стандартную стратегию (CPU или один GPU, если он есть).
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Опционально: включаем динамическое выделение памяти,
            # чтобы TensorFlow не занимал всю память сразу
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            strategy = tf.distribute.MirroredStrategy()  # Распределённая стратегия для одного или нескольких GPU
            print("Running on GPU(s) with strategy:", strategy)
        except RuntimeError as e:
            print("Ошибка при инициализации GPU-стратегии:", e)
            strategy = tf.distribute.get_strategy()
    else:
        print("GPU не найдены. Используем стандартную стратегию.")
        strategy = tf.distribute.get_strategy()
    return strategy


# Функция для создания директорий
# Функция создания директории (exist_ok=True)
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

# Обработчик ошибок удаления
def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logging.debug(f"Не удалось удалить {path}: {e}")

# Функция для формирования пути к чекпоинтам
def get_checkpoint_path(model_name, market_type):
    cp_path = os.path.join("/workspace/checkpoints/bearish", market_type, model_name)
    ensure_directory(cp_path)
    return cp_path
        

def save_logs_to_file(message):
    """
    Сохраняет логи в файл внутри директории /workspace/logs.
    """
    log_dir = "/workspace/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join("/workspace/logs", "training_log_bearish_nn.txt")
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} - {message}\n")

        
        
def calculate_cross_coin_features(data_dict):
    btc_data = data_dict['BTCUSDC']
    for symbol, df in data_dict.items():
        # CHANGED FOR SCALPING
        df['btc_corr'] = df['close'].rolling(15).corr(btc_data['close'])
        df['rel_strength_btc'] = (df['close'].pct_change() - btc_data['close'].pct_change())
        # CHANGED FOR SCALPING
        df['beta_btc'] = (
            df['close'].pct_change().rolling(15).cov(btc_data['close'].pct_change())
            / btc_data['close'].pct_change().rolling(15).var()
        )
        # CHANGED FOR SCALPING
        df['lead_lag_btc'] = df['close'].pct_change().shift(1).rolling(5).corr(btc_data['close'].pct_change())
        data_dict[symbol] = df
    return data_dict


def detect_anomalies(data):
    """
    Детектирует и фильтрует аномальные свечи.
    Для торговли на колебаниях используется более короткое окно (10 свечей) и сниженный порог обнаружения.
    """
    # Рассчитываем z-score для объёма и цены по окну из 10 свечей
    data['volume_zscore'] = ((data['volume'] - data['volume'].rolling(10).mean()) / 
                             data['volume'].rolling(10).std())
    data['price_zscore'] = ((data['close'] - data['close'].rolling(10).mean()) / 
                            data['close'].rolling(10).std())
    
    # Используем порог 2.5 вместо 3 для более чувствительного обнаружения экстремумов
    data['is_anomaly'] = ((abs(data['volume_zscore']) > 2.5) & (data['close'] < data['close'].shift(1))) | \
                         (abs(data['price_zscore']) > 2.5)
    return data



def validate_volume_confirmation(data):
    # CHANGED FOR SCALPING
    data['volume_trend_conf'] = np.where(
        (data['close'] > data['close'].shift(1)) &
        (data['volume'] > data['volume'].rolling(5).mean()),
        1,
        np.where(
            (data['close'] < data['close'].shift(1)) &
            (data['volume'] > data['volume'].rolling(5).mean()),
            -1,
            0
        )
    )
    data['volume_strength'] = (data['volume'] / data['volume'].rolling(5).mean()) * data['volume_trend_conf']
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(2).sum()
    return data

def remove_noise(data):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    # CHANGED FOR SCALPING
    kf.R = 2
    kf.Q = np.array([[0.1, 0.1],[0.1, 0.1]])  # Понижено для быстрого отклика фильтра
    smoothed_prices = []
    for price in data['close']:
        kf.predict()
        kf.update(price)
        smoothed_prices.append(float(kf.x[0]))
    data['smoothed_close'] = smoothed_prices
    return data



def preprocess_market_data(data_dict):
    """
    Комплексная предобработка данных с учетом межмонетных взаимосвязей.
    """
    # Добавляем межмонетные признаки
    data_dict = calculate_cross_coin_features(data_dict)
    
    for symbol, df in data_dict.items():
        # Детектируем аномалии
        df = detect_anomalies(df)
        
        # Добавляем подтверждение объемом
        df = validate_volume_confirmation(df)
        
        # Фильтруем шум
        df = remove_noise(df)
        
        data_dict[symbol] = df
    
    return data_dict


# ------------------ CheckpointGradientBoosting ------------------
class CheckpointGradientBoosting(GradientBoostingClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, #n_estimators=100
                 random_state=None, subsample=1.0, min_samples_split=2,
                 min_samples_leaf=1, **kwargs):
        # Извлекаем _checkpoint_dir, если передан, иначе создаём новый путь
        self._checkpoint_dir = kwargs.pop('_checkpoint_dir', get_checkpoint_path("gradient_boosting", market_type))
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, random_state=random_state,
                         subsample=subsample, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_checkpoint_dir', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._checkpoint_dir = get_checkpoint_path("gradient_boosting", market_type)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.pop('_checkpoint_dir', None)
        return params

    def fit(self, X, y):
        logging.info("[GradientBoosting] Начало обучения с чекпоинтами")
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]
        existing_stages = []
        for i in range(self.n_estimators):
            cp_path = os.path.join(self._checkpoint_dir, f"gradient_boosting_checkpoint_{i+1}.pkl")
            if os.path.exists(cp_path):
                try:
                    with open(cp_path, 'rb') as f:
                        stage = dill.load(f)
                    if hasattr(stage, 'n_features_') and stage.n_features_ == n_features:
                        existing_stages.append(stage)
                        logging.info(f"[GradientBoosting] Загружена итерация {i+1} из чекпоинта")
                except Exception as e:
                    logging.warning(f"[GradientBoosting] Не удалось загрузить чекпоинт {i+1}: {e}")
        if not existing_stages:
            if os.path.exists(self._checkpoint_dir):
                shutil.rmtree(self._checkpoint_dir, onerror=_on_rm_error)
            ensure_directory(self._checkpoint_dir)
            logging.info("[GradientBoosting] Начинаем обучение с нуля")
            super().fit(X, y)
        else:
            self.estimators_ = existing_stages
            remaining = self.n_estimators - len(existing_stages)
            if remaining > 0:
                orig_n_classes = self.n_classes_
                self.n_estimators = remaining
                super().fit(X, y)
                self.n_classes_ = orig_n_classes
                self.estimators_.extend(existing_stages)
                self.n_estimators = len(self.estimators_)
        for i, stage in enumerate(self.estimators_):
            cp_path = os.path.join(self._checkpoint_dir, f"gradient_boosting_checkpoint_{i+1}.pkl")
            if not os.path.exists(cp_path):
                try:
                    with open(cp_path, 'wb') as f:
                        dill.dump(stage, f)
                    logging.info(f"[GradientBoosting] Сохранен чекпоинт {i+1}")
                except Exception as e:
                    logging.warning(f"[GradientBoosting] Не удалось сохранить чекпоинт {i+1}: {e}")
        return self

class CheckpointXGBoost(XGBClassifier):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, #n_estimators=100
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                 random_state=None, objective=None, **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective=objective,
            **kwargs
        )
        self._checkpoint_dir = get_checkpoint_path("xgboost", market_type)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # Исключаем наш приватный атрибут, чтобы не передавался в конструктор при клонировании
        if '_checkpoint_dir' in params:
            del params['_checkpoint_dir']
        return params

    def fit(self, X, y, **kwargs):
        logging.info("[XGBoost] Начало обучения с чекпоинтами")
        model_path = os.path.join(self._checkpoint_dir, "xgboost_checkpoint")
        final_checkpoint = os.path.join("/workspace/saved_models", "bearish_stacked_ensemble_model.pkl")

        
        # Если уже есть сохранённый чекпоинт, пытаемся его загрузить
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if hasattr(saved_model, 'n_features_') and saved_model.n_features_ == X.shape[1]:
                    logging.info("[XGBoost] Загружена модель из чекпоинта")
                    return saved_model
            except Exception as e:
                logging.warning(f"[XGBoost] Не удалось загрузить чекпоинт: {e}")
        
        # Очищаем директорию чекпоинтов и создаём её заново
        if os.path.exists(self._checkpoint_dir):
            shutil.rmtree(self._checkpoint_dir, onerror=_on_rm_error)
        ensure_directory(self._checkpoint_dir)
        
        # Пытаемся вызвать super().fit() с переданными параметрами.
        # Если возникает ошибка из-за неподдерживаемого early_stopping_rounds, удаляем его и пробуем снова.
        try:
            super().fit(X, y, **kwargs)
        except TypeError as e:
            if 'early_stopping_rounds' in kwargs:
                logging.warning("[XGBoost] Параметр early_stopping_rounds не поддерживается, пробуем без него")
                kwargs.pop('early_stopping_rounds')
                super().fit(X, y, **kwargs)
            else:
                raise e
        
        # Сохраняем модель в чекпоинт (для возможности возобновления обучения)
        joblib.dump(self, final_checkpoint)
        logging.info("[XGBoost] Сохранен новый чекпоинт")
        return self



class CheckpointLightGBM(LGBMClassifier):
    def __init__(self, n_estimators=100, num_leaves=31, learning_rate=0.1, #n_estimators=100
                 min_data_in_leaf=20, max_depth=-1, random_state=None, **kwargs):
        self._checkpoint_dir = kwargs.pop('_checkpoint_dir', get_checkpoint_path("lightgbm", market_type))
        super().__init__(n_estimators=n_estimators, num_leaves=num_leaves,
                         learning_rate=learning_rate, min_data_in_leaf=min_data_in_leaf,
                         max_depth=max_depth, random_state=random_state, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_checkpoint_dir', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._checkpoint_dir = get_checkpoint_path("lightgbm", market_type)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.pop('_checkpoint_dir', None)
        return params

    def fit(self, X, y, **kwargs):
        logging.info("[LightGBM] Начало обучения с чекпоинтами")
        model_path = os.path.join(self._checkpoint_dir, "lightgbm_checkpoint")
        final_checkpoint = f"{model_path}_final.pkl"
        if os.path.exists(final_checkpoint):
            try:
                with open(final_checkpoint, 'rb') as f:
                    saved_model = dill.load(f)
                if hasattr(saved_model, '_n_features') and saved_model._n_features == X.shape[1]:
                    logging.info("[LightGBM] Загружена модель из чекпоинта")
                    self.__dict__.update(saved_model.__dict__)
                    _ = self.predict(X[:1])
                    return self
            except Exception as e:
                logging.warning(f"[LightGBM] Не удалось загрузить чекпоинт: {e}")
        if os.path.exists(self._checkpoint_dir):
            shutil.rmtree(self._checkpoint_dir, onerror=_on_rm_error)
        ensure_directory(self._checkpoint_dir)
        super().fit(X, y, **kwargs)
        self._n_features = X.shape[1]
        with open(final_checkpoint, 'wb') as f:
            dill.dump(self, f)
        logging.info("[LightGBM] Сохранен новый чекпоинт")
        return self

# Глобальный флаг для отключения сохранения чекпоинтов (True – сохранять не будем)
SKIP_CHECKPOINT = False

class CheckpointCatBoost(CatBoostClassifier):
    def __init__(self, iterations=1000, depth=6, learning_rate=0.1, #iterations=1000
                 random_state=None, **kwargs):
        # Если параметр 'save_snapshot' передан, удаляем его, чтобы избежать конфликтов
        if 'save_snapshot' in kwargs:
            del kwargs['save_snapshot']
        super().__init__(iterations=iterations,
                         depth=depth,
                         learning_rate=learning_rate,
                         random_state=random_state,
                         save_snapshot=False,
                         **kwargs)
        # Сохраняем путь к контрольным точкам в приватном атрибуте
        self._checkpoint_dir = get_checkpoint_path("catboost", market_type)

    def get_params(self, deep=True):
        """
        Переопределяем get_params, чтобы исключить _checkpoint_dir из параметров,
        передаваемых в конструктор при клонировании (например, GridSearchCV).
        """
        params = super().get_params(deep=deep)
        if '_checkpoint_dir' in params:
            del params['_checkpoint_dir']
        return params

    def fit(self, X, y, **kwargs):
        logging.info("[CatBoost] Начало обучения с чекпоинтами")
        model_path = os.path.join(self._checkpoint_dir, "catboost_checkpoint")
        final_checkpoint = f"{model_path}_final.joblib"
        
        # Если чекпоинт уже существует, пытаемся его загрузить
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if hasattr(saved_model, 'feature_count_') and saved_model.feature_count_ == X.shape[1]:
                    logging.info("[CatBoost] Загружена модель из чекпоинта")
                    return saved_model
            except Exception as e:
                logging.warning(f"[CatBoost] Не удалось загрузить существующий чекпоинт: {e}")
        
        # Удаляем директорию чекпоинтов, если она существует, и создаем заново
        if os.path.exists(self._checkpoint_dir):
            shutil.rmtree(self._checkpoint_dir, onerror=_on_rm_error)
        ensure_directory(self._checkpoint_dir)
        
        # Удаляем параметр save_snapshot из kwargs, если он присутствует
        if 'save_snapshot' in kwargs:
            del kwargs['save_snapshot']
        
        super().fit(X, y, **kwargs)
        
        # Если мы не в режиме GridSearch (где SKIP_CHECKPOINT=True), сохраняем чекпоинт
        if not SKIP_CHECKPOINT:
            joblib.dump(self, final_checkpoint)
            logging.info("[CatBoost] Сохранен финальный чекпоинт")
        else:
            logging.info("[CatBoost] Пропуск сохранения чекпоинта (режим GridSearch)")
        
        return self


class CheckpointRandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None, #n_estimators=100
                 min_samples_split=2, min_samples_leaf=1, random_state=None, **kwargs):
        self._checkpoint_dir = kwargs.pop('_checkpoint_dir', get_checkpoint_path("random_forest", market_type))
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, random_state=random_state, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_checkpoint_dir', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._checkpoint_dir = get_checkpoint_path("random_forest", market_type)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.pop('_checkpoint_dir', None)
        return params

    def fit(self, X, y):
        logging.info("[RandomForest] Начало обучения с чекпоинтами")
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]
        existing_trees = []
        for i in range(self.n_estimators):
            cp_path = os.path.join(self._checkpoint_dir, f"random_forest_tree_{i+1}.pkl")
            if os.path.exists(cp_path):
                try:
                    with open(cp_path, 'rb') as f:
                        tree = dill.load(f)
                    if hasattr(tree, 'tree_') and tree.tree_.n_features == n_features:
                        existing_trees.append(tree)
                        logging.info(f"[RandomForest] Загружено дерево {i+1} из чекпоинта")
                except Exception as e:
                    logging.warning(f"[RandomForest] Не удалось загрузить чекпоинт {i+1}: {e}")
        if not existing_trees:
            if os.path.exists(self._checkpoint_dir):
                shutil.rmtree(self._checkpoint_dir, onerror=_on_rm_error)
            ensure_directory(self._checkpoint_dir)
            logging.info("[RandomForest] Начинаем обучение с нуля")
            super().fit(X, y)
        else:
            self.estimators_ = existing_trees
            remaining = self.n_estimators - len(existing_trees)
            if remaining > 0:
                logging.info(f"[RandomForest] Продолжаем обучение: осталось {remaining} деревьев")
                orig_n_classes = self.n_classes_
                self.n_estimators = remaining
                super().fit(X, y)
                self.n_classes_ = orig_n_classes
                self.estimators_.extend(existing_trees)
                self.n_estimators = len(self.estimators_)
        for i, estimator in enumerate(self.estimators_):
            cp_path = os.path.join(self._checkpoint_dir, f"random_forest_tree_{i+1}.pkl")
            if not os.path.exists(cp_path):
                try:
                    with open(cp_path, 'wb') as f:
                        dill.dump(estimator, f)
                    logging.info(f"[RandomForest] Создан чекпоинт для дерева {i+1}")
                except Exception as e:
                    logging.warning(f"[RandomForest] Не удалось сохранить чекпоинт для дерева {i+1}: {e}")
        if not hasattr(self, 'n_outputs_'):
            self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        return self
    
        
def save_ensemble_checkpoint(ensemble_model, checkpoint_path):
    """Сохраняет общий чекпоинт ансамбля."""
    ensure_directory(os.path.dirname(checkpoint_path))
    joblib.dump(ensemble_model, checkpoint_path)
    logging.info(f"[Ensemble] Сохранен чекпоинт ансамбля: {checkpoint_path}")



def load_ensemble_checkpoint(checkpoint_path):
    """Загружает общий чекпоинт ансамбля."""
    if os.path.exists(checkpoint_path):
        logging.info(f"[Ensemble] Загрузка чекпоинта ансамбля: {checkpoint_path}")
        return joblib.load(checkpoint_path)
    logging.info(f"[Ensemble] Чекпоинт не найден: {checkpoint_path}")
    return None


# Получение исторических данных
def get_historical_data(symbols, bearish_periods, interval="1m", save_path="/workspace/data/binance_data_bearish.csv"):
    base_url_monthly = "https://data.binance.vision/data/spot/monthly/klines"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    all_data = []
    downloaded_files = set()
    download_lock = Lock()  # Для синхронизации многопоточного доступа

    def download_and_process(symbol, period):
        start_date = datetime.strptime(period["start"], "%Y-%m-%d")
        end_date = datetime.strptime(period["end"], "%Y-%m-%d")
        temp_data = []

        for current_date in pd.date_range(start=start_date, end=end_date, freq='MS'):
            year = current_date.year
            month = current_date.month
            month_str = f"{month:02d}"
            file_name = f"{symbol}-{interval}-{year}-{month_str}.zip"
            file_url = f"{base_url_monthly}/{symbol}/{interval}/{file_name}"

            with download_lock:
                if file_name in downloaded_files:
                    logging.info(f"⏩ Пропуск скачивания {file_name}, уже загружено.")
                    continue

            # Повторяем попытки загрузки (до 3-х)
            success = False
            for attempt in range(3):
                try:
                    logging.info(f"🔍 Проверка наличия файла: {file_url} (попытка {attempt+1})")
                    head_resp = requests.head(file_url, timeout=10)
                    if head_resp.status_code != 200:
                        logging.warning(f"⚠ Файл не найден: {file_url}")
                        break
                    logging.info(f"📥 Скачивание {file_url} (попытка {attempt+1})...")
                    response = requests.get(file_url, timeout=15)
                    if response.status_code != 200:
                        logging.warning(f"⚠ Ошибка загрузки {file_url}: Код {response.status_code}")
                        continue
                    # Если все прошло успешно, фиксируем файл в кэше и выходим из цикла
                    with download_lock:
                        downloaded_files.add(file_name)
                    success = True
                    break
                except Exception as e:
                    logging.warning(f"Попытка {attempt+1} для {file_url} не удалась: {e}")
                    time.sleep(2)  # небольшая задержка между попытками
            if not success:
                logging.error(f"❌ Не удалось скачать {file_url} после 3 попыток.")
                continue

            # Обработка скачанного zip-файла
            try:
                zip_file = zipfile.ZipFile(BytesIO(response.content))
                csv_file = file_name.replace('.zip', '.csv')
                with zip_file.open(csv_file) as file:
                    df = pd.read_csv(
                        file, header=None, 
                        names=["timestamp", "open", "high", "low", "close", "volume",
                               "close_time", "quote_asset_volume", "number_of_trades",
                               "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"],
                        dtype={
                            "timestamp": "int64",
                            "open": "float32",
                            "high": "float32",
                            "low": "float32",
                            "close": "float32",
                            "volume": "float32",
                            "quote_asset_volume": "float32",
                            "number_of_trades": "int32",
                            "taker_buy_base_asset_volume": "float32",
                            "taker_buy_quote_asset_volume": "float32"
                        }
                    )
                    if "timestamp" not in df.columns:
                        logging.error(f"❌ Ошибка: Колонка 'timestamp' отсутствует в df для {symbol}")
                        continue
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                    df.set_index("timestamp", inplace=True)
                    df["symbol"] = symbol
                    temp_data.append(df)
            except Exception as e:
                logging.error(f"❌ Ошибка обработки {symbol} за {current_date.strftime('%Y-%m')}: {e}")
            time.sleep(0.5)
        return pd.concat(temp_data) if temp_data else None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_and_process, symbol, period)
                   for symbol in symbols for period in bearish_periods]
        for future in futures:
            result = future.result()
            if result is not None:
                all_data.append(result)
    
    if not all_data:
        logging.error("❌ Не удалось загрузить ни одного месяца данных.")
        return None
    
    df = pd.concat(all_data, ignore_index=False)
    logging.info(f"📊 Колонки в загруженном df: {df.columns}")
    
    if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        logging.error(f"❌ Колонка 'timestamp' отсутствует. Доступные колонки: {df.columns}")
        return None

    df = df.resample('1min').ffill()
    num_nans = df.isna().sum().sum()
    if num_nans > 0:
        nan_percentage = num_nans / len(df)
        if nan_percentage > 0.05:
            logging.warning(f"⚠ Пропущено {nan_percentage:.2%} данных! Удаляем строки с NaN.")
            df.dropna(inplace=True)
        else:
            logging.info(f"🔧 Заполняем {nan_percentage:.2%} пропущенных данных методом ffill.")
            df.fillna(method='ffill', inplace=True)
    df.to_csv(save_path)
    logging.info(f"💾 Данные сохранены в {save_path}")
    return save_path


def load_bearish_data(symbols, bearish_periods, interval="1m", save_path="binance_data_bearish.csv"):
    """
    Загружает данные для заданных символов и периодов.
    Если файл save_path уже существует, новые данные объединяются с уже сохранёнными.
    Возвращает словарь, где для каждого символа содержится DataFrame с объединёнными данными.
    Чтение CSV выполняется по чанкам для снижения нагрузки на память.
    """
    CHUNK_SIZE = 200000  # размер чанка для чтения CSV

    # Если файл уже существует – читаем существующие данные по чанкам
    if os.path.exists(save_path):
        try:
            chunks = []
            for chunk in pd.read_csv(
                temp_file_path,
                index_col='timestamp',
                parse_dates=['timestamp'],
                on_bad_lines='skip',
                chunksize=CHUNK_SIZE,
                low_memory=False,
                dtype={
                    "open": float, "high": float, "low": float,
                    "close": float, "volume": float, "quote_asset_volume": float,
                    "number_of_trades": int, "taker_buy_base_asset_volume": float,
                    "taker_buy_quote_asset_volume": float
                }):
                # Сброс индекса, чтобы гарантировать наличие столбца с датами
                chunk = chunk.reset_index(drop=False)
                if 'timestamp' not in chunk.columns:
                    if 'index' in chunk.columns:
                        chunk.rename(columns={'index': 'timestamp'}, inplace=True)
                        logging.info("Столбец 'index' переименован в 'timestamp'.")
                    else:
                        raise ValueError("Отсутствует столбец с временными метками.")
                # Преобразуем столбец 'timestamp' в datetime с utc=True
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', utc=True)
                # Удаляем строки с нераспознанными датами
                chunk = chunk.dropna(subset=['timestamp'])
                # Устанавливаем 'timestamp' как индекс
                chunk = chunk.set_index('timestamp')
                chunks.append(chunk)
            existing_data = pd.concat(chunks, ignore_index=False)
            logging.info(f"Считаны существующие данные из {save_path}, строк: {len(existing_data)}")
        except Exception as e:
            logging.error(f"Ошибка при чтении существующего файла {save_path}: {e}")
            existing_data = pd.DataFrame()
    else:
        existing_data = pd.DataFrame()

    all_data = {}  # Словарь для хранения данных по каждому символу
    logging.info(f"🚀 Начало загрузки данных для символов: {symbols}")

    # Загрузка данных параллельно для каждого символа
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(get_historical_data, [symbol], bearish_periods, interval, save_path): symbol
            for symbol in symbols
        }
        for future in futures:
            symbol = futures[future]
            try:
                temp_file_path = future.result()
                if temp_file_path is not None:
                    # Читаем скачанный файл по чанкам
                    chunks = []
                    for chunk in pd.read_csv(temp_file_path,
                                             index_col='timestamp',
                                             parse_dates=['timestamp'],
                                             on_bad_lines='skip',
                                             chunksize=CHUNK_SIZE):
                        chunk = chunk.reset_index(drop=False)
                        if 'timestamp' not in chunk.columns:
                            if 'index' in chunk.columns:
                                chunk.rename(columns={'index': 'timestamp'}, inplace=True)
                                logging.info("Столбец 'index' переименован в 'timestamp' при чтении новых данных.")
                            else:
                                raise ValueError("Отсутствует столбец с временными метками в новых данных.")
                        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', utc=True)
                        chunk = chunk.dropna(subset=['timestamp'])
                        chunk = chunk.set_index('timestamp')
                        chunks.append(chunk)
                    if chunks:
                        new_data = pd.concat(chunks, ignore_index=False)
                    else:
                        new_data = pd.DataFrame()
                    
                    if symbol in all_data:
                        all_data[symbol].append(new_data)
                    else:
                        all_data[symbol] = [new_data]
                    logging.info(f"✅ Данные добавлены для {symbol}. Файлов: {len(all_data[symbol])}")
            except Exception as e:
                logging.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")

    # Объединяем данные для каждого символа
    for symbol in list(all_data.keys()):
        if all_data[symbol]:
            all_data[symbol] = pd.concat(all_data[symbol], ignore_index=False)
        else:
            del all_data[symbol]

    # Объединяем данные всех символов в один DataFrame
    if all_data:
        new_combined = pd.concat(all_data.values(), ignore_index=False)
    else:
        new_combined = pd.DataFrame()

    # Объединяем с уже существующими данными (если имеются)
    if not existing_data.empty:
        combined = pd.concat([existing_data, new_combined], ignore_index=False)
    else:
        combined = new_combined

    # --- Обработка временных меток ---
    # Сбросим индекс, чтобы гарантировать наличие столбца с датами
    combined = combined.reset_index(drop=False)
    if 'timestamp' not in combined.columns:
        if 'index' in combined.columns:
            combined.rename(columns={'index': 'timestamp'}, inplace=True)
            logging.info("Столбец 'index' переименован в 'timestamp' при окончательном объединении.")
        else:
            logging.error("Не найден столбец 'timestamp' или 'index' в итоговых данных!")
            raise ValueError("Отсутствует столбец с временными метками.")
    combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce', utc=True)
    combined = combined.dropna(subset=['timestamp'])
    combined = combined.set_index('timestamp')

    if not isinstance(combined.index, pd.DatetimeIndex):
        logging.error(f"После преобразования итоговый индекс имеет тип: {type(combined.index)}")
        raise ValueError("Индекс не является DatetimeIndex.")
    else:
        logging.info("Индекс успешно преобразован в DatetimeIndex.")

    # Сохраняем итоговый DataFrame с указанием имени колонки индекса
    combined.to_csv(save_path, index_label='timestamp')
    logging.info(f"💾 Обновлённые данные сохранены в {save_path} (итоговых строк: {len(combined)})")
    return all_data


    
'''def aggregate_to_2min(data):
    """
    Агрегация данных с интервала 1 минута до 2 минут.
    """
    logging.info("Агрегация данных с интервала 1 минута до 2 минут")

    # Проверка, установлен ли временной индекс
    if not isinstance(data.index, pd.DatetimeIndex):
        # Проверка и установка временного индекса
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')  # Преобразуем в datetime
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")

        # Убедитесь, что индекс является DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Индекс данных не является DatetimeIndex даже после преобразования.")

    # Агрегация данных
    data = data.resample('2T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logging.info(f"Агрегация завершена, размер данных: {len(data)} строк")
    logging.info(f"После агрегации данных на 2 минуты: NaN = {data.isna().sum().sum()}")
    return data'''


def triple_barrier_label(prices, T=5, window=50, k=1.0):
    """
    Вычисляет метки по тройному барьерному методу.
    
    Для каждого момента времени:
      - Рассчитывается волатильность доходностей за предыдущие `window` периодов (без утечки).
      - За горизонт T вычисляются будущие процентные изменения цены.
      - Если максимум будущих доходностей > (volatility * k)  => сигнал Buy (2).
      - Если минимум будущих доходностей < -(volatility * k) => сигнал Sell (1).
      - Иначе – сигнал Hold (0).
    
    Аргументы:
      prices - серия цен (pandas.Series).
      T - горизонт (количество периодов) для оценки будущего движения.
      window - окно для расчёта исторической волатильности.
      k - множитель для порогов.
    
    Возвращает:
      numpy.array меток той же длины, что и prices.
    """
    labels = np.zeros(len(prices), dtype=int)
    # Вычисляем доходности и историческую волатильность на основе прошлых данных
    returns = prices.pct_change().fillna(0)
    # Волатильность рассчитывается по предыдущим window периодам (без утечки: используем rolling до текущего момента)
    vol = returns.rolling(window=window, min_periods=1).std()
    
    # Для каждого момента времени рассчитываем будущее движение цены за горизонт T
    for i in range(len(prices) - T):
        # Будущие доходности относительно цены в момент i
        future_returns = prices.iloc[i+1:i+T+1].values / prices.iloc[i] - 1
        threshold = vol.iloc[i] * k
        if np.max(future_returns) > threshold:
            labels[i] = 2  # Buy
        elif np.min(future_returns) < -threshold:
            labels[i] = 1  # Sell
        else:
            labels[i] = 0  # Hold
    # Последние T записей не имеют полного горизонта – ставим Hold
    labels[-T:] = 0
    return labels


# Извлечение признаков
def extract_features(data):
    logging.info("Извлечение признаков для медвежьего рынка")
    data = data.copy()
    # Применяем фильтр Калмана, как и раньше
    data = remove_noise(data)

    # Базовые расчёты
    returns = data['close'].pct_change()
    data['returns'] = returns
    # Группируем rolling-вычисления для 'volume'
    volume_agg = data['volume'].rolling(10).agg(['mean', 'std'])
    data['volume_ma'] = volume_agg['mean']
    data['volume_ratio'] = data['volume'] / (volume_agg['mean'] + 1e-7)
    # Цена ускорения (diff от returns)
    price_acceleration = returns.diff()

    # MACD и связанные показатели – оставляем без изменений (они используют внешнюю библиотеку)
    macd = MACD(data['smoothed_close'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    data['macd_slope'] = data['macd_diff'].diff()
    
    # RSI с окном 5
    data['rsi_5'] = RSIIndicator(data['close'], window=5).rsi()
    
    # Bollinger Bands для определения положения цены
    bb = BollingerBands(data['smoothed_close'], window=20)
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()
    data['bb_width'] = bb.bollinger_wband()
    data['bb_position'] = (data['close'] - data['bb_low']) / ((data['bb_high'] - data['bb_low']) + 1e-7)

    # Пример динамических порогов (оставляем как есть)
    def calculate_dynamic_thresholds(window=10):
        vol = returns.rolling(window).std()
        avg_vol = vol.rolling(100).mean()
        vol_ratio = vol / (avg_vol + 1e-7)
        base_strong = -0.001
        base_medium = -0.0005
        strong_threshold = base_strong * np.where(vol_ratio > 1.5, 1.5, np.where(vol_ratio < 0.5, 0.5, vol_ratio))
        medium_threshold = base_medium * np.where(vol_ratio > 1.5, 1.5, np.where(vol_ratio < 0.5, 0.5, vol_ratio))
        return strong_threshold, medium_threshold

    strong_threshold, medium_threshold = calculate_dynamic_thresholds()

    # Изменённый блок для формирования целевой переменной (логика сохранена)
    data['target'] = np.where(
        (((returns.shift(-1) < -0.00005) | (price_acceleration < -0.00005)) & (data['macd_diff'] < 0)),
        2,
        np.where(
            (((returns.shift(-1) > 0.0001) | (data['rsi_5'] < 50)) & (data['bb_position'] < 0.6)),
            1,
            0
        )
    )

    # Дополнительные расчёты (объём, давление, волатильность, трендовые индикаторы и т.д.) – оставляем как есть.
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['selling_pressure'] = data['volume'] * np.abs(data['close'] - data['open']) * np.where(data['close'] < data['open'], 1, 0)
    data['buying_pressure'] = data['volume'] * np.abs(data['close'] - data['open']) * np.where(data['close'] > data['open'], 1, 0)
    data['pressure_ratio'] = data['selling_pressure'] / (data['buying_pressure'].replace(0, 1))
    data['volatility'] = returns.rolling(10).std()
    data['volatility_ma'] = data['volatility'].rolling(20).mean()
    data['volatility_ratio'] = data['volatility'] / (data['volatility_ma'] + 1e-7)

    # Трендовые индикаторы по разным периодам
    for period in [3, 5, 8, 13, 21]:
        data[f'sma_{period}'] = SMAIndicator(data['smoothed_close'], window=period).sma_indicator()
        data[f'ema_{period}'] = data['smoothed_close'].ewm(span=period, adjust=False).mean()

    # Объёмные индикаторы
    data['obv'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    # CHAID: guard against zero‐division in CMF
    try:
        data['cmf'] = ChaikinMoneyFlowIndicator(
            data['high'], data['low'], data['close'], data['volume']
        ).chaikin_money_flow()
    except ZeroDivisionError:
        logging.warning("Zero division in CMF; filling cmf=0")
        data['cmf'] = 0
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

    # Осцилляторы и уровни поддержки/сопротивления
    for period in [7, 14, 21]:
        data[f'rsi_{period}'] = RSIIndicator(data['close'], window=period).rsi()
    data['stoch_k'] = StochasticOscillator(data['high'], data['low'], data['close'], window=7).stoch()
    data['stoch_d'] = StochasticOscillator(data['high'], data['low'], data['close'], window=7).stoch_signal()
    data['support_level'] = data['low'].rolling(20).min()
    data['resistance_level'] = data['high'].rolling(20).max()
    data['price_to_support'] = data['close'] / data['support_level']

    # Свечной анализ
    data['candle_body'] = np.abs(data['close'] - data['open'])
    data['upper_shadow'] = data['high'] - np.maximum(data['close'], data['open'])
    data['lower_shadow'] = np.minimum(data['close'], data['open']) - data['low']
    data['body_to_shadow_ratio'] = data['candle_body'] / ((data['upper_shadow'] + data['lower_shadow']).replace(0, 0.001))

    # Ценовые уровни и прорывы
    data['price_level_breach'] = np.where(
        data['close'] < data['support_level'].shift(1), -1,
        np.where(data['close'] > data['resistance_level'].shift(1), 1, 0)
    )

    # Индикаторы скорости движения
    data['price_acceleration'] = returns.diff()
    data['volume_acceleration'] = data['volume_change'].diff()

    # Пересчёт Bollinger Bands (дополнительная проверка)
    bb2 = BollingerBands(data['smoothed_close'], window=20)
    data['bb_high'] = bb2.bollinger_hband()
    data['bb_low'] = bb2.bollinger_lband()
    data['bb_width'] = bb2.bollinger_wband()
    data['bb_position'] = (data['close'] - data['bb_low']) / ((data['bb_high'] - data['bb_low']) + 1e-7)

    # ATR индикаторы
    for period in [5, 10, 20]:
        data[f'atr_{period}'] = AverageTrueRange(data['high'], data['low'], data['close'], window=period).average_true_range()

    # Специальные признаки для HFT
    data['micro_trend'] = np.where(
        data['smoothed_close'] > data['smoothed_close'].shift(1), 1,
        np.where(data['smoothed_close'] < data['smoothed_close'].shift(1), -1, 0)
    )
    data['micro_trend_sum'] = data['micro_trend'].rolling(5).sum()
    data['volume_acceleration_5m'] = (data['volume'].diff() / data['volume'].rolling(5).mean()).fillna(0)

    # Если 'clean_returns' отсутствует, создаём его
    if 'clean_returns' not in data.columns:
        data['clean_returns'] = data['smoothed_close'].pct_change()

    # Признаки силы медвежьего движения
    data['bearish_strength'] = np.where(
        (data['close'] < data['open']) & 
        (data['volume'] > data['volume'].rolling(20).mean() * 1.5) & 
        (data['close'] == data['low']) & 
        (data['clean_returns'] < 0),
        3,
        np.where(
            (data['close'] < data['open']) &
            (data['volume'] > data['volume'].rolling(20).mean()) &
            (data['clean_returns'] < 0),
            2,
            np.where(data['close'] < data['open'], 1, 0)
        )
    )


    # Формирование словаря признаков
    features = {}
    features['target'] = data['target']

    for col in data.columns:
        if col not in ['market_type']:
            features[col] = data[col]

    # Добавляем межмонетные признаки, если они имеются
    if 'btc_corr' in data.columns:
        features['btc_corr'] = data['btc_corr']
    if 'rel_strength_btc' in data.columns:
        features['rel_strength_btc'] = data['rel_strength_btc']
    if 'beta_btc' in data.columns:
        features['beta_btc'] = data['beta_btc']

    # Признаки подтверждения объёмом, если они имеются
    if 'volume_strength' in data.columns:
        features['volume_strength'] = data['volume_strength']
    if 'volume_accumulation' in data.columns:
        features['volume_accumulation'] = data['volume_accumulation']

    # Очищенные от шума признаки, если они имеются
    if 'clean_returns' in data.columns:
        features['clean_returns'] = data['clean_returns']

    features_df = pd.DataFrame(features)

    logging.info(f"Количество признаков: {len(features_df.columns)}")
    logging.info(f"Проверка на NaN: {features_df.isna().sum().sum()}")
    logging.info(f"Распределение целевой переменной:\n{features_df['target'].value_counts()}")
    logging.info(f"✅ Итоговые признаки: {list(data.columns)}")

    num_nans = data.isna().sum().sum()
    if num_nans > 0:
        logging.warning(f"⚠ Найдено {num_nans} пропущенных значений. Заполняем...")
        data.fillna(0, inplace=True)

    # Возвращаем DataFrame с обработанными признаками
    return data.replace([np.inf, -np.inf], np.nan).ffill().bfill()



def clean_data(X, y):
    """
    Очистка данных: удаление пропущенных значений и дубликатов, синхронизация индексов.
    """
    # Удаление строк с пропущенными значениями
    mask = X.notnull().all(axis=1)
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]  # Синхронизация индексов
    logging.debug(f"После удаления пропусков: X = {X_clean.shape}, y = {y_clean.shape}")

    # Удаление дубликатов в X
    duplicated_indices = X_clean.index.duplicated(keep='first')
    X_clean = X_clean.loc[~duplicated_indices]
    y_clean = y_clean.loc[~duplicated_indices]  # Синхронизация индексов
    logging.debug(f"После удаления дубликатов: X = {X_clean.shape}, y = {y_clean.shape}")

    # Проверка индексов
    if not X_clean.index.equals(y_clean.index):
        raise ValueError("Индексы X и y не совпадают после очистки данных")

    return X_clean, y_clean


# Удаление выбросов
def remove_outliers(data):
    numeric_data = data.select_dtypes(include=[np.number])  # Только числовые столбцы
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = data[~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]
    logging.debug(f"Индексы после удаления выбросов: {filtered_data.index}")
    return filtered_data


# Кластеризация для выделения рыночных сегментов
def add_clustering_feature(data):
    """
    Кластеризация для выделения рыночных сегментов с признаками, специфичными для медвежьего рынка
    """
    features_for_clustering = [
        # Базовые ценовые признаки
        'close', 'volume', 'returns', 'log_returns',
        
        # Объемные индикаторы, которые точно есть
        'volume_ratio', 'volume_ma', 'cmf', 'obv',
        
        # Осцилляторы, которые создаются для разных периодов
        'rsi_7', 'rsi_14', 'rsi_21',  # RSI для разных периодов
        'stoch_k', 'stoch_d',
        
        # Трендовые индикаторы, созданные в extract_features
        'macd', 'macd_signal', 'macd_diff',
        'sma_3', 'sma_5', 'sma_8',    # Короткие SMA
        'ema_3', 'ema_5', 'ema_8',    # Короткие EMA
        
        # Волатильность
        'bb_width', 'atr_5', 'atr_10',
        
        # Специфичные признаки для медвежьего рынка
        'selling_pressure', 'pressure_ratio',
        'price_acceleration', 'volume_acceleration'
    ]

    # Проверка наличия признаков
    available_features = [f for f in features_for_clustering if f in data.columns]
    
    if len(available_features) < 5:  # Минимальное количество признаков для кластеризации
        raise ValueError("Недостаточно признаков для кластеризации")
    
     # Если данных много, берём сэмпл (например, 200k строк)
    max_for_kmeans = 200_000
    if len(data) > max_for_kmeans:
        sample_df = data.sample(n=max_for_kmeans, random_state=42)
    else:
        sample_df = data
        
    # Кластеризация
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(sample_df[features_for_clustering])
    data['cluster'] = kmeans.fit_predict(data[available_features])
    
    logging.info(f"Использовано признаков для кластеризации: {len(available_features)}")
    logging.info(f"Распределение кластеров:\n{data['cluster'].value_counts()}")
    
    return data


# Аугментация данных (добавление шума)
def augment_data(data):
    noise = np.random.normal(0, 0.01, data.shape)
    augmented_data = data + noise
    augmented_data.index = data.index  # Восстанавливаем исходные индексы
    logging.debug(f"Индексы после augment_data: {augmented_data.index}")
    return augmented_data

# Функции для SMOTETomek
def smote_process(X_chunk, y_chunk, chunk_id):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_chunk, y_chunk)
    return X_resampled, y_resampled

def parallel_smote(X, y, n_chunks=4):
    # Гарантируем, что исходный массив X является копией (изменяемой)
    X = np.copy(X)
    # Разбиваем на чанки и для каждого делаем копию
    X_chunks = [np.copy(chunk) for chunk in np.array_split(X, n_chunks)]
    y_chunks = np.array_split(y, n_chunks)
    results = Parallel(n_jobs=n_chunks)(
        delayed(smote_process)(X_chunk, y_chunk, idx)
        for idx, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks))
    )
    X_resampled = np.vstack([res[0] for res in results])
    y_resampled = np.hstack([res[1] for res in results])
    return X_resampled, y_resampled


def prepare_data(data):
    logging.info("Начало подготовки данных")

    # Если входные данные представлены словарём (по монетам), объединяем их в один DataFrame
    if isinstance(data, dict):
        if not data:
            raise ValueError("Входные данные пусты (пустой словарь)")
        data = pd.concat(data.values())
    
    if data.empty:
        raise ValueError("Входные данные пусты")
    logging.info(f"Исходная форма данных: {data.shape}")

    # Приводим индекс к DatetimeIndex (если это не так)
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            dt_index = pd.to_datetime(data.index, errors='coerce')
            if dt_index.isnull().all():
                raise ValueError("Невозможно преобразовать индекс в DatetimeIndex")
            data.index = dt_index
            # Если столбца 'timestamp' нет, добавляем его
            if 'timestamp' not in data.columns:
                data['timestamp'] = dt_index
            logging.info("Индекс успешно преобразован в DatetimeIndex и, при необходимости, добавлен как колонка 'timestamp'.")
        except Exception as e:
            raise ValueError("Данные не содержат временного индекса или колонки 'timestamp'.") from e

    # Если данные по нескольким монетам, их межмонетные признаки можно добавить до чанкования
    # (если изначально data был dict, он уже сконкатенирован)
    # Здесь можно вызвать calculate_cross_coin_features(data) при необходимости.

    # Применяем предобработку по чанкам:
    def process_chunk(df_chunk):
         df_chunk = detect_anomalies(df_chunk)
         df_chunk = validate_volume_confirmation(df_chunk)
         df_chunk = remove_noise(df_chunk)
         df_chunk = extract_features(df_chunk)
         return df_chunk

    data = apply_in_chunks(data, process_chunk, chunk_size=100000)
    logging.info(f"После извлечения признаков: {data.shape}")

    # Теперь отделяем признаки от целевой переменной
    drop_cols = ['symbol', 'close_time', 'ignore']
    X = data.drop(columns=drop_cols + ['target'], errors='ignore')
    y = data['target']
    
    # Оставляем только числовые признаки (убираем datetime, если он остался)
    X = X.select_dtypes(include=[np.number])
    
    X, y = clean_data(X, y)
    logging.info(f"После clean_data: X = {X.shape}, y = {y.shape}")
    
    # Применяем SMOTETomek – теперь X содержит только числовые данные
    X_resampled, y_resampled = parallel_smote(X, y, n_chunks=4)
    logging.info(f"После SMOTETomek: X = {X_resampled.shape}, y = {len(y_resampled)}")
    
    # Создаем DataFrame из пересэмплированных данных и добавляем колонку target
    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data['target'] = y_resampled
    
    resampled_data = remove_outliers(resampled_data)
    logging.info(f"После удаления выбросов: {resampled_data.shape}")
    
    resampled_data = add_clustering_feature(resampled_data)
    logging.info(f"После кластеризации: {resampled_data.shape}")
    
    # Формируем список признаков: берем только числовые столбцы, исключая 'target'
    numeric_cols = resampled_data.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col != 'target']
    logging.info(f"Количество признаков: {len(features)}")
    logging.info(f"Распределение target:\n{resampled_data['target'].value_counts()}")
    
    return resampled_data, features



def get_checkpoint_path(model_name, market_type):
    checkpoint_path = os.path.join("checkpoints", market_type, model_name)
    ensure_directory(checkpoint_path)
    return checkpoint_path

# Определение функции балансировки классов
def balance_classes(X, y):
    logging.info("Начало балансировки классов")
    logging.info(f"Размеры данных до балансировки: X={X.shape}, y={y.shape}")
    logging.info(f"Уникальные классы в y: {np.unique(y, return_counts=True)}")

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Данные для балансировки пусты. Проверьте исходные данные и фильтры.")
    
    max_for_smote = 300_000
    if len(X) > max_for_smote:
        X_sample = X.sample(n=max_for_smote, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y
    
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_sample, y_sample)

    logging.info(f"Размеры данных после балансировки: X={X_resampled.shape}, y={y_resampled.shape}")
    # Возвращаем результат как DataFrame и Series, если исходные X и y были такими:
    X_resampled = pd.DataFrame(X_resampled, columns=X_sample.columns, index=X_sample.index[:len(X_resampled)])
    y_resampled = pd.Series(y_resampled, index=X_resampled.index)
    return X_resampled, y_resampled



def check_class_balance(y):
    """Проверка баланса классов"""
    class_counts = pd.Series(y).value_counts()
    total = len(y)
    
    logging.info("Распределение классов:")
    for class_label, count in class_counts.items():
        percentage = (count / total) * 100
        logging.info(f"Класс {class_label}: {count} примеров ({percentage:.2f}%)")
    
    # Проверяем дисбаланс
    if class_counts.max() / class_counts.min() > 10:
        logging.warning("Сильный дисбаланс классов (>10:1)")
        

def check_feature_quality(X, y):
    """Проверка качества признаков"""
    # Проверяем дисперсию
    zero_var_features = []
    for col in X.columns:
        if X[col].std() == 0:
            zero_var_features.append(col)
    if zero_var_features:
        logging.warning(f"Признаки с нулевой дисперсией: {zero_var_features}")
    
    # Проверяем корреляции
    corr_matrix = X.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    if high_corr_pairs:
        logging.warning(f"Сильно коррелирующие признаки: {high_corr_pairs}")
    
    # Проверяем важность признаков
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    logging.info("Топ-10 важных признаков:")
    logging.info(feature_scores.head(10))



# Обучение модели
def train_model(model, X_train, y_train, name):
    logging.info(f"Начало обучения модели {name}")
    model.fit(X_train, y_train)
    return model

def load_progress(base_learners, meta_model, checkpoint_path):
    """
    Загружает прогресс для базовых моделей и мета-модели из контрольных точек.
    """
    for i, (name, model) in enumerate(base_learners):
        intermediate_path = f"{checkpoint_path}_{name}.pkl"
        if os.path.exists(intermediate_path):
            logging.info(f"Загрузка прогресса модели {name} из {intermediate_path}")
            base_learners[i] = (name, joblib.load(intermediate_path))
        else:
            logging.info(f"Контрольная точка для {name} не найдена. Начало обучения с нуля.")
    
    meta_model_path = f"{checkpoint_path}_meta.pkl"
    if not os.path.exists(os.path.dirname(meta_model_path)):
        os.makedirs(os.path.dirname(meta_model_path))
    if os.path.exists(meta_model_path):
        logging.info(f"Загрузка прогресса мета-модели из {meta_model_path}")
        meta_model = joblib.load(meta_model_path)
    else:
        logging.info("Контрольная точка для мета-модели не найдена. Начало обучения с нуля.")
    
    return base_learners, meta_model


# Обучение ансамбля моделей
def train_ensemble_model(data, selected_features, model_filename='models/bearish_stacked_ensemble_model.pkl'):
    """
    Обучает ансамбль моделей для медвежьего рынка (3 класса: hold=0, sell=1, buy=2).

    Шаги:
      1. Проверка и очистка входных данных.
      2. Преобразование индекса в DatetimeIndex и корректная обработка колонки 'timestamp'.
      3. Выбор целевой переменной и признаков.
      4. Балансировка классов, разделение на обучающую и тестовую выборки, масштабирование и аугментация.
      5. Балансировка через SMOTETomek и создание подвыборки для GridSearchCV.
      6. Подбор гиперпараметров для базовых моделей с использованием GridSearchCV.
      7. Обучение базовых моделей на полном датасете с сохранением чекпоинтов.
      8. Обучение мета-модели и финального стекинг-ансамбля.
      9. Сохранение итогового ансамбля и масштабировщика.

    :param data: DataFrame с исходными данными (должен содержать колонку 'target').
    :param selected_features: список признаков (не должен содержать 'target' и 'timestamp').
    :param model_filename: имя файла для сохранения итогового ансамбля.
    :return: словарь с обученным ансамблем, масштабировщиком и списком признаков.
    """
    
    logging.info("Начало обучения ансамбля моделей для медвежьего рынка (3 класса: hold/sell/buy)")
    
    # 1. Проверка входных данных
    if data.empty:
        raise ValueError("Входные данные пусты")
    if not isinstance(selected_features, list):
        raise TypeError("selected_features должен быть списком")
    if 'target' not in data.columns:
        raise KeyError("Отсутствует колонка 'target' во входных данных")
    # Удаляем случайно присутствующий 'timestamp' из списка признаков
    selected_features = [feat for feat in selected_features if feat != 'timestamp']
    
    logging.info(f"Входные данные shape: {data.shape}")
    logging.info(f"Входные колонки: {data.columns.tolist()}")
    logging.info(f"Распределение классов до обучения:\n{data['target'].value_counts()}")
    
    # 2. Преобразование индекса в DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
        if data.index.isnull().all():
            raise ValueError("Невозможно преобразовать индекс в DatetimeIndex.")
        logging.info("Индекс преобразован в DatetimeIndex.")
    
    # 3. Обработка колонки 'timestamp'
    if 'timestamp' in data.columns:
        logging.info("Колонка 'timestamp' уже присутствует, удаляем её для избежания дублирования.")
        data = data.drop(columns=['timestamp'])
    data['timestamp'] = data.index.copy()
    data.reset_index(drop=True, inplace=True)
    data = data.loc[:, ~data.columns.duplicated()]
    logging.info("Обработка столбца 'timestamp' завершена.")
    
    # 4. Выбор целевой переменной и признаков
    y = data['target'].copy()
    X = data[selected_features].copy()
    logging.info(f"Размер X до фильтрации: {X.shape}")
    logging.info(f"Размер y до фильтрации: {y.shape}")
    logging.info(f"Уникальные значения y: {np.unique(y, return_counts=True)}")
    
    X = X.astype(float)
    y = y.astype(int)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    if X.size == 0 or y.size == 0:
        logging.error("X или y пусты после удаления NaN. Проверьте обработку данных.")
        raise ValueError("X или y пусты после удаления NaN.")
    logging.info(f"Размер X после фильтрации: {X.shape}")
    logging.info(f"Размер y после фильтрации: {y.shape}")
    
    # 5. Балансировка классов
    try:
        X_resampled, y_resampled = balance_classes(X, y)
        logging.info(f"Размеры после балансировки: X_resampled={X_resampled.shape}, y_resampled={y_resampled.shape}")
        logging.info(f"Распределение классов после балансировки:\n{pd.Series(y_resampled).value_counts()}")
    except ValueError as e:
        logging.error(f"Ошибка при балансировке классов: {e}")
        raise
    
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    
    # 6. Разделение данных на обучающую и тестовую выборки (с стратификацией)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    logging.info(f"Train size: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Test size: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 7. Масштабирование и аугментация обучающих данных
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_train_aug = augment_data(X_train_scaled_df)
    logging.info(f"После аугментации: X_train_aug = {X_train_aug.shape}")
    
    # 8. Балансировка через SMOTETomek
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled2, y_resampled2 = smote_tomek.fit_resample(X_train_aug, y_train)
    logging.info(f"После SMOTETomek: X = {X_resampled2.shape}, y = {len(y_resampled2)}")
    logging.info(f"Распределение классов после SMOTETomek:\n{pd.Series(y_resampled2).value_counts()}")
    
    # 9. Создание подвыборки для GridSearchCV (20% данных)
    sample_size = int(0.2 * X_resampled2.shape[0])
    sample_indices = np.random.choice(np.arange(X_resampled2.shape[0]), size=sample_size, replace=False)
    if hasattr(X_resampled2, 'iloc'):
        X_sample = X_resampled2.iloc[sample_indices]
    else:
        X_sample = X_resampled2[sample_indices]
    y_sample = y_resampled2[sample_indices]
    logging.info(f"Подвыборка для GridSearchCV: X_sample = {X_sample.shape}, y_sample = {y_sample.shape}")
    
    perform_grid_search = True
    base_learners = []

    # 10. Инициализация базовых моделей с классами чекпоинтов
    rf_model = CheckpointRandomForest(n_estimators=200, max_depth=6, min_samples_leaf=5) #n_estimators=200
    gb_model = CheckpointGradientBoosting(n_estimators=150, max_depth=5, learning_rate=0.03, subsample=0.8) #n_estimators=150
    xgb_model = CheckpointXGBoost(n_estimators=150, max_depth=5, subsample=0.8, #n_estimators=150
                                  min_child_weight=3, learning_rate=0.03,
                                  objective='multi:softprob', num_class=3)
    lgbm_model = CheckpointLightGBM(n_estimators=150, num_leaves=32, learning_rate=0.03, #n_estimators=150
                                    min_data_in_leaf=5, random_state=42,
                                    objective="multiclass", num_class=3)
    catboost_model = CheckpointCatBoost(iterations=300, depth=6, learning_rate=0.03, #iterations=300
                                         min_data_in_leaf=5, random_state=42,
                                         loss_function='MultiClass')
    for name, model in [('rf', rf_model), ('gb', gb_model), ('xgb', xgb_model),
                        ('lgbm', lgbm_model), ('catboost', catboost_model)]:
        if perform_grid_search:
            if name == 'rf':
                param_grid = {'n_estimators': [150, 200], 'max_depth': [4, 6, 8], 'min_samples_leaf': [5, 8]} #'n_estimators': [150, 200],
            elif name == 'gb':
                param_grid = {'n_estimators': [150, 200], 'max_depth': [4, 5, 6], #'n_estimators': [150, 200],
                              'learning_rate': [0.03, 0.05], 'subsample': [0.8, 0.9]}
            elif name == 'xgb':
                param_grid = {'n_estimators': [150, 200], 'max_depth': [4, 5, 6], #'n_estimators': [150, 200],
                              'learning_rate': [0.03, 0.05], 'min_child_weight': [3, 5]}
            elif name == 'lgbm':
                param_grid = {'n_estimators': [150, 200], 'num_leaves': [32, 40], #'n_estimators': [150, 200],
                              'learning_rate': [0.03, 0.05], 'min_data_in_leaf': [5, 8]}
            elif name == 'catboost':
                param_grid = {'iterations': [300, 350], 'depth': [5, 6], #'iterations': [300, 350],
                              'learning_rate': [0.03, 0.05]}
            grid_search = GridSearchCV(model, param_grid, cv=2, scoring='f1_weighted', n_jobs=1)
            grid_search.fit(scaler.transform(X_sample), y_sample)
            logging.info(f"[GridSearch] Лучшие параметры для {name}: {grid_search.best_params_}")
            base_learners.append((name, grid_search.best_estimator_))
        else:
            base_learners.append((name, model))
    
    # 11. Обучение базовых моделей на полном датасете
    X_resampled_scaled = scaler.fit_transform(X_resampled2)
    for name, model in base_learners:
        checkpoint_path_model = get_checkpoint_path(name, "bearish")
        final_checkpoint = os.path.join(checkpoint_path_model, f"{name}_final.pkl")
        logging.info(f"[Ensemble] Обучение модели {name}")
        if isinstance(model, (CheckpointXGBoost, CheckpointLightGBM)):
            model.fit(X_resampled2, y_resampled2, eval_set=[(X_test, y_test)], early_stopping_rounds=20)
        else:
            model.fit(X_resampled_scaled, y_resampled2)
        try:
            with open(final_checkpoint, 'wb') as f:
                dill.dump(model, f, protocol=2)
            logging.info(f"[Ensemble] Модель {name} сохранена в {final_checkpoint}")
        except Exception as e:
            logging.warning(f"[Ensemble] Не удалось сохранить чекпоинт для {name}: {e}")
        if hasattr(model, "feature_importances_"):
            logging.info(f"[{name}] Feature importances: {model.feature_importances_}")
    
    # 12. Обучение мета-модели через GridSearchCV
    # Определяем базовую модель с ранней остановкой
    base_lr = LogisticRegression(
        penalty='l2',
        max_iter=5000,
        tol=1e-8,
        solver='saga',
        multi_class='multinomial',
        random_state=42
    )

    # Настраиваем перебор гиперпараметров через GridSearchCV, если необходимо
    param_grid = {'C': [0.01, 0.08, 0.1]}
    meta_model = GridSearchCV(base_lr, param_grid, cv=2, scoring='f1_weighted', n_jobs=1)
    meta_model.fit(X_resampled_scaled, y_resampled2)
    
    # 13. Удаление старых чекпоинтов базовых моделей
    for name, _ in base_learners:
        checkpoint_dir = get_checkpoint_path(name, "bearish")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            ensure_directory(checkpoint_dir)
    
    # 14. Обучение финального стекинг-ансамбля
    logging.info("[Ensemble] Обучение стекинг-ансамбля (3 класса: hold/sell/buy)")
    ensemble_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,
        passthrough=True,
        cv=5,
        n_jobs=1
    )
    ensemble_model.fit(X_resampled_scaled, y_resampled2)
    
    # 15. Оценка ансамбля на тестовой выборке
    y_pred = ensemble_model.predict(scaler.transform(X_test))
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    logging.info(f"F1-Score (weighted): {f1:.4f}")
    logging.info(f"Precision (weighted): {precision:.4f}")
    logging.info(f"Recall (weighted): {recall:.4f}")
    
    # 16. Сохранение итогового ансамбля
    save_data = {"ensemble_model": ensemble_model, "scaler": scaler}
    dir_path = os.path.dirname(model_filename)
    if dir_path:
        ensure_directory(dir_path)
    joblib.dump(save_data, model_filename)
    logging.info(f"[Ensemble] Ансамбль сохранён в {model_filename}")
    
    output_dir = os.path.join("/workspace/output", "bearish_ensemble")
    copy_output("Ensemble_Bearish", output_dir)
    
    return {"ensemble_model": ensemble_model, "scaler": scaler, "features": selected_features}



# Основной запуск
if __name__ == "__main__":

    strategy = initialize_strategy()
    
    symbols = ['BTCUSDC', 'ETHUSDC', 'BNBUSDC','XRPUSDC', 'ADAUSDC', 'SOLUSDC', 'DOTUSDC', 'LINKUSDC', 'TONUSDC', 'NEARUSDC']
        
    bearish_periods = [
            {"start": "2018-01-17", "end": "2018-03-31"},
            {"start": "2018-09-01", "end": "2018-12-31"},
            {"start": "2021-05-12", "end": "2021-08-31"},
            {"start": "2022-05-01", "end": "2022-07-31"},
            {"start": "2022-09-01", "end": "2022-12-15"},
            {"start": "2022-12-16", "end": "2023-01-31"}
        ]
    
    try:
        data_dict = load_bearish_data(symbols, bearish_periods, interval="1m", save_path="/workspace/data/binance_data_bearish.csv")
        logging.info("Данные успешно загружены")
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")
        
    data, selected_features = prepare_data(data_dict)
    logging.debug(f"Доступные столбцы после подготовки данных: {data.columns.tolist()}")
    logging.debug(f"Выбранные признаки: {selected_features}")
    
    ensemble_model, scaler, features = train_ensemble_model(data, selected_features, ensemble_model_filename)
    logging.info("Обучение ансамбля моделей завершено!")
    
    save_logs_to_file("Обучение ансамбля моделей завершено!")
    logging.info("Программа завершена успешно.")
    sys.exit(0)



