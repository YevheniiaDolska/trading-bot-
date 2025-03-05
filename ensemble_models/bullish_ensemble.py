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
import logging
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import joblib
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import sys
from filterpy.kalman import KalmanFilter
from sklearn.metrics import f1_score, precision_score, recall_score
import shutil
import requests
import zipfile
from io import BytesIO
import tensorflow as tf
from threading import Lock
import time
from sklearn.model_selection import GridSearchCV


# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug_log_bullish_ensemble.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # Вывод в консоль с поддержкой юникода
    ]
)


# Имя файла для сохранения модели
market_type = "bullish"

ensemble_model_filename = 'bullish_stacked_ensemble_model.pkl'

checkpoint_base_dir = f"checkpoints/{market_type}"

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


def ensure_directory(path):
    """Создает директорию, если она не существует."""
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def calculate_cross_coin_features(data_dict):
    """
    Рассчитывает межмонетные признаки для всех пар.
    
    Args:
        data_dict (dict): Словарь DataFrame'ов по каждой монете
    Returns:
        dict: Словарь DataFrame'ов с добавленными признаками
    """
    btc_data = data_dict['BTCUSDC']
    
    for symbol, df in data_dict.items():
        # Корреляции с BTC
        df['btc_corr'] = df['close'].rolling(30).corr(btc_data['close'])
        
        # Относительная сила к BTC
        df['rel_strength_btc'] = (df['close'].pct_change() - 
                                btc_data['close'].pct_change())
        
        # Бета к BTC
        df['beta_btc'] = (df['close'].pct_change().rolling(30).cov(
            btc_data['close'].pct_change()) / 
            btc_data['close'].pct_change().rolling(30).var())
        
        # Опережение/следование за BTC
        df['lead_lag_btc'] = df['close'].pct_change().shift(-1).rolling(10).corr(
            btc_data['close'].pct_change())
            
        data_dict[symbol] = df
        
    return data_dict

def detect_anomalies(data):
    """
    Детектирует и фильтрует аномальные свечи.
    """
    # Рассчитываем z-score для разных метрик
    data['volume_zscore'] = ((data['volume'] - data['volume'].rolling(100).mean()) / 
                            data['volume'].rolling(100).std())
    data['price_zscore'] = ((data['close'] - data['close'].rolling(100).mean()) / 
                           data['close'].rolling(100).std())
    data['range_zscore'] = (((data['high'] - data['low']) - 
                            (data['high'] - data['low']).rolling(100).mean()) / 
                           (data['high'] - data['low']).rolling(100).std())
    
    # Фильтруем экстремальные выбросы
    data['is_anomaly'] = ((abs(data['volume_zscore']) > 4) | 
                         (abs(data['price_zscore']) > 4) | 
                         (abs(data['range_zscore']) > 4))
    
    return data

def validate_volume_confirmation_bullish(data):
    """
    Добавляет признаки подтверждения движений объемом для бычьего рынка.
    """
    # Объемное подтверждение тренда - более чувствительное к росту
    data['volume_trend_conf'] = np.where(
        (data['close'] > data['close'].shift(1)) & 
        (data['volume'] > data['volume'].rolling(10).mean()),  # Уменьшенное окно для быстрой реакции
        2,  # Усиленный сигнал для роста
        np.where(
            (data['close'] < data['close'].shift(1)) & 
            (data['volume'] > data['volume'].rolling(20).mean()),
            -1,
            0
        )
    )
    
    # Сила объемного подтверждения - с акцентом на рост
    data['volume_strength'] = (data['volume'] / 
                             data['volume'].rolling(10).mean() *  # Меньшее окно
                             np.where(data['volume_trend_conf'] > 0, 1.5, 1.0) *  # Усиление растущих движений
                             data['volume_trend_conf'])
    
    # Накопление объема - с фокусом на последовательный рост
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(3).sum()  # Меньшее окно
    
    return data


def remove_noise(data):
    """
    Улучшенная фильтрация шума.
    """
    # Kalman filter для сглаживания цены
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    kf.R = 5
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])
    
    smoothed_prices = []
    for price in data['close']:
        kf.predict()
        kf.update(price)
        smoothed_prices.append(float(kf.x[0]))
    
    data['smoothed_close'] = smoothed_prices
    
    # Вычисляем "чистые" движения
    data['clean_returns'] = (data['smoothed_close'].pct_change() * 
                           (1 - data['is_anomaly']))
    
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

# GradientBoosting: сохранение после каждой итерации
class CheckpointGradientBoosting(GradientBoostingClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None, 
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                         random_state=random_state, subsample=subsample, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf)
        self.checkpoint_dir = get_checkpoint_path("gradient_boosting", market_type)
        
    def fit(self, X, y):
        logging.info("[GradientBoosting] Начало обучения с чекпоинтами")
        
        # Инициализация
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]
        
        # Проверяем существующие чекпоинты
        existing_stages = []
        for i in range(self.n_estimators):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"gradient_boosting_checkpoint_{i + 1}.joblib")
            if os.path.exists(checkpoint_path):
                try:
                    stage = joblib.load(checkpoint_path)
                    if stage.n_features_ == n_features:
                        existing_stages.append(stage)
                        logging.info(f"[GradientBoosting] Загружена итерация {i + 1} из чекпоинта")
                except:
                    logging.warning(f"[GradientBoosting] Не удалось загрузить чекпоинт {i + 1}")

        # Если чекпоинты не подходят, очищаем директорию
        if not existing_stages:
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            ensure_directory(self.checkpoint_dir)
            logging.info("[GradientBoosting] Начинаем обучение с нуля")
            super().fit(X, y)
        else:
            self.estimators_ = existing_stages
            remaining_stages = self.n_estimators - len(existing_stages)
            if remaining_stages > 0:
                orig_n_classes = self.n_classes_
                self.n_estimators = remaining_stages
                super().fit(X, y)
                self.n_classes_ = orig_n_classes
                self.estimators_.extend(self.estimators_)
                self.n_estimators = len(self.estimators_)

        # Сохраняем чекпоинты
        for i, stage in enumerate(self.estimators_):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"gradient_boosting_checkpoint_{i + 1}.joblib")
            if not os.path.exists(checkpoint_path):
                joblib.dump(stage, checkpoint_path)
                logging.info(f"[GradientBoosting] Сохранен чекпоинт {i + 1}")

        return self

# XGBoost: сохранение каждые 3 итерации
class CheckpointXGBoost(XGBClassifier):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
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
        self.checkpoint_dir = get_checkpoint_path("xgboost", market_type)

    def fit(self, X, y, **kwargs):
        logging.info("[XGBoost] Начало обучения с чекпоинтами")
        model_path = os.path.join(self.checkpoint_dir, "xgboost_checkpoint")
        
        # Проверяем существующий чекпоинт
        final_checkpoint = f"{model_path}_final.joblib"
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if saved_model.n_features_ == X.shape[1]:
                    logging.info("[XGBoost] Загружена модель из чекпоинта")
                    return saved_model
            except:
                logging.warning("[XGBoost] Не удалось загрузить существующий чекпоинт")
        
        # Если чекпоинт не подходит, очищаем и начинаем заново
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        ensure_directory(self.checkpoint_dir)
        
        super().fit(X, y)
        joblib.dump(self, final_checkpoint)
        logging.info("[XGBoost] Сохранен новый чекпоинт")
        return self


# LightGBM: сохранение каждые 3 итерации
class CheckpointLightGBM(LGBMClassifier):
    def __init__(self, n_estimators=100, num_leaves=31, learning_rate=0.1,
                 min_data_in_leaf=20, max_depth=-1, random_state=None, **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_data_in_leaf=min_data_in_leaf,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs  # Пробрасываем дополнительные параметры, например, objective, num_class и т.д.
        )
        self._checkpoint_path = get_checkpoint_path("lightgbm", market_type)


    def fit(self, X, y, **kwargs):
        logging.info("[LightGBM] Начало обучения с чекпоинтами")
        model_path = os.path.join(self._checkpoint_path, "lightgbm_checkpoint")
        final_checkpoint = f"{model_path}_final.joblib"
        
        # Проверяем существующий чекпоинт
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if hasattr(saved_model, '_n_features') and saved_model._n_features == X.shape[1]:
                    logging.info("[LightGBM] Загружена модель из чекпоинта")
                    # Копируем атрибуты из сохраненной модели
                    self.__dict__.update(saved_model.__dict__)
                    # Проверяем, что модель обучена
                    _ = self.predict(X[:1])
                    return self
            except:
                logging.warning("[LightGBM] Не удалось загрузить существующий чекпоинт")
        
        # Если чекпоинт не подходит, очищаем и начинаем заново
        if os.path.exists(self._checkpoint_path):
            shutil.rmtree(self._checkpoint_path)
        ensure_directory(self._checkpoint_path)
        
        # Обучаем модель
        super().fit(X, y, **kwargs)
        self._n_features = X.shape[1]  # Сохраняем количество признаков
        joblib.dump(self, final_checkpoint)
        logging.info("[LightGBM] Сохранен новый чекпоинт")
        return self
    
    
# CatBoost: сохранение каждые 3 итерации
class CheckpointCatBoost(CatBoostClassifier):
    def __init__(self, iterations=1000, depth=6, learning_rate=0.1,
                 random_state=None, **kwargs):
        # Удаляем save_snapshot из kwargs если он там есть
        if 'save_snapshot' in kwargs:
            del kwargs['save_snapshot']
            
        super().__init__(
            iterations=iterations, 
            depth=depth, 
            learning_rate=learning_rate, 
            random_state=random_state,
            save_snapshot=False,  # Устанавливаем save_snapshot только здесь
            **kwargs
        )
        self.checkpoint_dir = get_checkpoint_path("catboost", market_type)

    def fit(self, X, y, **kwargs):
        logging.info("[CatBoost] Начало обучения с чекпоинтами")
        model_path = os.path.join(self.checkpoint_dir, "catboost_checkpoint")
        final_checkpoint = f"{model_path}_final.joblib"
        
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if hasattr(saved_model, 'feature_count_') and saved_model.feature_count_ == X.shape[1]:
                    logging.info("[CatBoost] Загружена модель из чекпоинта")
                    return saved_model
            except:
                logging.warning("[CatBoost] Не удалось загрузить существующий чекпоинт")
        
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        ensure_directory(self.checkpoint_dir)
        
        # Удаляем save_snapshot из kwargs при вызове fit если он там есть
        if 'save_snapshot' in kwargs:
            del kwargs['save_snapshot']
        
        super().fit(X, y, **kwargs)
        joblib.dump(self, final_checkpoint)
        logging.info("[CatBoost] Сохранен финальный чекпоинт")
        
        return self
    
    
# RandomForest: сохранение после каждого дерева
class CheckpointRandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, 
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, random_state=random_state)
        self.checkpoint_dir = get_checkpoint_path("random_forest", market_type)

    def fit(self, X, y):
        logging.info("[RandomForest] Начало обучения с чекпоинтами")
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]
        
        # Загружаем сохранённые деревья
        existing_trees = []
        for i in range(self.n_estimators):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"random_forest_tree_{i + 1}.joblib")
            if os.path.exists(checkpoint_path):
                try:
                    tree = joblib.load(checkpoint_path)
                    if tree.tree_.n_features == n_features:
                        existing_trees.append(tree)
                        logging.info(f"[RandomForest] Загружено дерево {i + 1} из чекпоинта")
                except Exception as e:
                    logging.warning(f"[RandomForest] Не удалось загрузить чекпоинт {i + 1}, будет создано новое дерево")
        
        if not existing_trees:
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            ensure_directory(self.checkpoint_dir)
            logging.info("[RandomForest] Начинаем обучение с нуля")
            super().fit(X, y)
        else:
            self.estimators_ = existing_trees
            remaining_trees = self.n_estimators - len(existing_trees)
            if remaining_trees > 0:
                logging.info(f"[RandomForest] Продолжаем обучение: осталось {remaining_trees} деревьев")
                orig_n_classes = self.n_classes_
                self.n_estimators = remaining_trees
                super().fit(X, y)
                self.n_classes_ = orig_n_classes
                self.estimators_.extend(self.estimators_)
                self.n_estimators = len(self.estimators_)
        
        # Сохраняем чекпоинты для всех деревьев, которых еще нет
        for i, estimator in enumerate(self.estimators_):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"random_forest_tree_{i + 1}.joblib")
            if not os.path.exists(checkpoint_path):
                joblib.dump(estimator, checkpoint_path)
                logging.info(f"[RandomForest] Создан чекпоинт для нового дерева {i + 1}")
        
        # Явно устанавливаем n_outputs_, если он не был установлен (обычно делает родительский метод fit)
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

        
        
def debug_target_presence(data, stage_name):
    """
    Отслеживает наличие и состояние колонки target на каждом этапе обработки
    """
    print(f"\n=== Отладка {stage_name} ===")
    print(f"Shape данных: {data.shape}")
    print(f"Колонки: {data.columns.tolist()}")
    if 'target' in data.columns:
        print(f"Распределение target:\n{data['target'].value_counts()}")
        print(f"Первые 5 значений target:\n{data['target'].head()}")
    else:
        print("ВНИМАНИЕ: Колонка 'target' отсутствует!")
    print("=" * 50)
    

# Функция загрузки данных с использованием многопоточности
def load_all_data(symbols, start_date, end_date, interval):
    all_data = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(get_historical_data, symbol, interval, start_date, end_date): symbol for symbol in symbols}
        for future in futures:
            symbol = futures[future]
            try:
                symbol_data = future.result()
                if symbol_data is not None:
                    symbol_data['symbol'] = symbol
                    all_data.append(symbol_data)
            except Exception as e:
                logging.error(f"Ошибка при загрузке данных для {symbol}: {e}")
                save_logs_to_file(f"Ошибка при загрузке данных для {symbol}: {e}")

    if not all_data:
        logging.error("Не удалось получить данные ни для одного символа")
        save_logs_to_file("Не удалось получить данные ни для одного символа")
        raise ValueError("Не удалось получить данные ни для одного символа")

    data = pd.concat(all_data)
    logging.info(f"Всего загружено {len(data)} строк данных")
    save_logs_to_file(f"Всего загружено {len(data)} строк данных")
    return data

# Получение исторических данных

def get_historical_data(symbols, bullish_periods, interval="1m", save_path="binance_data_bullish.csv"):
    """
    Скачивает исторические данные с Binance (архив) и сохраняет в один CSV-файл.

    :param symbols: список торговых пар (пример: ['BTCUSDC', 'ETHUSDC'])
    :param bullish_periods: список словарей с периодами (пример: [{"start": "2019-01-01", "end": "2019-12-31"}])
    :param interval: временной интервал (по умолчанию "1m" - 1 минута)
    :param save_path: путь к файлу для сохранения CSV (по умолчанию 'binance_data_bullish.csv')
    """
    base_url_monthly = "https://data.binance.vision/data/spot/monthly/klines"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    all_data = []
    downloaded_files = set()
    download_lock = Lock()  # Используем threading.Lock

    def download_and_process(symbol, period):
        start_date = datetime.strptime(period["start"], "%Y-%m-%d")
        end_date = datetime.strptime(period["end"], "%Y-%m-%d")
        temp_data = []

        for current_date in pd.date_range(start=start_date, end=end_date, freq='MS'):  # MS = Monthly Start
            year = current_date.year
            month = current_date.month
            month_str = f"{month:02d}"

            file_name = f"{symbol}-{interval}-{year}-{month_str}.zip"
            file_url = f"{base_url_monthly}/{symbol}/{interval}/{file_name}"

            # Блокируем доступ к скачиванию и проверяем, был ли файл уже скачан
            with download_lock:
                if file_name in downloaded_files:
                    logging.info(f"⏩ Пропуск скачивания {file_name}, уже загружено.")
                    continue  # Пропускаем скачивание

                logging.info(f"🔍 Проверка наличия файла: {file_url}")
                response = requests.head(file_url, timeout=5)
                if response.status_code != 200:
                    logging.warning(f"⚠ Файл не найден: {file_url}")
                    continue

                logging.info(f"📥 Скачивание {file_url}...")
                response = requests.get(file_url, timeout=15)
                if response.status_code != 200:
                    logging.warning(f"⚠ Ошибка загрузки {file_url}: Код {response.status_code}")
                    continue

                logging.info(f"✅ Успешно загружен {file_name}")
                downloaded_files.add(file_name)  # Добавляем в кэш загруженных файлов

            try:
                zip_file = zipfile.ZipFile(BytesIO(response.content))
                csv_file = file_name.replace('.zip', '.csv')

                with zip_file.open(csv_file) as file:
                    df = pd.read_csv(file, header=None, names=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                    ], dtype={
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
                    })

                    # 🛠 Проверяем, загружен ли timestamp
                    if "timestamp" not in df.columns:
                        logging.error(f"❌ Ошибка: Колонка 'timestamp' отсутствует в df для {symbol}")
                        return None

                    # 🛠 Преобразуем timestamp в datetime и ставим индекс
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                    df.set_index("timestamp", inplace=True)
                    
                    df["symbol"] = symbol

                    temp_data.append(df)
            except Exception as e:
                logging.error(f"❌ Ошибка обработки {symbol} за {current_date.strftime('%Y-%m')}: {e}")

            time.sleep(0.5)  # Минимальная задержка между скачиваниями

        return pd.concat(temp_data) if temp_data else None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_and_process, symbol, period) for symbol in symbols for period in bullish_periods]
        for future in futures:
            result = future.result()
            if result is not None:
                all_data.append(result)

    if not all_data:
        logging.error("❌ Не удалось загрузить ни одного месяца данных.")
        return None

    df = pd.concat(all_data, ignore_index=False)  # Не используем ignore_index, чтобы сохранить timestamp  

    # Проверяем, какие колонки есть в DataFrame
    logging.info(f"📊 Колонки в загруженном df: {df.columns}")

    # Проверяем, установлен ли временной индекс
    if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        logging.error(f"❌ Колонка 'timestamp' отсутствует. Доступные колонки: {df.columns}")
        return None

    # Теперь можно применять resample
    df = df.resample('1min').ffill()  # Минутные интервалы, заполняем пропущенные значения

    # Проверяем NaN
    num_nans = df.isna().sum().sum()
    if num_nans > 0:
        nan_percentage = num_nans / len(df)
        if nan_percentage > 0.05:  # Если более 5% данных пропущены
            logging.warning(f"⚠ Пропущено {nan_percentage:.2%} данных! Удаляем пропущенные строки.")
            df.dropna(inplace=True)
        else:
            logging.info(f"🔧 Заполняем {nan_percentage:.2%} пропущенных данных ffill.")
            df.fillna(method='ffill', inplace=True)  # Заполняем предыдущими значениями

    df.to_csv(save_path)
    logging.info(f"💾 Данные сохранены в {save_path}")

    return save_path


def load_bullish_data(symbols, bullish_periods, interval="1m", save_path="binance_data_bullish.csv"):
    """
    Загружает данные для флэтового рынка для заданных символов и периодов.
    Если файл save_path уже существует, новые данные объединяются с уже сохранёнными.
    Возвращает словарь, где для каждого символа содержится DataFrame с объединёнными данными.
    """
    # Если файл уже существует – читаем существующие данные
    if os.path.exists(save_path):
        try:
            existing_data = pd.read_csv(save_path, index_col=0, parse_dates=True, on_bad_lines='skip')
            logging.info(f"Считаны существующие данные из {save_path}, строк: {len(existing_data)}")
        except Exception as e:
            logging.error(f"Ошибка при чтении существующего файла {save_path}: {e}")
            existing_data = pd.DataFrame()
    else:
        existing_data = pd.DataFrame()

    all_data = {}  # Словарь для хранения данных по каждому символу
    logging.info(f"🚀 Начало загрузки данных за заданные периоды для символов: {symbols}")

    # Запускаем загрузку данных параллельно для каждого символа
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Передаём в get_historical_data параметр save_path, чтобы все загрузки записывались в один файл
        futures = {
            executor.submit(get_historical_data, [symbol], bullish_periods, interval, save_path): symbol
            for symbol in symbols
        }
        for future in futures:
            symbol = futures[future]
            try:
                # get_historical_data возвращает путь к файлу с загруженными данными
                temp_file_path = future.result()
                if temp_file_path is not None:
                    # Используем on_bad_lines='skip', чтобы пропустить проблемные строки
                    new_data = pd.read_csv(temp_file_path, index_col=0, parse_dates=True, on_bad_lines='skip')
                    if symbol in all_data:
                        all_data[symbol].append(new_data)
                    else:
                        all_data[symbol] = [new_data]
                    logging.info(f"✅ Данные добавлены для {symbol}. Текущий список: {len(all_data[symbol])} файлов.")
            except Exception as e:
                logging.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")

    # Объединяем данные для каждого символа, если список не пустой
    for symbol in list(all_data.keys()):
        if all_data[symbol]:
            all_data[symbol] = pd.concat(all_data[symbol])
        else:
            del all_data[symbol]

    # Объединяем данные всех символов в один DataFrame
    if all_data:
        new_combined = pd.concat(all_data.values(), ignore_index=False)
    else:
        new_combined = pd.DataFrame()

    # Объединяем с уже существующими данными (если таковые имеются)
    if not existing_data.empty:
        combined = pd.concat([existing_data, new_combined], ignore_index=False)
    else:
        combined = new_combined

    # Сохраняем итоговый объединённый DataFrame в единый CSV-файл
    combined.to_csv(save_path)
    logging.info(f"💾 Обновлённые данные сохранены в {save_path} (итоговых строк: {len(combined)})")

    # Возвращаем словарь с данными по каждому символу (обновлёнными только новыми данными)
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


def diagnose_nan(data, stage):
    """Проверка на наличие пропущенных значений и запись в лог."""
    if data.isnull().any().any():
        logging.warning(f"Пропущенные значения обнаружены на этапе: {stage}")
        nan_summary = data.isnull().sum()
        logging.warning(f"Суммарно NaN:\n{nan_summary}")
    else:
        logging.info(f"На этапе {stage} пропущенные значения отсутствуют.")
        

def log_class_distribution(y, stage):
    """Запись распределения классов в лог."""
    if y.empty:
        logging.warning(f"Целевая переменная пуста на этапе {stage}.")
    else:
        class_distribution = y.value_counts()
        logging.info(f"Распределение классов на этапе {stage}:\n{class_distribution}")


# Извлечение признаков
def extract_features(data):
    logging.info("Извлечение признаков для бычьего рынка")
    data = data.copy()
    
    # 1. Базовые метрики доходности и импульса
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['momentum_1m'] = data['close'].diff(1)
    data['momentum_3m'] = data['close'].diff(3)
    data['acceleration'] = data['momentum_1m'].diff()
    
    # 2. Расчет динамических порогов на основе волатильности
    volatility = data['returns'].rolling(20).std()
    volume_volatility = data['volume'].pct_change().rolling(20).std()
    
    # Базовые пороги для бычьего рынка
    base_strong = 0.001  # 0.1%
    base_medium = 0.0005  # 0.05%
    
    # Корректировка порогов
    strong_threshold = base_strong * (1 + volatility/volatility.mean())
    medium_threshold = base_medium * (1 + volatility/volatility.mean())
    volume_factor = 1 + (volume_volatility/volume_volatility.mean())
    
    # 3. Объемные показатели для определения силы движения
    data['volume_delta'] = data['volume'].diff()
    data['volume_momentum'] = data['volume'].diff().rolling(3).sum()
    volume_ratio = data['volume'] / data['volume'].rolling(5).mean() * volume_factor
    
    # 4. Быстрые трендовые индикаторы для HFT
    data['sma_2'] = SMAIndicator(data['close'], window=2).sma_indicator()
    data['sma_3'] = SMAIndicator(data['close'], window=3).sma_indicator()
    data['sma_10'] = SMAIndicator(data['close'], window=10).sma_indicator()
    data['ema_3'] = data['close'].ewm(span=3, adjust=False).mean()
    data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
    
    # 5. Микро-тренды для HFT
    data['micro_trend'] = np.where(
        data['close'] > data['close'].shift(1), 1,
        np.where(data['close'] < data['close'].shift(1), -1, 0)
    )
    data['micro_trend_strength'] = data['micro_trend'].rolling(3).sum()
    
    # 6. Быстрый MACD для 1-минутных свечей
    macd = MACD(data['close'], window_slow=12, window_fast=6, window_sign=3)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    data['macd_acceleration'] = data['macd_diff'].diff()
    
    # 7. Короткие осцилляторы
    data['rsi_5'] = RSIIndicator(data['close'], window=5).rsi()
    data['rsi_3'] = RSIIndicator(data['close'], window=3).rsi()
    stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=5)
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()
    
    # 8. Волатильность для оценки рисков
    bb = BollingerBands(data['close'], window=10)
    data['bb_width'] = bb.bollinger_wband()
    data['atr_3'] = AverageTrueRange(data['high'], data['low'], data['close'], window=3).average_true_range()
    
    # 9. Свечной анализ
    data['candle_body'] = data['close'] - data['open']
    data['body_ratio'] = abs(data['candle_body']) / (data['high'] - data['low'])
    data['upper_wick'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_wick'] = np.minimum(data['open'], data['close']) - data['low']
    
    # 10. Объемные индикаторы
    data['obv'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    data['volume_trend'] = data['volume'].diff() / data['volume'].shift(1)
    
    # 11. Индикаторы ускорения для HFT
    data['price_acceleration'] = data['returns'].diff()
    data['volume_acceleration'] = data['volume_delta'].diff()
    
    # 12. Многоуровневая целевая переменная для бычьего рынка
    data['target'] = np.where(
        # Сильный сигнал: разворот вверх после коррекции
        (data['returns'].shift(-1) > 0.001) & 
        (data['close'] < data['sma_10']) &  # Цена ниже 10-периодной SMA
        (volume_ratio > 1.2) & 
        (data['rsi_5'] < 40),  # Перепроданность на RSI
        2,
        np.where(
            # Умеренный сигнал: локальный отскок
            (data['returns'].shift(-1) > 0.0005) & 
            (data['micro_trend_strength'] < 0) &  # Предшествующее падение
            (data['volume_trend_conf'] > 0),  # Подтверждение объемом
            1,
            0
        )
    )
    
    return data.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def clean_data(X, y):
    """
    Очистка данных от пропущенных значений и дубликатов
    """
    logging.debug(f"Начало clean_data: X = {X.shape}, y = {y.shape}")
    
    # Удаление строк с пропущенными значениями
    mask = X.notnull().all(axis=1)
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]
    logging.debug(f"После удаления пропусков: X = {X_clean.shape}, y = {y_clean.shape}")

    # Удаление дубликатов
    duplicated_indices = X_clean.index.duplicated(keep='first')
    X_clean = X_clean.loc[~duplicated_indices]
    y_clean = y_clean.loc[~duplicated_indices]
    logging.debug(f"После удаления дубликатов: X = {X_clean.shape}, y = {y_clean.shape}")
    
    # Проверка индексов
    if not X_clean.index.equals(y_clean.index):
        raise ValueError("Индексы X и y не совпадают после очистки данных")

    return X_clean, y_clean


# Удаление выбросов
def remove_outliers(data):
    logging.info("Удаление выбросов: Начало")
    logging.info(f"Входные колонки: {data.columns.tolist()}")
    
    if 'target' not in data.columns:
        raise KeyError("Колонка 'target' отсутствует перед удалением выбросов")
    
    # Исходное распределение классов
    logging.info(f"Исходное распределение target:\n{data['target'].value_counts()}")
    
    # Разделяем данные по классам
    data_0 = data[data['target'] == 0]
    data_1 = data[data['target'] == 1]
    
    logging.info(f"Размеры по классам до обработки: class 0 = {len(data_0)}, class 1 = {len(data_1)}")
    
    def remove_outliers_from_group(group_data):
        # Работаем только с числовыми признаками, исключая target
        numeric_cols = [col for col in group_data.select_dtypes(include=[np.number]).columns if col != 'target']
        numeric_data = group_data[numeric_cols]
        
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1  # Вычисление интерквартильного размаха
        
        # Используем более мягкий порог для выбросов (2.0 вместо 1.5)
        mask = ~((numeric_data < (Q1 - 2.0 * IQR)) | (numeric_data > (Q3 + 2.0 * IQR))).any(axis=1)
        return group_data[mask]
    
    # Обрабатываем каждый класс отдельно
    cleaned_0 = remove_outliers_from_group(data_0)
    cleaned_1 = remove_outliers_from_group(data_1)
    
    logging.info(f"Размеры по классам после обработки: class 0 = {len(cleaned_0)}, class 1 = {len(cleaned_1)}")
    
    # Объединяем обратно
    filtered_data = pd.concat([cleaned_0, cleaned_1])
    
    if 'target' not in filtered_data.columns:
        raise KeyError("Колонка 'target' была удалена в процессе обработки выбросов")
    
    logging.info("Удаление выбросов: Завершено")
    logging.info(f"Выходные колонки: {filtered_data.columns.tolist()}")
    logging.info(f"Итоговое распределение target:\n{filtered_data['target'].value_counts()}")
    
    return filtered_data


def add_clustering_feature(data):
    logging.info("Кластеризация: Начало")
    logging.info(f"Входные колонки: {data.columns.tolist()}")
    
    # Сохраняем target
    target = data['target'].copy() if 'target' in data.columns else None
    
    features_for_clustering = [
        'close', 'volume', 'rsi', 'macd', 'atr', 'sma_10', 'sma_30', 'ema_50', 'ema_200',
        'bb_width', 'macd_diff', 'obv', 'returns', 'log_returns'
    ]
    
    # Проверка наличия признаков
    available_features = [f for f in features_for_clustering if f in data.columns]
    
    if not available_features:
        raise ValueError("Нет доступных признаков для кластеризации")
        
    # Кластеризация
    kmeans = KMeans(n_clusters=5, random_state=42)
    clustered_data = data.copy()
    clustered_data['cluster'] = kmeans.fit_predict(data[available_features])
    
    # Восстанавливаем target
    if target is not None:
        clustered_data['target'] = target
        
    logging.info("Кластеризация: Завершено")
    logging.info(f"Выходные колонки: {clustered_data.columns.tolist()}")
    return clustered_data


# Аугментация данных (добавление шума)
def augment_data(X):
    """
    Аугментация только признаков (без target)
    Args:
        X: DataFrame с признаками (без target)
    Returns:
        DataFrame: аугментированные признаки
    """
    logging.info(f"Начало аугментации. Shape входных данных: {X.shape}")
    
    # Добавляем шум только к признакам
    noise = np.random.normal(0, 0.01, X.shape)
    augmented_features = X + noise
    
    # Восстанавливаем индексы и колонки
    augmented_features = pd.DataFrame(augmented_features, 
                                    columns=X.columns, 
                                    index=X.index)
    
    logging.info(f"Завершение аугментации. Shape выходных данных: {augmented_features.shape}")
    return augmented_features


# Функции для SMOTETomek
def smote_process(X_chunk, y_chunk, chunk_id):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_chunk, y_chunk)
    
    if 'target' not in data.columns:
        logging.error("Колонка 'target' отсутствует в данных.")
        raise KeyError("Колонка 'target' отсутствует.")

    return X_resampled, y_resampled


def parallel_smote(X, y, n_chunks=4):
    # Проверяем наличие как минимум двух классов
    unique_classes = y.unique()
    logging.info(f"Классы в y: {unique_classes}")
    if len(unique_classes) < 2:
        raise ValueError(f"Невозможно применить SMOTETomek, так как y содержит только один класс: {unique_classes}")

    X_chunks = np.array_split(X, n_chunks)
    y_chunks = np.array_split(y, n_chunks)
    results = Parallel(n_jobs=n_chunks)(
        delayed(smote_process)(X_chunk, y_chunk, idx)
        for idx, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks))
    )
    X_resampled = np.vstack([res[0] for res in results])
    y_resampled = np.hstack([res[1] for res in results])
    
    if 'target' not in data.columns:
        logging.error("Колонка 'target' отсутствует в данных.")
        raise KeyError("Колонка 'target' отсутствует.")

    return X_resampled, y_resampled

def ensure_datetime_index(data):
    """
    Гарантирует, что DataFrame имеет DatetimeIndex и колонку 'timestamp'.
    Если колонка 'timestamp' отсутствует, функция проверяет, является ли индекс уже DatetimeIndex.
    Если индекс не является DatetimeIndex, пытается преобразовать его в datetime.
    Если преобразование не удалось – выбрасывается ValueError.
    """
    if 'timestamp' in data.columns:
        # Если колонка уже есть, попробуем её привести к datetime и установить как индекс.
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data = data.dropna(subset=['timestamp'])
            data = data.set_index('timestamp')
            logging.info("Колонка 'timestamp' успешно приведена к datetime и установлена как индекс.")
        except Exception as e:
            raise ValueError("Не удалось преобразовать колонку 'timestamp' в DatetimeIndex.") from e
    else:
        # Если колонки нет, проверяем индекс
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                new_index = pd.to_datetime(data.index, errors='coerce')
                if new_index.isnull().all():
                    raise ValueError("Индекс не удалось преобразовать в DatetimeIndex.")
                data.index = new_index
                data['timestamp'] = new_index
                logging.info("Индекс успешно преобразован в DatetimeIndex и добавлен как колонка 'timestamp'.")
            except Exception as e:
                raise ValueError("Данные не содержат временного индекса или колонки 'timestamp'.") from e
    return data



# Подготовка данных для модели
def prepare_data(data):
    logging.info("Начало подготовки данных")
    
    # Если данные являются словарём, объединяем их в один DataFrame
    if isinstance(data, dict):
        data = pd.concat(data.values(), ignore_index=False)

    # Проверка на пустые данные
    if data.empty:
        raise ValueError("Входные данные пусты")

    logging.info(f"Исходная форма данных: {data.shape}")

    # Убедимся, что временной индекс установлен
    logging.info(f"Исходная форма данных: {data.shape}")
    data = ensure_datetime_index(data)

    
    # Предобработка данных для нескольких монет
    if isinstance(data, dict):
        
        data = preprocess_market_data(data)  # Это комплексная предобработка, которая уже включает в себя все нужные функции
        if isinstance(data, dict):
            data = pd.concat(data.values())
            
    else:
        
        # Если данные в виде одного DataFrame, применяем предобработку последовательно
        data = detect_anomalies(data)
        logging.info("Аномалии обнаружены и помечены")
        
        data = validate_volume_confirmation_bullish(data)  # Используем специальную версию для бычьего рынка
        logging.info("Добавлены признаки подтверждения объемом для бычьего рынка")
        
        data = remove_noise(data)
        logging.info("Шум отфильтрован")

    # Извлечение признаков
    data = extract_features(data)
    logging.info(f"После извлечения признаков: {data.shape}")

    # Удаление выбросов
    data = remove_outliers(data)
    logging.info(f"После удаления выбросов: {data.shape}")

    # Добавление кластеризационного признака
    data = add_clustering_feature(data)
    logging.info(f"После кластеризации: {data.shape}")

    # Список признаков
    features = [col for col in data.columns if col not in ['target', 'symbol', 'close_time', 'ignore']]
    logging.info(f"Количество признаков: {len(features)}")
    logging.info(f"Список признаков: {features}")
    logging.info(f"Распределение target:\n{data['target'].value_counts()}")

    return data, features


def update_model_if_new_data(ensemble_model, selected_features, model_filename, new_data_available, updated_data):
    """
    Обновление модели при наличии новых данных.
    """
    if new_data_available:
        logging.info("Обнаружены новые данные. Обновление модели...")
        ensemble_model, selected_features = train_ensemble_model(updated_data, selected_features, model_filename)
        logging.info("Модель успешно обновлена.")
    return ensemble_model


def get_checkpoint_path(model_name, market_type):
    """
    Создает уникальный путь к чекпоинтам для каждой модели.
    
    Args:
        model_name (str): Название модели ('rf', 'xgb', 'lgbm', etc.)
        market_type (str): Тип рынка ('bullish', 'bearish', 'flat')
    
    Returns:
        str: Путь к директории чекпоинтов
    """
    checkpoint_path = os.path.join("checkpoints", market_type, model_name)
    ensure_directory(checkpoint_path)
    return checkpoint_path
# Определение функции балансировки классов
def balance_classes(X, y):
    """
    Балансирует классы с использованием SMOTETomek.
    """
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res


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

def train_models_for_intervals(data, intervals, selected_features=None):
    """
    Обучение моделей для разных временных интервалов.
    """
    models = {}
    for interval in intervals:
        logging.info(f"Агрегация данных до {interval} интервала")
        interval_data = data.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logging.info(f"Извлечение признаков для {interval} интервала")
        prepared_data, features = prepare_data(interval_data)
        selected_features = features if selected_features is None else selected_features

        logging.info(f"Обучение модели для {interval} интервала")
        X = prepared_data[selected_features]
        y = prepared_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        models[interval] = (model, selected_features)
    return models


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

def train_ensemble_model(data, selected_features, model_filename='bullish_stacked_ensemble_model.pkl'):
    """
    Обучает ансамбль моделей (3-классная классификация: 0=HOLD, 1=SELL, 2=BUY)
    специально для бычьего рынка, с учётом SMOTETomek, аугментации и т.д.
    """
    logging.info("Начало обучения ансамбля моделей (3-класса)")

    # 1) Проверяем входные данные
    if data.empty:
        raise ValueError("Входные данные пусты")
    if not isinstance(selected_features, list):
        raise TypeError("selected_features должен быть списком")
    assert all(feat != 'target' for feat in selected_features), "target не должен быть в списке признаков"
    
    logging.info(f"Входные данные shape: {data.shape}")
    logging.info(f"Входные колонки: {data.columns.tolist()}")
    debug_target_presence(data, "Начало обучения ансамбля")

    if 'target' not in data.columns:
        raise KeyError("Отсутствует колонка 'target' во входных данных")

    # 2) Проверяем, что есть хотя бы 2 класса
    target_dist = data['target'].value_counts()
    if len(target_dist) < 2:
        raise ValueError(f"Недостаточно классов в target: {target_dist}")

    # 3) X, y + логирование
    y = data['target'].copy()
    X = data[selected_features].copy()

    logging.info(f"Распределение классов до обучения:\n{y.value_counts()}")
    logging.info(f"Размеры данных: X = {X.shape}, y = {y.shape}")
    debug_target_presence(pd.concat([X, y], axis=1), "Перед очисткой данных")

    # 4) Очистка данных
    X_clean, y_clean = clean_data(X, y)
    logging.info(f"После clean_data: X = {X_clean.shape}, y = {y_clean.shape}")
    debug_target_presence(pd.concat([X_clean, y_clean], axis=1), "После очистки данных")

    # Если после очистки остался один класс, ошибка
    if len(pd.unique(y_clean)) < 2:
        raise ValueError(f"После очистки остался только один класс: {pd.value_counts(y_clean)}")

    # 5) Удаляем выбросы
    logging.info("Удаление выбросов: Начало")
    combined_data = pd.concat([X_clean, y_clean], axis=1)
    combined_data_cleaned = remove_outliers(combined_data)
    removed_count = combined_data.shape[0] - combined_data_cleaned.shape[0]
    logging.info(f"Удалено выбросов: {removed_count} строк")
    X_clean = combined_data_cleaned.drop(columns=['target'])
    y_clean = combined_data_cleaned['target']
    logging.info(f"После remove_outliers: X = {X_clean.shape}, y = {y_clean.shape}")
    assert X_clean.index.equals(y_clean.index), "Индексы X и y не совпадают после удаления выбросов!"

    # 6) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    logging.info(f"Train size: X = {X_train.shape}, y = {y_train.shape}")
    logging.info(f"Test size: X = {X_test.shape}, y = {y_test.shape}")
    debug_target_presence(pd.concat([X_train, y_train], axis=1), "После разделения на выборки")

    # 7) Масштабирование
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 8) Аугментация
    X_augmented = augment_data(pd.DataFrame(X_train_scaled, columns=X_train.columns))
    logging.info(f"После аугментации: X = {X_augmented.shape}")
    debug_target_presence(pd.DataFrame(X_augmented, columns=X_train.columns), "После аугментации")

    # 9) Применяем SMOTETomek (балансировка классов)
    logging.info("Применение SMOTETomek")
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_augmented, y_train)
    logging.info(f"После SMOTETomek: X = {X_resampled.shape}, y = {y_resampled.shape}")
    logging.info(f"Распределение классов после SMOTETomek:\n{pd.Series(y_resampled).value_counts()}")
    debug_target_presence(pd.DataFrame(X_resampled, columns=X_train.columns), "После SMOTETomek")

    check_class_balance(y_resampled)
    check_feature_quality(pd.DataFrame(X_resampled, columns=X_train.columns), y_resampled)

    # 10) Проверяем, не был ли ансамбль уже сохранён
    if os.path.exists(ensemble_checkpoint_path):
        logging.info(f"[Ensemble] Загрузка ансамбля из {ensemble_checkpoint_path}")
        saved_data = joblib.load(ensemble_checkpoint_path)
        return saved_data["ensemble_model"], saved_data["scaler"], saved_data["features"]

    # 11) Инициализация базовых моделей под 3 класса
    rf_model = CheckpointRandomForest(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=5
    )

    gb_model = CheckpointGradientBoosting(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8
    )

    xgb_model = CheckpointXGBoost(
        n_estimators=100,
        max_depth=3,
        subsample=0.8,
        min_child_weight=5,
        learning_rate=0.01,
        objective='multi:softprob',  # <-- многоклассный XGBoost
        num_class=3                  # <-- 3 класса
    )

    lgbm_model = CheckpointLightGBM(
        n_estimators=100,
        num_leaves=16,
        learning_rate=0.1,
        min_data_in_leaf=5,
        random_state=42,
        **{"objective": "multiclass", "num_class": 3}
    )

    catboost_model = CheckpointCatBoost(
        iterations=200,
        depth=4,
        learning_rate=0.1,
        min_data_in_leaf=5,
        save_snapshot=False,
        random_state=42,
        loss_function='MultiClass'
    )

    # Оптимизированные веса для бычьего рынка (будут использоваться в мета-модели, если понадобится)
    meta_weights = {
        'xgb': 0.3,     # Самый высокий вес для быстрой реакции
        'lgbm': 0.3,    # Тоже быстрая реакция
        'catboost': 0.2,  # Средний вес
        'gb': 0.1,      # Низкий вес
        'rf': 0.1       # Низкий вес из-за медленной реакции
    }

    base_learners = [
        ('rf', rf_model),
        ('gb', gb_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('catboost', catboost_model)
    ]

    # 12) Ещё раз масштабируем итоговый X_resampled
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # 13) Обучаем каждую базовую модель (3 класса)
    for name, model in base_learners:
        checkpoint_path = get_checkpoint_path(name, market_type)
        final_checkpoint = os.path.join(checkpoint_path, f"{name}_final.joblib")
        logging.info(f"[Ensemble] Обучение модели {name}")
        model.fit(X_resampled_scaled, y_resampled)
        joblib.dump(model, final_checkpoint)
        logging.info(f"[Ensemble] Модель {name} сохранена в {final_checkpoint}")

    # 14) Обучение стекинг-ансамбля
    logging.info("[Ensemble] Обучение стекинг-ансамбля (3-класса)")
    # --- Этап поиска оптимальных гиперпараметров для мета-модели ---
    meta_model_candidate = XGBClassifier(
        random_state=42
    )
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=meta_model_candidate,
                               param_grid=param_grid,
                               cv=3,
                               scoring='f1_macro',
                               n_jobs=-1)
    grid_search.fit(X_resampled_scaled, y_resampled)
    meta_model = grid_search.best_estimator_
    logging.info(f"Лучшие гиперпараметры мета-модели: {grid_search.best_params_}")
    # ------------------------------------------------------------------

    # Удаляем старые чекпоинты для базовых моделей
    for name, _ in base_learners:
        checkpoint_dir = os.path.join(checkpoint_base_dir, name)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            ensure_directory(checkpoint_dir)

    # Пересчитываем масштабирование для ансамбля
    X_resampled_scaled = RobustScaler().fit_transform(X_resampled)

    ensemble_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,  # Используем оптимизированную мета-модель из GridSearchCV
        passthrough=True,  # добавляет исходные признаки на вход мета-модели
        cv=5,
        n_jobs=1
    )

    ensemble_model.fit(X_resampled_scaled, y_resampled)

    # 15) Оценка на тестовой выборке
    y_pred = ensemble_model.predict(X_test_scaled)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    logging.info(f"F1-Score (macro, 3 класса): {f1_macro:.4f}")
    logging.info(f"Precision (macro, 3 класса): {precision:.4f}")
    logging.info(f"Recall (macro, 3 класса): {recall:.4f}")
    logging.info(f"[Ensemble] Weighted F1-score (3-класса): {f1_macro:.4f}")

    # 16) Сохраняем итог
    save_data = {
        "ensemble_model": ensemble_model,
        "scaler": scaler,
        "features": selected_features
    }
    ensure_directory(os.path.dirname(ensemble_checkpoint_path))
    joblib.dump(save_data, ensemble_checkpoint_path)
    logging.info(f"[Ensemble] Ансамбль (3-класса) сохранён в {ensemble_checkpoint_path}")
    
    return {"ensemble_model": ensemble_model, "scaler": scaler, "features": selected_features}




# Основной запуск
if __name__ == "__main__":
    
    strategy = initialize_strategy()
    
    # Инициализация клиента Binance
    
    symbols = ['BTCUSDC', 'ETHUSDC', 'BNBUSDC','XRPUSDC', 'ADAUSDC', 'SOLUSDC', 'DOTUSDC', 'LINKUSDC', 'TONUSDC', 'NEARUSDC']
        
    bullish_periods = [
            {"start": "2017-01-01", "end": "2017-03-31"},
            {"start": "2017-06-01", "end": "2017-08-31"},
            {"start": "2017-11-01", "end": "2018-01-16"},
            {"start": "2020-11-01", "end": "2021-01-31"},
            {"start": "2021-03-01", "end": "2021-04-30"},
            {"start": "2021-08-15", "end": "2021-10-20"},
            {"start": "2023-02-01", "end": "2023-03-31"},
            {"start": "2023-03-15", "end": "2023-05-15"}
        ]



    data_dict = load_bullish_data(symbols, bullish_periods, interval="1m")

    # Проверяем, что словарь не пустой
    if not data_dict:
        raise ValueError("Ошибка: load_bullish_data() вернула пустой словарь!")

    # Объединяем все DataFrame в один
    data = pd.concat(data_dict.values(), ignore_index=False)

    # Проверяем наличие 'timestamp'
    if 'timestamp' not in data.columns:
        logging.warning("'timestamp' отсутствует, проверяем индекс.")
        if isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = data.index
            logging.info("Индекс преобразован в колонку 'timestamp'.")
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")


    data, selected_features = prepare_data(data_dict)
    logging.debug(f"Доступные столбцы после подготовки данных: {data.columns.tolist()}")
    logging.debug(f"Выбранные признаки: {selected_features}")
        

    try:
        # Подготовка данных
        logging.info("Начало подготовки данных...")
        prepared_data, selected_features = prepare_data(data)
        
        # Проверки после подготовки данных
        if prepared_data.empty:
            raise ValueError("Подготовленные данные пусты")
            
        if 'target' not in prepared_data.columns:
            raise KeyError("Отсутствует колонка 'target' в подготовленных данных")
            
        if not selected_features:
            raise ValueError("Список признаков пуст")
            
        logging.info(f"Подготовка данных завершена. Размер данных: {prepared_data.shape}")
        logging.info(f"Количество выбранных признаков: {len(selected_features)}")
        
    except Exception as e:
        logging.error(f"Ошибка при подготовке данных: {e}")
        sys.exit(1)

    try:
        # Обучение ансамбля моделей
        logging.info("Начало обучения моделей...")
        ensemble_model, scaler, features = train_ensemble_model(prepared_data, selected_features, ensemble_model_filename)
        logging.info("Обучение ансамбля моделей завершено!")
        
    except Exception as e:
        logging.error(f"Ошибка при обучении моделей: {e}")
        sys.exit(1)

    try:
        # Сохранение модели
        if not os.path.exists('models'):
            os.makedirs('models')
            
        model_path = os.path.join('models', ensemble_model_filename)
        joblib.dump((ensemble_model, features), model_path)
        logging.info(f"Модель успешно сохранена в {model_path}")
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели: {e}")
        sys.exit(1)

    logging.info("Программа завершена успешно")
    sys.exit(0)