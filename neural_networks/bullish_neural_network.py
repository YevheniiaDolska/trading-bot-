import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as K
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from binance.client import Client
from datetime import datetime, timedelta
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from imblearn.combine import SMOTETomek
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
from tensorflow.keras.backend import clear_session
import glob
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBRegressor
from sklearn.metrics import f1_score
from tensorflow.keras.models import Model
import requests
import zipfile
from io import BytesIO
from threading import Lock
import joblib
from xgboost import XGBClassifier
from filterpy.kalman import KalmanFilter
from utils_output import ensure_directory, copy_output, save_model_output
from sklearn.impute import SimpleImputer
from numpy.random import RandomState
from tensorflow.keras.metrics import CategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from sklearn.metrics import f1_score



# Создаем необходимые директории
required_dirs = [
    "/workspace/logs",
    "/workspace/saved_models/bullish",
    "/workspace/checkpoints/bullish",
    "/workspace/data",
    "/workspace/output/bullish_ensemble"
]

for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)


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



network_name = "bullish_neural_network"
checkpoint_path_regular = f"checkpoints/bullish/{network_name}_checkpoint_epoch_{{epoch:02d}}.h5"
checkpoint_path_best = f"checkpoints/bullish/{network_name}_best_model.h5"

# Имя файла для сохранения модели
nn_model_filename = os.path.join("/workspace/saved_models/bullish", 'bullish_nn_model.h5')
log_file = os.path.join("/workspace/logs", "training_log_bullish_nn.txt")

# Инициализация TPU
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

    
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_logs_to_file(log_message):
    with open(log_file, 'a') as log_f:
        log_f.write(f"{datetime.now()}: {log_message}\n")
        
def check_feature_quality(X, y):
    logging.info("Проверка качества признаков...")
    logging.info(f"Форма X: {X.shape}")

    if isinstance(X, pd.DataFrame):
        logging.info("X представлен в виде DataFrame.")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        logging.info(f"Типы данных в X после приведения:\n{X.dtypes}")
    elif isinstance(X, np.ndarray):
        logging.info("X представлен в виде NumPy массива.")
        # Преобразуем массив в DataFrame с авто-сгенерированными именами столбцов
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        # Если тип колонок все же 'object', приводим их к числовому типу
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        logging.info(f"Типы данных в X после приведения:\n{X.dtypes}")

    else:
        logging.error(f"Ошибка: X имеет неизвестный тип: {type(X)}. Ожидается DataFrame или NumPy массив.")
        raise ValueError(f"Ошибка: Неверный формат данных X ({type(X)})")

    # Удаляем нечисловые колонки
    non_numeric_cols = X.columns[X.dtypes == 'object'].tolist()
    if non_numeric_cols:
        logging.warning(f"Удаляем нечисловые колонки: {non_numeric_cols}")
        X.drop(columns=non_numeric_cols, inplace=True)

    if X.shape[1] == 0:
        logging.error("Ошибка: В X не осталось числовых колонок после удаления нечисловых данных!")
        raise ValueError("X не содержит числовых признаков после фильтрации. Проверьте исходные данные.")

    # Заполняем пропуски медианными значениями с помощью SimpleImputer,
    # чтобы не терять все строки, как это делает dropna()
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    X = X.astype(np.float32)

    logging.info(f"Количество оставшихся пропущенных значений в X: {np.isnan(X).sum()}")

    # Вычисляем важность признаков
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_

    feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Score": scores
    }).sort_values("Score", ascending=False)

    logging.info("Топ-10 важных признаков:")
    logging.info(importance_df.head(10).to_string(index=False))

    return importance_df


def train_xgboost_on_embeddings(X_emb, y):
    logging.info("Обучение XGBoost на эмбеддингах...")
    xgb_model = XGBClassifier(
        objective='multi:softprob',  # многоклассовая задача
        n_estimators=100,
        max_depth=4,
        learning_rate=0.01,
        random_state=42,
        num_class=3
    )
    xgb_model.fit(X_emb, y)
    logging.info("XGBoost обучен.")
    return xgb_model

        
def calculate_cross_coin_features(data_dict):
    btc_data = data_dict['BTCUSDC']
    for symbol, df in data_dict.items():
        # CHANGED FOR SCALPING
        df['btc_corr'] = df['close'].rolling(15).corr(btc_data['close'])  # было 30
        df['rel_strength_btc'] = (df['close'].pct_change() - btc_data['close'].pct_change())
        
        # CHANGED FOR SCALPING
        df['beta_btc'] = (
            df['close'].pct_change().rolling(15).cov(btc_data['close'].pct_change())
            / btc_data['close'].pct_change().rolling(15).var()
        )  # было 30
        
        # CHANGED FOR SCALPING
        df['lead_lag_btc'] = df['close'].pct_change().shift(1).rolling(5).corr(
            btc_data['close'].pct_change()
        )  # было 10
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
        (data['volume'] > data['volume'].rolling(5).mean()),  # было 10
        1,
        np.where(
            (data['close'] < data['close'].shift(1)) & 
            (data['volume'] > data['volume'].rolling(5).mean()),
            -1,
            0
        )
    )
    data['volume_strength'] = (
        data['volume'] / data['volume'].rolling(5).mean()
    ) * data['volume_trend_conf']  # было 10
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(2).sum()  # было 3
    return data



def remove_noise(data):
    """
    Улучшенная фильтрация шума с использованием фильтра Калмана.
    Параметры фильтра настроены для быстрой адаптации к колебаниям на 1-минутном таймфрейме.
    """
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    kf.R = 2
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])  # Понижено для быстрого отклика фильтра
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

# Кастомная функция потерь для бычьего рынка, ориентированная на прибыль
def custom_profit_loss(y_true, y_pred):
    """
    Custom loss для HFT: штраф за flip-flop и комиссию + вся твоя логика.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    diff = y_pred - y_true
    log_factor = tf.math.log1p(tf.abs(diff) + 1e-7)

    loss_hold = tf.where(
        y_pred > 0.7,
        0.5 * tf.abs(diff) * log_factor,
        tf.where(
            y_pred < 0.2,
            0.5 * tf.abs(diff) * log_factor,
            2.0 * tf.abs(diff) * log_factor
        )
    )
    loss_buy = tf.where(
        y_pred <= 0.7,
        3.0 * tf.abs(diff) * log_factor,
        tf.abs(diff) * log_factor
    )
    loss_sell = tf.where(
        y_pred >= 0.2,
        3.0 * tf.abs(diff) * log_factor,
        tf.abs(diff) * log_factor
    )

    base_loss = tf.where(
        tf.equal(y_true, 0.0),
        loss_hold,
        tf.where(
            tf.equal(y_true, 1.0),
            loss_buy,
            loss_sell
        )
    )

    uncertainty_penalty = tf.where(
        tf.logical_and(y_pred > 0.3, y_pred < 0.7),
        1.0 * tf.abs(diff) * log_factor,
        0.0
    )
    time_penalty = 0.1 * tf.abs(diff) * (
        tf.expand_dims(tf.cast(tf.range(tf.shape(diff)[0]), tf.float32), axis=1) / tf.cast(tf.shape(diff)[0], tf.float32)
    )
    # Новый блок — Flip-Flop и комиссия
    action = tf.round(y_pred)  # округление к ближайшему действию (0, 1, 2)
    action_change = tf.abs(action[1:] - action[:-1])
    flip_flop_penalty = 0.002 * tf.reduce_sum(action_change)
    transaction_cost = 0.001 * tf.reduce_sum(action_change)
    total_loss = tf.reduce_mean(base_loss + uncertainty_penalty + time_penalty) + transaction_cost + flip_flop_penalty
    return total_loss


class F1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_data
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        f1 = f1_score(y_val, y_pred, average='weighted')
        print(f"F1-score (weighted): {f1:.4f}")


# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.math.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.math.reduce_sum(output, axis=1)


def load_all_data(symbols, start_date, end_date, interval='1m'):
    """
    Загружает данные для всех символов и добавляет межмонетные признаки.
    
    Args:
        symbols (list): Список торговых пар
        start_date (datetime): Начальная дата
        end_date (datetime): Конечная дата
        interval (str): Интервал свечей
    
    Returns:
        pd.DataFrame: Объединенные данные со всеми признаками
    """
    # Словарь для хранения данных по каждой монете
    symbol_data_dict = {}
    
    logging.info(f"Начало загрузки данных для символов: {symbols}")
    logging.info(f"Период: с {start_date} по {end_date}, интервал: {interval}")
    
    # Загрузка данных параллельно
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(get_historical_data, symbol, interval, start_date, end_date): symbol 
                  for symbol in symbols}
        
        for future in futures:
            symbol = futures[future]
            try:
                logging.info(f"Ожидание данных для {symbol}")
                symbol_data = future.result()
                
                if symbol_data is not None:
                    # Добавляем базовые признаки для каждой монеты
                    symbol_data = detect_anomalies(symbol_data)
                    symbol_data = validate_volume_confirmation(symbol_data)
                    symbol_data = remove_noise(symbol_data)
                    
                    symbol_data_dict[symbol] = symbol_data
                    logging.info(f"Данные для {symbol} успешно загружены и обработаны, количество строк: {len(symbol_data)}")
            except Exception as e:
                logging.error(f"Ошибка при загрузке данных для {symbol}: {e}")
                save_logs_to_file(f"Ошибка при загрузке данных для {symbol}: {e}")
    
    if not symbol_data_dict:
        error_msg = "Не удалось получить данные ни для одного символа"
        logging.error(error_msg)
        save_logs_to_file(error_msg)
        raise ValueError(error_msg)
    
    # Добавляем межмонетные признаки
    try:
        logging.info("Добавление межмонетных признаков...")
        symbol_data_dict = calculate_cross_coin_features(symbol_data_dict)
        
        # Объединяем все данные
        all_data = []
        for symbol, df in symbol_data_dict.items():
            df['symbol'] = symbol
            all_data.append(df)
        
        data = pd.concat(all_data)
        
        # Проверяем наличие всех необходимых признаков
        expected_features = ['btc_corr', 'rel_strength_btc', 'beta_btc', 'lead_lag_btc',
                           'volume_strength', 'volume_accumulation', 'is_anomaly', 
                           'clean_returns']
        
        missing_features = [f for f in expected_features if f not in data.columns]
        if missing_features:
            logging.warning(f"Отсутствуют следующие признаки: {missing_features}")
        
        # Удаляем строки с пропущенными значениями
        initial_rows = len(data)
        data = data.dropna()
        dropped_rows = initial_rows - len(data)
        if dropped_rows > 0:
            logging.info(f"Удалено {dropped_rows} строк с пропущенными значениями")
        
        logging.info(f"Всего загружено и обработано {len(data)} строк данных")
        save_logs_to_file(f"Всего загружено и обработано {len(data)} строк данных")
        
        return data
        
    except Exception as e:
        error_msg = f"Ошибка при обработке межмонетных признаков: {e}"
        logging.error(error_msg)
        save_logs_to_file(error_msg)
        raise

# Загрузка данных с Binance
def get_historical_data(symbols, bullish_periods, interval="1m", save_path="/workspace/data/binance_data_bullish.csv"
):
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

                    # 🛠 Проверяем, загружен ли `timestamp`
                    if "timestamp" not in df.columns:
                        logging.error(f"❌ Ошибка: Колонка 'timestamp' отсутствует в df для {symbol}")
                        return None

                    # 🛠 Преобразуем timestamp в datetime и ставим индекс
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                    df.set_index("timestamp", inplace=True)
                    
                    # Если индекс не является DatetimeIndex, пытаемся его преобразовать
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index, errors="coerce")
                        except Exception as e:
                            logging.error(f"Ошибка преобразования индекса в DatetimeIndex: {e}")
                                    
                    # Если по какой-то причине столбца 'timestamp' больше нет, добавляем его из индекса
                    if "timestamp" not in df.columns:
                        df["timestamp"] = df.index
                                        
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

    df = pd.concat(all_data, ignore_index=False)  # Не используем ignore_index, чтобы сохранить `timestamp`  

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
            logging.info(f"🔧 Заполняем {nan_percentage:.2%} пропущенных данных `ffill`.")
            df.fillna(method='ffill', inplace=True)  # Заполняем предыдущими значениями

    df.to_csv(save_path, index_label='timestamp')
    logging.info(f"💾 Данные сохранены в {save_path}")

    return save_path


def load_bullish_data(symbols, bullish_periods, interval="1m", save_path="binance_data_bullish.csv"):
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
            for chunk in pd.read_csv(save_path,
                                     index_col='timestamp',
                                     parse_dates=['timestamp'],
                                     on_bad_lines='skip',
                                     chunksize=CHUNK_SIZE):
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
            executor.submit(get_historical_data, [symbol], bullish_periods, interval, save_path): symbol
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

def convert_df_dtypes(df):
    # Конвертируем float64 в float32
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
    # Конвертируем int64 в int32
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = df[col].astype(np.int32)
    return df


# Извлечение признаков
def extract_features(data, multi_horizon=[1,2,3], profit_thresholds=[0.00005, 0.0001, 0.0002], label_smoothing=0.05):
    """
    Извлечение признаков для каждого символа отдельно. Полная версия!
    - multi_horizon: список шагов вперед для multi-label target
    - profit_thresholds: список порогов для BUY/SELL на каждый горизонт
    - label_smoothing: небольшое сглаживание целевой переменной (anti-stuck effect)
    """
    def _extract_features_per_symbol(data):
        data = data.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float32)
        
        # 1) Заменяем ±inf → NaN и заполняем пропуски вперёд-назад
        data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        # 2) Убираем строки только там, где в фичах ещё остались NaN,
        #    но не трогаем столбец 'target' (и не удаляем по нему).
        features = [c for c in data.columns if c not in ("target", "timestamp", "symbol")]
        data = data.dropna(subset=features).reset_index(drop=True)


        # 3) Теперь вы можете спокойно брать features и таргет
        features = [c for c in data.columns if c not in ("target","timestamp","symbol")]
        X_df = data[features].apply(pd.to_numeric, errors="coerce")
        y     = data["target"].astype(int).values

        data = convert_df_dtypes(data)

        # Базовая метка рынка (bullish → 0)
        data['market_type'] = 0

        # 1. Базовые метрики доходности и импульса
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1) + 1e-7)
        data['momentum_1m'] = data['close'].diff(1)
        data['momentum_3m'] = data['close'].diff(3)
        data['acceleration'] = data['momentum_1m'].diff()

        # 2. Волатильность
        roll_returns_20 = data['returns'].rolling(20)
        volatility = roll_returns_20.std()
        volume_volatility = data['volume'].pct_change().rolling(20).std()
        base_strong = 0.001
        base_medium = 0.0005
        strong_threshold = base_strong * (1 + volatility / (volatility.mean() + 1e-7))
        medium_threshold = base_medium * (1 + volatility / (volatility.mean() + 1e-7))
        volume_factor = 1 + (volume_volatility / (volume_volatility.mean() + 1e-7))

        # 3. Объем
        data['volume_delta'] = data['volume'].diff()
        data['volume_momentum'] = data['volume'].diff().rolling(3).sum()
        volume_mean_5 = data['volume'].rolling(5).mean()
        data['volume_ratio'] = data['volume'] / (volume_mean_5 * volume_factor + 1e-7)

        # 4. Быстрые тренды
        data['sma_2'] = SMAIndicator(data['close'], window=2).sma_indicator()
        data['sma_3'] = SMAIndicator(data['close'], window=3).sma_indicator()
        data['sma_10'] = SMAIndicator(data['close'], window=10).sma_indicator()
        data['ema_3'] = data['close'].ewm(span=3, adjust=False).mean()
        data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()

        # 5. Микро-тренды
        data['micro_trend'] = np.where(
            data['close'] > data['close'].shift(1), 1,
            np.where(data['close'] < data['close'].shift(1), -1, 0)
        )
        data['micro_trend_strength'] = data['micro_trend'].rolling(3).sum()

        # 6. MACD
        macd = MACD(data['close'], window_slow=12, window_fast=6, window_sign=3)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        data['macd_acceleration'] = data['macd_diff'].diff()

        # 7. Осцилляторы
        data['rsi_5'] = RSIIndicator(data['close'], window=5).rsi()
        data['rsi_3'] = RSIIndicator(data['close'], window=3).rsi()
        stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=5)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()

        # 8. Волатильность и позиции в Bollinger Bands
        bb = BollingerBands(data['close'], window=10)
        data['bb_width'] = bb.bollinger_wband()
        data['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-7)
        data['atr_3'] = AverageTrueRange(data['high'], data['low'], data['close'], window=3).average_true_range()

        # 9. Свечной анализ
        data['candle_body'] = data['close'] - data['open']
        data['body_ratio'] = np.abs(data['candle_body']) / (data['high'] - data['low'] + 1e-7)
        data['upper_wick'] = data['high'] - np.maximum(data['open'], data['close'])
        data['lower_wick'] = np.minimum(data['open'], data['close']) - data['low']

        # 10. Объемные индикаторы
        data['obv'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
        data['volume_trend'] = data['volume'].diff() / (data['volume'].shift(1) + 1e-7)

        # 11. Индикаторы ускорения
        data['price_acceleration'] = data['returns'].diff()
        data['volume_acceleration'] = data['volume_delta'].diff()

        # 12. Multi-horizon target (агрессивная)
        # Покрываем сразу несколько горизонтов и несколько порогов
        buy = np.zeros(len(data), dtype=bool)
        sell = np.zeros(len(data), dtype=bool)
        for horizon, thr in zip(multi_horizon, profit_thresholds):
            future_ret = data['close'].shift(-horizon) / data['close'] - 1
            buy |= (future_ret > thr)
            sell |= (future_ret < -thr)
        # Label smoothing: чуть-чуть BUY/SELL в HOLD
        target = np.zeros(len(data), dtype=np.float32) + label_smoothing
        target[buy] = 1 - label_smoothing
        target[sell] = 2 - label_smoothing
        data = data.iloc[:-max(multi_horizon)] # урезаем хвост, где нет future

        # Дополнительное условие для SELL — высокая RSI и верхняя BB, для BUY — объём, MACD, нижняя BB
        future_ret = data['close'].shift(-1) / data['close'] - 1  # самый короткий горизонт для условия
        strong_buy = (
            (future_ret > profit_thresholds[0]) &
            (data['volume'] > data['volume'].rolling(5).mean()) &
            (data['macd_diff'] > 0) &
            (data['bb_position'] < 0.6)
        )
        strong_sell = (
            (future_ret < -profit_thresholds[0]) &
            (data['rsi_5'] > 60) &
            (data['bb_position'] > 0.4)
        )
        target[strong_buy] = 1
        target[strong_sell] = 2

        # Округляем для категоризации
        data['target'] = np.round(target).astype(int)
        return data.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    if 'symbol' in data.columns:
        grouped = data.groupby('symbol')
        features = grouped.apply(_extract_features_per_symbol).reset_index(drop=True)
    else:
        features = _extract_features_per_symbol(data)
    # Логируем дистрибуцию!
    print("Target distrib:", features['target'].value_counts(normalize=True))
    return features




def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]


def add_clustering_feature(data):
    features_for_clustering = [
        'close', 'volume', 'rsi', 'macd', 'atr', 'sma_2', 'sma_3', 'sma_10', 'ema_3', 'ema_5',
        'bb_width', 'macd_diff', 'obv', 'returns', 'log_returns'
    ]
    
    max_for_kmeans = 200_000
    if len(data) > max_for_kmeans:
        sample_df = data.sample(n=max_for_kmeans, random_state=42)
    else:
        sample_df = data

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(sample_df[features_for_clustering])
    data['cluster'] = kmeans.predict(data[features_for_clustering])
    return data


def prepare_data(data, chunk_size=100_000, nan_threshold=0.10):
    logging.info("Начало подготовки данных")
    if data.empty:
        raise ValueError("Входные данные пусты")
    logging.info(f"Исходная форма данных: {data.shape}")

    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Данные не содержат временного индекса или колонки 'timestamp'.")

    def filter_bad_features(df, nan_threshold=0.10):
        bad_cols = [col for col in df.columns if df[col].isna().mean() + np.isinf(df[col]).mean() > nan_threshold]
        if bad_cols:
            logging.warning(f"Удаляются плохие признаки: {bad_cols}")
            df = df.drop(columns=bad_cols)
        return df

    def process_chunk(df_chunk):
        df_chunk = extract_features(df_chunk)
        df_chunk = remove_outliers(df_chunk)
        df_chunk = add_clustering_feature(df_chunk)
        df_chunk = filter_bad_features(df_chunk, nan_threshold)
        return df_chunk

    data = apply_in_chunks(data, process_chunk, chunk_size=chunk_size)
    data = filter_bad_features(data, nan_threshold)

    logging.info(f"После обработки: {data.shape}")
    features = [col for col in data.columns if col != 'target']
    logging.info(f"Количество признаков: {len(features)}")
    logging.info(f"Список признаков: {features}")
    logging.info(f"Распределение target:\n{data['target'].value_counts()}")
    return data, features


def clean_data(X, y):
    logging.info("Очистка данных от бесконечных значений и NaN")
    mask = np.isfinite(X).all(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    return X_clean, y_clean


def load_last_saved_model(model_filename):
    try:
        models = sorted(
            [f for f in os.listdir() if f.startswith(model_filename)],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        last_model = models[-1] if models else None
        if last_model:
            logging.info(f"Загрузка последней модели: {last_model}")
            return load_model(last_model)
        else:
            return None
    except Exception as e:
        logging.error(f"Ошибка при загрузке последней модели: {e}")
        return None



def balance_classes(X, y):
    print("Before balance:", Counter(y))
    max_for_smote = 300_000

    # 1) стратифицированная подвыборка, чтобы сохранить все классы
    if len(y) > max_for_smote:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=max_for_smote, stratify=y, random_state=42
        )
    else:
        X_sample, y_sample = X, y

    # 2) на всякий случай, если какой-то класс всё же выпал
    missing = set([0,1,2]) - set(np.unique(y_sample))
    if missing:
        for cls in missing:
            idx0 = np.where(y == cls)[0][0]
            X_sample = np.vstack([X_sample, X[idx0]])
            y_sample = np.append(y_sample, cls)

    # 3) собственно SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_sample, y_sample.astype(int))
    print("After balance:", Counter(y_res))
    return X_res, y_res





def ensemble_predict(nn_model, xgb_model, feature_extractor, X_seq, weight_nn=0.5, weight_xgb=0.5):
    # Получаем вероятности от нейросети — ожидаемая форма (n_samples, 3)
    nn_probs = nn_model.predict(X_seq)
    logging.info(f"nn_probs.shape = {nn_probs.shape}")
    
    # Извлекаем эмбеддинги для XGBoost
    embeddings = feature_extractor.predict(X_seq)
    
    # Получаем вероятности от XGBoost (форма (n_samples, 3))
    xgb_probs = xgb_model.predict_proba(embeddings)
    logging.info(f"xgb_probs.shape = {xgb_probs.shape}")
    
    # Взвешиваем и суммируем вероятности
    final_probs = weight_nn * nn_probs + weight_xgb * xgb_probs
    
    # Выбираем класс с максимальной вероятностью для каждого примера
    final_pred_class = np.argmax(final_probs, axis=1)
    return final_pred_class


# Двухэтапное обучение: baseline + fine-tune
def build_bullish_neural_network(data):
    """
    Двухэтапное обучение нейросети для бычьего рынка...
    """
    # --- Проверка существующей финальной модели ---
    final_path = "/workspace/saved_models/bullish_neural_network.h5"
    if os.path.exists(final_path):
        model = tf.keras.models.load_model(
            final_path,
            custom_objects={
                "custom_profit_loss": custom_profit_loss,
                "bull_profit_metric": bull_profit_metric
            }
        )
        logging.info("Loaded existing final model, skipping training.")
        return {"ensemble_model": {"nn_model": model}, "scaler": None}

    # --- Подготовка директорий ---
    os.makedirs("checkpoints/bullish", exist_ok=True)
    os.makedirs("/workspace/saved_models/bullish", exist_ok=True)
    
    fine_ckpt = "checkpoints/bullish/bullish_best_model.h5"
    
    def bull_profit_metric(y_true, y_pred):
            true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
            diff = true_one_hot - y_pred
            missed = tf.where(diff > 0, diff, 0.0)
            return tf.reduce_mean(missed)

    """
    Двухэтапное обучение нейросети для бычьего рынка:
      1) Train простой baseline LSTM-модели с CrossEntropy
      2) Fine-tune с custom_profit_loss и bull_profit_metric
    После этого — ансамблирование с XGBoost на эмбеддингах.
    Возвращает словарь с nn_model, xgb_model, feature_extractor и scaler.
    """
    finetune_callbacks = [
        EarlyStopping(
            monitor="val_bull_profit_metric",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath="checkpoints/bullish/bullish_checkpoint_epoch_{epoch:02d}.h5",
            save_weights_only=True,
            save_best_only=False,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=fine_ckpt,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_bull_profit_metric",
            verbose=1
        ),
        TensorBoard(log_dir=f"logs/finetune/{int(time.time())}")
    ]

    # --- 1. Подготовка данных и feature engineering должен быть выполнен до вызова ---
    # Предполагаем, что data уже содержит features и колонки ['target','timestamp','symbol']
    # --- 1) Сразу вынесем целевой столбец в отдельную переменную и очистим data от NaN/inf ---
    if "target" not in data.columns:
        raise KeyError("Входной DataFrame не содержит колонки 'target'")
    # сохраняем y_raw с оригинальными индексами
    y_raw = data["target"].astype(int).copy()

    # заменяем inf → NaN, затем заполняем по столбцам
    data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # оставляем только признаки (не трогаем y_raw)
    features = [c for c in data.columns if c not in ("target", "timestamp", "symbol")]
    X_df = data[features].copy()

    # 2) Отбрасываем строки, где после заполнения всё ещё есть NaN
    valid_idx = X_df.dropna(how="any").index
    X_df = X_df.loc[valid_idx]
    # и синхронно обрезаем y_raw
    y = y_raw.loc[valid_idx].values

    # 3) Преобразуем X в числовой массив и балансируем классы
    X = X_df.apply(pd.to_numeric, errors="coerce").values
    X, y = balance_classes(X, y)

    # 2. Train/Val split с стратификацией и масштабирование
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = RobustScaler(quantile_range=(10, 90))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 3. Создание временных последовательностей
    def create_sequences(X_arr, y_arr, timesteps=10):
        Xs, ys = [], []
        for i in range(len(X_arr) - timesteps):
            Xs.append(X_arr[i : i + timesteps])
            ys.append(y_arr[i + timesteps])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

    timesteps = 10
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, timesteps)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, timesteps)

    # --- 4. Baseline training (CrossEntropy) ---
    strategy = (
        tf.distribute.MirroredStrategy()
        if tf.config.list_physical_devices("GPU")
        else tf.distribute.get_strategy()
    )
    with strategy.scope():
        inp = Input(shape=(timesteps, X_train_seq.shape[2]))
        x = LSTM(64,  name='lstm1', return_sequences=False)(inp)
        x = BatchNormalization(name='bn_baseline')(x)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        out = Dense(3, activation="softmax")(x)
        baseline_model = Model(inp, out)
        baseline_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="baseline_acc")]
        )
    
    def find_best_lr(model, X_train, y_train, batch_size=128, min_lr=1e-5, max_lr=1e-2, num_lrs=30):
        import numpy as np
        import matplotlib.pyplot as plt
        orig_weights = model.get_weights()
        lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_lrs)
        losses = []
        for lr in lrs:
            tf.keras.backend.set_value(model.optimizer.lr, lr)
            model.set_weights(orig_weights)  # сбросить веса!
            loss = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0).history['loss'][-1]
            losses.append(loss)
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.show()
        # Верни оптимальный lr, если хочешь:
        return lrs[np.argmin(losses)]

    best_lr = find_best_lr(baseline_model, X_train_seq, y_train_seq)
    print("Лучший lr (визуально или автоматически):", best_lr)


    # --- Загрузка последнего baseline чекпоинта, если есть ---
    base_checks = sorted(
        glob.glob("checkpoints/bullish/baseline_checkpoint_epoch_*.h5")
    )
    if base_checks:
        baseline_model.load_weights(base_checks[-1])
        logging.info(f"Loaded last baseline checkpoint: {base_checks[-1]}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq))
        .shuffle(2048)
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq))
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )

    base_ckpt = "/workspace/saved_models/bullish/baseline_bullish_weights.h5"
    f1_callback = F1Callback((X_val_seq, y_val_seq))
    baseline_callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath="checkpoints/bullish/baseline_checkpoint_epoch_{epoch:02d}.h5",
            save_weights_only=True,
            save_best_only=False,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=base_ckpt,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        TensorBoard(log_dir=f"logs/baseline/{int(time.time())}")
    ]
    
    
    baseline_model = Model(inp, out)  # переинициализация!
    baseline_model.compile(
        optimizer=Adam(learning_rate=best_lr),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="baseline_acc")]
    )


    baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,
        callbacks=baseline_callbacks,
        verbose=1
    )
    baseline_model.save_weights(base_ckpt)

    # --- 5. Fine-tune: расширенная архитектура с custom_profit_loss и bull_profit_metric ---
    with strategy.scope():
        inp = Input(shape=(timesteps, X_train_seq.shape[2]))
        x = LSTM(64, return_sequences=True, name="lstm1")(inp)
        x = BatchNormalization(name="bn1")(x)
        x = Dropout(0.3, name="drop1")(x)
        x = LSTM(128, return_sequences=True, name="lstm2")(x)
        x = BatchNormalization(name="bn2")(x)
        x = Dropout(0.3, name="drop2")(x)
        x = LSTM(64, return_sequences=False, name="lstm3")(x)
        x = BatchNormalization(name="bn3")(x)
        x = Dropout(0.3, name="drop3")(x)
        x = Dense(128, activation="relu", name="dense1")(x)
        x = BatchNormalization(name="bn4")(x)
        x = Dropout(0.3, name="drop4")(x)
        emb = Dense(64, activation="relu", name="embedding_layer")(x)
        x = BatchNormalization(name="bn5")(emb)
        outputs = Dense(3, activation="softmax", name="output")(x)
        fine_model = Model(inp, outputs)
        fine_model.get_layer("lstm1").set_weights(
            baseline_model.get_layer("lstm1").get_weights()
        )


        fine_model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss=custom_profit_loss,
            metrics=[bull_profit_metric, CategoricalAccuracy(name="acc")]
        )

    # 1) вычисляем веса
    classes = np.unique(y_train_seq)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_seq
    )
    # 2) собираем словарь {класс: вес}
    class_weights = { int(cls): float(w) for cls, w in zip(classes, weights) }
    
    finetune_callbacks = [
        EarlyStopping(
            monitor="val_bull_profit_metric",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=fine_ckpt,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_bull_profit_metric",
            verbose=1
        ),
        TensorBoard(log_dir=f"logs/finetune/{int(time.time())}")
    ]
    fine_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,
        class_weight=class_weights,
        callbacks=finetune_callbacks,
        verbose=1
    )
    fine_model.save("/workspace/saved_models/bullish_neural_network.h5")

    # --- 6. Ансамблирование с XGBoost ---
    feat_ext = Model(
        inputs=fine_model.input,
        outputs=fine_model.get_layer("embedding_layer").output
    )
    emb_train = feat_ext.predict(X_train_seq)
    emb_val = feat_ext.predict(X_val_seq)
    xgb_model = XGBClassifier(objective="multi:softprob", num_class=3, random_state=42)
    xgb_model.fit(emb_train, y_train_seq)
    # 1) Получаем сырые вероятности от XGBoost (может быть (N,1), (N,2) или (N,3))
    raw_xgb_p = xgb_model.predict_proba(emb_val)

    # 2) Создаём гарантированный full-массив размера (N,3)
    n = raw_xgb_p.shape[0]
    full_xgb_p = np.zeros((n, 3), dtype=raw_xgb_p.dtype)

    # 3) Заполняем только те столбцы, которые видел XGBoost
    for idx, cls in enumerate(xgb_model.classes_):
        full_xgb_p[:, int(cls)] = raw_xgb_p[:, idx]

    # 4) Энсамблируем с neural-сеткой
    nn_p      = fine_model.predict(X_val_seq)
    ens_logits = 0.5 * nn_p + 0.5 * full_xgb_p
    ens        = np.argmax(ens_logits, axis=1)
    logging.info(
        f"Ensemble F1: {f1_score(y_val_seq, ens, average='weighted'):.4f}"
    )
    joblib.dump(xgb_model, "/workspace/saved_models/xgb_model_bullish.pkl")

    return {
        "ensemble_model": {
            "nn_model": fine_model,
            "xgb_model": xgb_model,
            "feature_extractor": feat_ext,
            "ensemble_weight_nn": 0.5,
            "ensemble_weight_xgb": 0.5
        },
        "scaler": scaler
    }




# Основной процесс обучения
if __name__ == "__main__":
    try:
        # Инициализация стратегии (TPU или CPU/GPU)
        strategy = initialize_strategy()

        symbols = ['BTCUSDC', 'ETHUSDC', 'BNBUSDC','XRPUSDC', 'ADAUSDC', 'SOLUSDC', 'DOTUSDC', 'LINKUSDC', 'TONUSDC', 'NEARUSDC']
        
        bullish_periods = [
            {"start": "2017-06-01", "end": "2017-08-31"},
            {"start": "2017-11-01", "end": "2018-01-16"},
            {"start": "2020-11-01", "end": "2021-01-31"},
            {"start": "2021-03-01", "end": "2021-04-30"},
            {"start": "2021-08-15", "end": "2021-10-20"},
            {"start": "2023-02-01", "end": "2023-03-31"},
            {"start": "2023-03-15", "end": "2023-05-15"},
            {"start": "2024-04-01", "end": "2024-06-30"}
        ]



        # Загрузка данных для бычьих периодов
        logging.info("Загрузка данных для бычьих периодов")
        data = load_bullish_data(symbols, bullish_periods, interval="1m")

        # Проверяем, что data — это словарь, и он не пуст
        if not isinstance(data, dict) or not data:
            raise ValueError("Ошибка: load_bullish_data() вернула пустой словарь!")

        # Объединяем все DataFrame из словаря `data` в один общий DataFrame
        data = pd.concat(data.values(), ignore_index=False)

        # Проверяем наличие 'timestamp' в колонках
        if 'timestamp' not in data.columns:
            logging.warning("'timestamp' отсутствует, проверяем индекс.")
            if isinstance(data.index, pd.DatetimeIndex):
                data['timestamp'] = data.index
                logging.info("Индекс преобразован в колонку 'timestamp'.")
            else:
                raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")


        # Проверка наличия колонки `timestamp`
        if 'timestamp' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Загруженные данные не содержат колонку 'timestamp'. Проверьте этап загрузки.")

        # Извлечение признаков
        logging.info("Извлечение признаков из данных")
        data = extract_features(data)

        # Очистка данных
        logging.info(f"Пропущенные значения перед очисткой:\n{data.isna().sum()}")
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Проверка наличия данных после очистки
        if data.empty:
            logging.error("После очистки данные отсутствуют. Проверьте входные данные.")
            raise ValueError("После очистки данные отсутствуют.")

        # Убедитесь, что пути к чекпоинтам определены
        checkpoint_path_regular = os.path.join("/workspace/checkpoints", f"{network_name}_checkpoint_epoch_{{epoch:02d}}.h5")
        checkpoint_path_best = os.path.join("/workspace/checkpoints", f"{network_name}_best_model.h5")


        # Обучение модели
        logging.info("Начало обучения модели для бычьего рынка")
        build_bullish_neural_network(data)

    except Exception as e:
        logging.error(f"Ошибка во время выполнения программы: {e}")
    finally:
        # Очистка сессии TensorFlow
        logging.info("Очистка сессии TensorFlow...")
        clear_session()  # Закрывает все графы и фоновые процессы TensorFlow


        logging.info("Программа завершена.")


