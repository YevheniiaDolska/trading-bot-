import pandas as pd
import numpy as np
import tensorflow as tf
import time
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
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
from tensorflow.keras.backend import clear_session
import glob
import requests
import zipfile
from io import BytesIO
from threading import Lock
from ta.trend import SMAIndicator
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import joblib
from filterpy.kalman import KalmanFilter
from utils_output import ensure_directory, copy_output, save_model_output


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

network_name = "bearish_neural_network"  # Имя модели
checkpoint_path_regular = os.path.join("/workspace/checkpoints/bearish", f"{network_name}_checkpoint_epoch_{{epoch:02d}}.h5")
checkpoint_path_best = os.path.join("/workspace/checkpoints/bearish", f"{network_name}_best_model.h5")

# Имя файла для сохранения модели
nn_model_filename = os.path.join("/workspace/saved_models/bearish", 'bearish_nn_model.h5')
log_file = os.path.join("/workspace/logs", "training_log_bearish_nn.txt")


def save_logs_to_file(log_message):
    with open(log_file, 'a') as log_f:
        log_f.write(f"{datetime.now()}: {log_message}\n")
        
        

def cleanup_training_files():
    """
    Удаляет файлы с обучающими данными после завершения обучения нейросети.
    """
    files_to_delete = glob.glob("binance_data*.csv")  # Ищем все файлы данных
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logging.info(f"🗑 Удалён файл: {file_path}")
        except Exception as e:
            logging.error(f"⚠ Ошибка удаления {file_path}: {e}")
            
        
        
def calculate_cross_coin_features(data_dict):
    """
    Для скальпинга уменьшаем окна rolling с 30 до 15, а rolling(10) до 5,
    чтобы быстрее реагировать на краткосрочные колебания.
    """
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
    """
    Укорачиваем окно rolling(10) до rolling(5), а накопление – с 3 до 2,
    чтобы сигналы подтверждения были более быстрыми.
    """
    # CHANGED FOR SCALPING
    data['volume_trend_conf'] = np.where(
        (data['close'] > data['close'].shift(1)) & 
        (data['volume'] > data['volume'].rolling(5).mean()),  # было 10
        1,
        np.where(
            (data['close'] < data['close'].shift(1)) & 
            (data['volume'] > data['volume'].rolling(5).mean()),  # было 10
            -1,
            0
        )
    )
    # CHANGED FOR SCALPING
    data['volume_strength'] = (
        data['volume'] / data['volume'].rolling(5).mean()
    ) * data['volume_trend_conf']  # было 10
    
    # CHANGED FOR SCALPING
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
    kf.R = 2  # Понижено для уменьшения влияния шума измерения
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
    

# Кастомная функция потерь для медвежьего рынка, ориентированная на минимизацию убытков
def custom_profit_loss(y_true, y_pred):
    """
    Функция потерь для медвежьего рынка.
    y_true: тензор истинных меток, где 0 = Hold, 1 = Buy, 2 = Sell.
    y_pred: тензор предсказаний (распределение вероятностей) с shape (batch_size, 3).
    
    Преобразуем y_pred в скалярное предсказание, вычислив взвешенную сумму по классам.
    """
    # Преобразование распределения softmax в скалярное предсказание:
    # Здесь веса соответствуют номерам классов: 0.0 для Hold, 1.0 для Buy, 2.0 для Sell.
    class_weights = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    # При tensordot по оси=1 получаем результат shape (batch_size,)
    y_pred_scalar = tf.tensordot(y_pred, class_weights, axes=1)
    
    # Приводим y_true к float32 и убеждаемся, что его shape = (batch_size,)
    y_true = tf.cast(y_true, dtype=tf.float32)
    # Вычисляем разницу между скалярным предсказанием и истинным значением
    diff = y_pred_scalar - y_true
    
    # Логарифмический коэффициент для усиления больших ошибок
    log_factor = tf.math.log1p(tf.abs(diff) + 1e-7)
    
    # Параметры штрафов (настраиваются эмпирически)
    false_long_penalty = 1.0   # штраф за ложное срабатывание Buy (если y_true == 0, а y_pred_scalar > 0.5)
    false_short_penalty = 1.0  # штраф за ложное срабатывание Sell (если y_true == 0, а y_pred_scalar < 0.2)
    missed_rally_penalty = 2.0  # штраф за пропущенный Buy (если y_true == 1, а y_pred_scalar <= 0.5)
    missed_drop_penalty = 2.0   # штраф за пропущенный Sell (если y_true == 2, а y_pred_scalar >= 0.2)
    
    # CASE A: Если истинное значение Hold (0)
    loss_hold = tf.where(
        y_pred_scalar > 0.5,
        false_long_penalty * tf.abs(diff) * log_factor,
        tf.where(
            y_pred_scalar < 0.2,
            false_short_penalty * tf.abs(diff) * log_factor,
            tf.abs(diff) * log_factor
        )
    )
    
    # CASE B: Если истинное значение Buy (1)
    loss_buy = tf.where(
        y_pred_scalar <= 0.5,
        missed_rally_penalty * tf.abs(diff) * log_factor,
        tf.abs(diff) * log_factor
    )
    
    # CASE C: Если истинное значение Sell (2)
    loss_sell = tf.where(
        y_pred_scalar >= 0.2,
        missed_drop_penalty * tf.abs(diff) * log_factor,
        tf.abs(diff) * log_factor
    )
    
    # Объединяем случаи в зависимости от истинной метки
    base_loss = tf.where(
        tf.equal(y_true, 0.0),
        loss_hold,
        tf.where(
            tf.equal(y_true, 1.0),
            loss_buy,
            loss_sell
        )
    )
    
    # Штраф за неуверенные предсказания: если предсказание находится между 0.3 и 0.7
    uncertainty_penalty = tf.where(
        tf.logical_and(y_pred_scalar > 0.3, y_pred_scalar < 0.7),
        0.5 * tf.abs(diff) * log_factor,
        0.0
    )
    
    # Штраф за задержку реакции:
    batch_size = tf.shape(diff)[0]
    time_indices = tf.cast(tf.range(batch_size), tf.float32)
    time_penalty = 0.1 * tf.abs(diff) * time_indices / tf.cast(batch_size, tf.float32)
    
    # Штраф за транзакционные издержки: резкие изменения предсказаний между соседними точками
    transaction_cost = 0.001 * tf.reduce_sum(tf.abs(y_pred_scalar[1:] - y_pred_scalar[:-1]))
    
    total_loss = tf.reduce_mean(base_loss + uncertainty_penalty + time_penalty) + transaction_cost
    
    return total_loss


# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        return K.sum(output, axis=1)


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


# Получение исторических данных
def get_historical_data(symbols, bearish_periods, interval="1m", save_path="/workspace/data/binance_data_bearish.csv"
):
    """
    Скачивает исторические данные с Binance (архив) и сохраняет в один CSV-файл.

    :param symbols: список торговых пар (пример: ['BTCUSDC', 'ETHUSDC'])
    :param bearish_periods: список словарей с периодами (пример: [{"start": "2019-01-01", "end": "2019-12-31"}])
    :param interval: временной интервал (по умолчанию "1m" - 1 минута)
    :param save_path: путь к файлу для сохранения CSV (по умолчанию 'binance_data_bearish.csv')
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
        futures = [executor.submit(download_and_process, symbol, period) for symbol in symbols for period in bearish_periods]
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
            for chunk in pd.read_csv(save_path,
                                     index_col='timestamp',
                                     parse_dates=['timestamp'],
                                     on_bad_lines='skip',
                                     chunksize=CHUNK_SIZE):
                chunk = chunk.reset_index(drop=False)
                if 'timestamp' not in chunk.columns:
                    if 'index' in chunk.columns:
                        chunk.rename(columns={'index': 'timestamp'}, inplace=True)
                # Приведение числовых столбцов к числовому типу
                numeric_cols = [
                    "open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
                ]
                for col in numeric_cols:
                    if col in chunk.columns:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', utc=True)
                chunk = chunk.dropna(subset=['timestamp'])
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
                        # Приведение числовых столбцов к числовому типу
                        numeric_cols = [
                            "open", "high", "low", "close", "volume",
                            "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
                        ]
                        for col in numeric_cols:
                            if col in chunk.columns:
                                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
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
    
    Parameters:
        data (pd.DataFrame): Исходные данные с временной меткой в колонке 'timestamp'.
    
    Returns:
        pd.DataFrame: Агрегированные данные.
    """
    # Проверяем наличие 'timestamp'
    if 'timestamp' not in data.columns:
        logging.error("Колонка 'timestamp' отсутствует в данных для агрегации.")
        if isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = data.index
            logging.info("Индекс преобразован в колонку 'timestamp'.")
        else:
            raise ValueError("Колонка 'timestamp' отсутствует в данных.")

    # Убедитесь, что временной индекс установлен
    data['timestamp'] = pd.to_datetime(data['timestamp'])  # Убедитесь, что временные метки в формате datetime
    data = data.set_index('timestamp')

    # Агрегация данных
    data = data.resample('2T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()  # Удаление строк с пропущенными значениями

    logging.info(f"Агрегация завершена. Размер данных: {len(data)} строк.")
    return data'''


def adjust_target(data, threshold=-0.0005, trend_window=50):
    """
    Изменение целевой переменной для акцентирования на резких падениях.
    
    Parameters:
        data (pd.DataFrame): Данные с колонкой 'returns'.
        threshold (float): Порог падения (например, -0.05 для падений > 5%).
        
    Returns:
        pd.DataFrame: Обновленные данные с колонкой 'target'.
    """
    data['target'] = (data['returns'] < threshold).astype(int)
    data['trend'] = (data['close'] < data['close'].rolling(trend_window).mean()).astype(int)
    data['target'] = np.where(data['target'] + data['trend'] > 0, 1, 0)
    logging.info(f"Целевая переменная обновлена: {data['target'].value_counts().to_dict()}")
    return data

# Извлечение признаков
def extract_features(data):
    logging.info("Извлечение признаков для медвежьего рынка")
    data = data.copy()
    # Применяем фильтр Калмана, как и раньше
    data = remove_noise(data)

    # Базовые расчёты
    returns = data['close'].pct_change()
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
    data['cmf'] = ChaikinMoneyFlowIndicator(data['high'], data['low'], data['close'], data['volume']).chaikin_money_flow()
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




def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]


def add_clustering_feature(data):
    features_for_clustering = [
        'close', 'volume', 'rsi', 'macd', 'atr', 'sma_3',  'sma_5', 'sma_8', 'ema_3', 'ema_5','ema_8',
        'bb_width', 'macd_diff', 'obv', 'returns', 'log_returns'
    ]
    
    # Если данных много, берём сэмпл (например, 200k строк)
    max_for_kmeans = 200_000
    if len(data) > max_for_kmeans:
        sample_df = data.sample(n=max_for_kmeans, random_state=42)
    else:
        sample_df = data
        
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(sample_df[features_for_clustering])
    # Применяем модель ко всему DataFrame
    data['cluster'] = kmeans.fit_predict(data[features_for_clustering])
    return data

def prepare_data(data):
    logging.info("Начало подготовки данных")
    
    # Проверка на пустые данные
    if data.empty:
        raise ValueError("Входные данные пусты")
        
    logging.info(f"Исходная форма данных: {data.shape}")
    
    # Убедимся, что временной индекс установлен
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Данные не содержат временного индекса или колонки 'timestamp'.")
    
    def process_chunk(df_chunk):
        df_chunk = extract_features(df_chunk)
        df_chunk = remove_outliers(df_chunk)
        df_chunk = add_clustering_feature(df_chunk)
        return df_chunk

    # Применяем обработку по чанкам, если датасет очень большой
    data = apply_in_chunks(data, process_chunk, chunk_size=100000)
    logging.info(f"После обработки (извлечение признаков, удаление выбросов, кластеризация): {data.shape}")
    
    # Список признаков - ИСКЛЮЧАЕМ timestamp и другие нечисловые колонки
    features = [col for col in data.columns if col not in ['target', 'timestamp'] and pd.api.types.is_numeric_dtype(data[col])]
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

def balance_classes(X, y, max_for_smote=300_000):
    """
    Балансировка классов с поддержкой numpy и pandas.

    Параметры:
    X: numpy.ndarray или pandas.DataFrame — признаки
    y: numpy.ndarray или pandas.Series — метки
    max_for_smote: int — максимальный размер выборки для SMOTETomek

    Возвращает:
    X_resampled, y_resampled
    """
    logging.info("Начало балансировки классов")
    logging.info(f"Размеры данных до балансировки: X={getattr(X, 'shape', None)}, y={getattr(y, 'shape', None)}")
    logging.info(f"Уникальные классы в y: {np.unique(y, return_counts=True)}")

    # Определяем тип и делаем подвыборку, если нужно
    if isinstance(X, pd.DataFrame):
        # Для pandas DataFrame
        if len(X) > max_for_smote:
            X_sample = X.sample(n=max_for_smote, random_state=42)
            # Поддерживаем pandas.Series или numpy.ndarray для y
            if isinstance(y, pd.Series):
                y_sample = y.loc[X_sample.index]
            else:
                # y — numpy.ndarray, используем iloc по индексам DataFrame
                y_sample = y[X_sample.index.to_numpy()]
        else:
            X_sample = X
            y_sample = y
    else:
        # Для numpy.ndarray или других типов
        n_samples = X.shape[0]
        if n_samples > max_for_smote:
            idx = np.random.choice(n_samples, size=max_for_smote, replace=False)
            X_sample = X[idx]
            y_sample = y[idx]
        else:
            X_sample = X
            y_sample = y

    # Проверяем наличие данных
    if len(X_sample) == 0 or len(y_sample) == 0:
        raise ValueError("Данные для балансировки пусты. Проверьте входные данные.")

    # Применяем SMOTETomek
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_sample, y_sample)

    logging.info(f"Размеры данных после балансировки: X={getattr(X_resampled, 'shape', None)}, y={getattr(y_resampled, 'shape', None)}")
    return X_resampled, y_resampled


def train_xgboost_on_embeddings(X_emb, y):
    """
    Обучает XGBoost-классификатор на эмбеддингах, извлечённых из нейросети.
    Предполагается, что целевая переменная y принимает значения из {0, 1, 2}.
    Эта функция адаптирована для медвежьего рынка, но при необходимости можно
    изменить параметры для лучшей подгонки под ваши данные.
    """
    logging.info("Обучение XGBoost на эмбеддингах...")
    xgb_model = XGBClassifier(
        objective='multi:softprob',  # многоклассовая задача
        n_estimators=10,
        max_depth=3,
        learning_rate=0.01,
        random_state=42,
        num_class=3  # 3 класса
    )
    xgb_model.fit(X_emb, y)
    logging.info("XGBoost обучен на эмбеддингах.")
    return xgb_model



def prepare_timestamp_column(data):
    """
    Гарантированно создаёт столбец 'timestamp' для DataFrame.
    
    Алгоритм:
      1. Если столбец 'timestamp' уже присутствует, он удаляется.
      2. Выполняется data.reset_index(), чтобы преобразовать индекс (который должен быть DatetimeIndex)
         в столбец. Если индекс имеет собственное имя (например, 'timestamp' или другое), его переименовываем в 'timestamp'.
      3. Результат возвращается — DataFrame, где столбец 'timestamp' точно присутствует.
      
    Этот метод гарантированно создаёт нужный столбец без риска дублирования.
    """
    logging.info("Убеждаемся, что столбец 'timestamp' присутствует, используя reset_index().")
    
    # Если 'timestamp' уже есть, удаляем его, чтобы избежать дублирования
    if 'timestamp' in data.columns:
        logging.info("Обнаружен столбец 'timestamp'. Удаляем его для корректного создания нового.")
        data = data.drop(columns=['timestamp'])
    
    # Сбрасываем индекс: если индекс является DatetimeIndex, то он превратится в колонку
    data = data.reset_index()
    
    # Если после сброса индекс назывался не 'timestamp', переименовываем его
    if 'timestamp' not in data.columns:
        # Если индекс сброшен как 'index', то переименовываем его в 'timestamp'
        if 'index' in data.columns:
            data.rename(columns={'index': 'timestamp'}, inplace=True)
            logging.info("Колонка 'index' переименована в 'timestamp'.")
        else:
            # Если ни 'timestamp', ни 'index' не присутствуют, создаём столбец на основе текущего индекса
            data['timestamp'] = data.index
            logging.info("Столбец 'timestamp' создан из индекса.")
    else:
        # Приводим существующий столбец к типу datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        logging.info("Столбец 'timestamp' приведён к типу datetime.")
    
    # Дополнительно можно переставить столбец 'timestamp' на первую позицию, если это нужно
    cols = list(data.columns)
    if cols[0] != 'timestamp':
        cols.insert(0, cols.pop(cols.index('timestamp')))
        data = data[cols]
        logging.info("Столбец 'timestamp' переставлен в начало DataFrame.")
    
    return data


def build_bearish_neural_network(data, model_filename):
    """
    Обучает нейронную сеть для медвежьего рынка с корректной обработкой временной метки.

    Parameters:
        data (pd.DataFrame): Данные для обучения.
        model_filename (str): Имя файла для сохранения модели (.h5).

    Returns:
        ensemble_model: dict with keys nn_model, xgb_model, feature_extractor,
                        ensemble_weight_nn, ensemble_weight_xgb
        scaler: the fitted RobustScaler
    """
    logging.info("Начало обучения медвежьей нейросети")

    # --- параметры сети и пути к чекпоинтам
    network_name = "bearish_nn"
    checkpoint_path_regular = f"checkpoints/{network_name}_checkpoint_epoch_{{epoch:02d}}.h5"
    checkpoint_path_best    = f"checkpoints/{network_name}_best.h5"

    # 1) Гарантированно создаём столбец timestamp
    data = prepare_timestamp_column(data)

    # 2) Выбор числовых признаков
    features   = [c for c in data.columns if c not in ['target','timestamp'] and pd.api.types.is_numeric_dtype(data[c])]
    X_df       = data[features].astype(float).copy()
    y_series   = data['target'].astype(int).copy()
    logging.info(f"Исходные размеры: X={X_df.shape}, y={y_series.shape}")

    # 3) Удаляем NaN и бесконечности
    mask = X_df.notnull().all(axis=1) & np.isfinite(X_df).all(axis=1)
    X_df, y_series = X_df.loc[mask], y_series.loc[mask]
    logging.info(f"После очистки: X={X_df.shape}, y={y_series.shape}")

    # 4) Балансировка классов
    X_bal, y_bal = balance_classes(X_df, y_series)
    logging.info(f"После балансировки: X={X_bal.shape}, y={y_bal.shape}")

    # 5) Разделение на train/validation
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    logging.info(f"Train: {X_train_df.shape}, Val: {X_val_df.shape}")

    # 6) Масштабирование
    scaler       = RobustScaler()
    X_train_np   = scaler.fit_transform(X_train_df)
    X_val_np     = scaler.transform(X_val_df)

    # 7) reshape для LSTM (samples, timesteps=1, features)
    X_train = X_train_np.reshape((X_train_np.shape[0], 1, X_train_np.shape[1]))
    X_val   = X_val_np.reshape((X_val_np.shape[0], 1, X_val_np.shape[1]))
    logging.info(f"После reshape: X_train={X_train.shape}, X_val={X_val.shape}")

    # 8) Создание tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

    # 9) Метрики HFT
    def hft_metrics(y_true, y_pred):
        rt   = tf.reduce_mean(tf.abs(y_pred[1:] - y_pred[:-1]))
        stab = tf.reduce_mean(tf.abs(y_pred[2:] - 2*y_pred[1:-1] + y_pred[:-2]))
        return rt, stab

    def profit_ratio(y_true, y_pred):
        succ  = tf.reduce_sum(tf.where(tf.logical_and(y_true>=1, y_pred>=0.5), 1.0, 0.0))
        fals = tf.reduce_sum(tf.where(tf.logical_and(y_true==0, y_pred>=0.5), 1.0, 0.0))
        return succ / (fals + K.epsilon())

    # 10) Инициализация стратегии
    os.makedirs('checkpoints', exist_ok=True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Используется MirroredStrategy (GPU)")
    else:
        strategy = tf.distribute.get_strategy()
        logging.info("Используется стандартная стратегия (CPU)")

    # 11) Построение и компиляция модели
    with strategy.scope():
        inp = Input(shape=(X_train.shape[1], X_train.shape[2]))
        def branch(x):
            x = LSTM(256, return_sequences=True)(x)
            x = BatchNormalization()(x)
            return Dropout(0.2)(x)
        b1, b2, b3 = branch(inp), branch(inp), branch(inp)
        x = Add()([b1, b2, b3])
        x = LSTM(256, return_sequences=False)(x)
        x = BatchNormalization()(x); x = Dropout(0.3)(x)
        x = Dense(128, activation='relu', name='embedding_layer')(x)
        x = BatchNormalization()(x); x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x); x = Dropout(0.3)(x)
        out = Dense(3, activation='softmax')(x)
        model = Model(inp, out)
        model.compile(
            optimizer=Adam(1e-3),
            loss=custom_profit_loss,
            metrics=[hft_metrics, profit_ratio]
        )

    # 12) Callbacks и обучение
    cb_reg  = ModelCheckpoint(checkpoint_path_regular, save_weights_only=True, verbose=1)
    cb_best = ModelCheckpoint(checkpoint_path_best, save_weights_only=True, save_best_only=True,
                                monitor='val_loss', mode='min', verbose=1)
    cb_lr   = ReduceLROnPlateau('val_loss', factor=0.5, patience=3, verbose=1)
    cb_es   = EarlyStopping('val_loss', patience=15, restore_best_weights=True)
    class_weights = {0:1.0, 1:2.0, 2:3.0}
    history = model.fit(train_ds, epochs=200, validation_data=val_ds,
                        class_weight=class_weights,
                        callbacks=[cb_reg, cb_best, cb_lr, cb_es])

    # 13) Удаление старых чекпоинтов
    for f in glob.glob(f'checkpoints/{network_name}_checkpoint_epoch_*.h5'):
        if not f.endswith(f'{network_name}_best.h5'):
            os.remove(f)

    # 14) Ансамблирование с XGBoost
    feat_ext = Model(model.input, model.get_layer('embedding_layer').output)
    emb_tr = feat_ext.predict(X_train)
    emb_vl = feat_ext.predict(X_val)
    xgb_m = train_xgboost_on_embeddings(emb_tr, y_train)
    nn_pred = model.predict(X_val)
    xgb_pred = xgb_m.predict_proba(emb_vl)
    ens = 0.5*nn_pred + 0.5*xgb_pred
    cls = np.argmax(ens, axis=1)
    logging.info(f"F1 ensemble bearish: {f1_score(y_val, cls, average='weighted')}")

    # 15) Сохранение моделей
    model.save(model_filename)
    joblib.dump(xgb_m, os.path.join(os.path.dirname(model_filename), 'xgb_bearish.pkl'))

    return {"nn_model": model, "xgb_model": xgb_m,
            "feature_extractor": feat_ext,
            "ensemble_weight_nn": 0.5, "ensemble_weight_xgb": 0.5}, scaler


if __name__ == "__main__":
    try:
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
        
        logging.info("🔄 Загрузка данных для медвежьего периода...")
        data_dict = load_bearish_data(symbols, bearish_periods, interval="1m")
        if not data_dict:
            raise ValueError("❌ Ошибка: Данные не были загружены!")
        data = pd.concat(data_dict.values(), ignore_index=False)
        if data.empty:
            raise ValueError("❌ Ошибка: После загрузки данные пусты!")
        # Здесь обязательно вызываем prepare_timestamp_column, чтобы создать столбец 'timestamp'
        data = prepare_timestamp_column(data)
        logging.info(f"ℹ После подготовки столбца, колонки: {data.columns.tolist()}")
        logging.info(f"📈 Размер данных после загрузки: {data.shape}")
        logging.info("🛠 Извлечение признаков из данных...")
        data = extract_features(data)
        data.dropna(inplace=True)
        data = data.loc[:, ~data.columns.duplicated()]
        if data.empty:
            raise ValueError("❌ Ошибка: После очистки данные пусты!")
        logging.info("🚀 Начало обучения модели для медвежьего рынка...")
        build_bearish_neural_network(data)
    except Exception as e:
        logging.error(f"❌ Ошибка во время выполнения программы: {e}")
    finally:
        logging.info("🗑 Очистка временных файлов...")
        cleanup_training_files()
        logging.info("🧹 Очистка сессии TensorFlow...")
        clear_session()
        logging.info("✅ Программа завершена.")
    sys.exit(0)
