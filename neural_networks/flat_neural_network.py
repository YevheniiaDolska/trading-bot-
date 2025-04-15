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
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
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
import glob
from filterpy.kalman import KalmanFilter
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from threading import Lock
import requests
import zipfile
from io import BytesIO
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import xgboost as xgb  # если потребуется
import joblib
from utils_output import ensure_directory, copy_output, save_model_output



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

required_dirs = [
    "/workspace/logs",
    "/workspace/saved_models/flat",
    "/workspace/checkpoints/flat",
    "/workspace/data",
    "/workspace/output/flat_ensemble"
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


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Имя файла для сохранения модели
nn_model_filename = os.path.join("/workspace/saved_models/flat", 'flat_nn_model.h5')
log_file = os.path.join("/workspace/logs", "training_log_flat_nn.txt")


network_name = "flat_neural_network"  # Имя модели
checkpoint_path_regular = os.path.join("/workspace/checkpoints/flat", f"{network_name}_checkpoint_epoch_{{epoch:02d}}.h5")
checkpoint_path_best = os.path.join("/workspace/checkpoints/flat", f"{network_name}_best_model.h5")


def save_logs_to_file(log_message):
    with open(log_file, 'a') as log_f:
        log_f.write(f"{datetime.now()}: {log_message}\n")
        
        
def calculate_cross_coin_features(data_dict):
    btc_data = data_dict['BTCUSDC']
    for symbol, df in data_dict.items():
        # CHANGED FOR SCALPING
        df['btc_corr'] = df['close'].rolling(15).corr(btc_data['close'])
        df['rel_strength_btc'] = df['close'].pct_change() - btc_data['close'].pct_change()
        
        # CHANGED FOR SCALPING
        df['beta_btc'] = df['close'].pct_change().rolling(15).cov(btc_data['close'].pct_change()) / \
                         btc_data['close'].pct_change().rolling(15).var()
        
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
    data['volume_strength'] = (
        data['volume'] / data['volume'].rolling(5).mean()
    ) * data['volume_trend_conf']
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(2).sum()
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
        if 'clean_returns' not in data.columns:
            raise ValueError("❌ ERROR: 'clean_returns' не был создан в remove_noise()!")

        # После remove_noise()
        df = remove_noise(df)

        if 'clean_returns' not in df.columns:
            raise ValueError("❌ ERROR: 'clean_returns' пропал после remove_noise()!")

        # Перед extract_features()
        if 'clean_returns' not in data.columns:
            data['clean_returns'] = 0.0
            print("🔧 WARNING: 'clean_returns' отсутствовал, добавлен вручную!")

        
        data_dict[symbol] = df
    
    
    return data_dict

# Кастомная функция потерь для флэтового рынка, ориентированная на минимизацию волатильности и частоту сделок
def custom_profit_loss(y_true, y_pred):
    """
    Ваш "diff"-подход (BUY/SELL/HOLD) в многоклассовом варианте:
      - y_true: (batch,) ∈ {0=HOLD,1=SELL,2=BUY}
      - y_pred: (batch,3), softmax: [p(HOLD), p(SELL), p(BUY)]
      
    Логика (аналог вашей):
      diff = y_pred - y_true_onehot
      log_factor = log1p( sum(|diff|) )  [на сэмпл]
      underestimation_penalty = (y_true_onehot > y_pred)? (..)^2 : 0
      overestimation_penalty  = (y_true_onehot < y_pred)? (..)^2 : 0
      gain = max(diff,0)
      loss = abs(min(diff,0))
      total_loss = mean( loss*2 + log_factor*1.5 + underest*3 - gain*1.5 + overest*2 )
    """
    # Превращаем метки в one-hot
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)  # (batch,3)

    # diff (batch,3)
    diff = y_pred - y_true_onehot

    # Лог-фактор (для крупных ошибок)
    eps = 1e-7
    # Сумма |diff| по классам => скаляр на сэмпл
    diff_magnitude = tf.reduce_sum(tf.abs(diff), axis=1)
    log_factor = tf.math.log1p(diff_magnitude + eps)  # (batch,)

    # Недооценка (когда y_true_onehot > y_pred)
    underestimation_penalty = tf.where(y_true_onehot > y_pred,
                                       tf.square(y_true_onehot - y_pred), 0.0)

    # Переоценка (когда y_true_onehot < y_pred)
    overestimation_penalty = tf.where(y_true_onehot < y_pred,
                                      tf.square(y_pred - y_true_onehot), 0.0)

    # gain = max(diff,0), loss = abs(min(diff,0)) (покомпонентно)
    gain = tf.math.maximum(diff, 0.0)      # (batch,3)
    negative_part = tf.math.minimum(diff, 0.0)
    loss_ = tf.math.abs(negative_part)     # (batch,3)

    # Сборка частей
    # Пер-классная сумма: loss_*2 + underest*3 - gain*1.5 + overest*2
    per_class_term = (
        loss_ * 2.0 +
        underestimation_penalty * 3.0 -
        gain * 1.5 +
        overestimation_penalty * 2.0
    )  # shape=(batch,3)

    # Складываем по классам
    per_sample_sum = tf.reduce_sum(per_class_term, axis=1)  # (batch,)

    # Добавляем log_factor *1.5
    total = per_sample_sum + log_factor * 1.5

    # Среднее по батчу
    return tf.reduce_mean(total)



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
        e = tf.math.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * tf.expand_dims(a, -1)
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


def get_historical_data(symbols, flat_periods, interval="1m", save_path="/workspace/data/binance_data_flat.csv"
):
    """
    Скачивает исторические данные с Binance (архив) и сохраняет в один CSV-файл.

    :param symbols: список торговых пар (пример: ['BTCUSDC', 'ETHUSDC'])
    :param flat_periods: список словарей с периодами (пример: [{"start": "2019-01-01", "end": "2019-12-31"}])
    :param interval: временной интервал (по умолчанию "1m" - 1 минута)
    :param save_path: путь к файлу для сохранения CSV (по умолчанию 'binance_data_flat.csv')
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
        futures = [executor.submit(download_and_process, symbol, period) for symbol in symbols for period in flat_periods]
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


def load_flat_data(symbols, flat_periods, interval="1m", save_path="binance_data_flat.csv"):
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
            executor.submit(get_historical_data, [symbol], flat_periods, interval, save_path): symbol
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
    
    # Убедитесь, что временной индекс установлен
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")

    # Агрегация данных
    aggregated_data = data.resample('2T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logging.info(f"Агрегация завершена, размер данных: {len(aggregated_data)} строк")
    return aggregated_data'''



def smooth_data(data, window=5):
    """
    Сглаживание временного ряда с помощью скользящего среднего.
    
    Parameters:
        data (pd.Series): Исходный временной ряд.
        window (int): Размер окна сглаживания.
        
    Returns:
        pd.Series: Сглаженные данные.
    """
    return data.rolling(window=window, min_periods=1).mean()



def extract_features(data):
    logging.info("Извлечение признаков для флэтового рынка")
    data = data.copy()
    
    # 1. Базовые метрики
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # 2. Метрики диапазона
    data['range_width'] = data['high'] - data['low']
    # Группируем rolling для range_width с окном 10 и 20
    roll_range_10 = data['range_width'].rolling(10)
    data['range_stability'] = roll_range_10.std()
    roll_range_20 = data['range_width'].rolling(20)
    data['range_mean_20'] = roll_range_20.mean()
    data['range_ratio'] = data['range_width'] / data['range_mean_20']
    data['price_in_range'] = (data['close'] - data['low']) / data['range_width']
    
    # 3. Быстрые индикаторы для HFT
    data['sma_3'] = SMAIndicator(data['close'], window=3).sma_indicator()
    data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema_8'] = data['close'].ewm(span=8, adjust=False).mean()
    if 'clean_returns' in data.columns:
        data['clean_volatility'] = data['clean_returns'].rolling(20).std()
    
    # 4. Короткие осцилляторы
    data['rsi_3'] = RSIIndicator(data['close'], window=3).rsi()
    data['rsi_5'] = RSIIndicator(data['close'], window=5).rsi()
    stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=5)
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()
    
    # 5. Волатильность малых периодов
    bb = BollingerBands(data['close'], window=10)
    data['bb_width'] = bb.bollinger_wband()
    data['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    data['atr_5'] = AverageTrueRange(data['high'], data['low'], data['close'], window=5).average_true_range()
    
    # 6. Объемные показатели – группируем rolling по volume с окном 10
    roll_volume_10 = data['volume'].rolling(10)
    data['volume_ma'] = roll_volume_10.mean()
    data['volume_std'] = roll_volume_10.std()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['volume_stability'] = data['volume_std'] / data['volume_ma']
    
    # 7. Индикаторы пробоя
    data['breakout_intensity'] = abs(data['close'] - data['close'].shift(1)) / data['range_width']
    data['false_breakout'] = (data['high'] > data['high'].shift(1)) & (data['close'] < data['close'].shift(1))
    
    # 8. Микро-паттерны
    data['micro_trend'] = np.where(
        data['close'] > data['close'].shift(1), 1,
        np.where(data['close'] < data['close'].shift(1), -1, 0)
    )
    data['micro_trend_change'] = (data['micro_trend'] != data['micro_trend'].shift(1)).astype(int)
    
    # 9. Целевая переменная для флэтового рынка
    volatility = data['returns'].rolling(20).std()
    avg_volatility = volatility.rolling(100).mean()
    threshold = 0.0002
    data['target'] = np.where(
        (data['returns'].shift(-1) > threshold) &
        (data['volume'] > data['volume_ma']) &
        (data['rsi_3'] < 40) &
        (data['bb_position'] < 0.3),
        2,
        np.where(
            (data['returns'].shift(-1) < -threshold) &
            (data['volume'] > data['volume_ma']) &
            (data['rsi_3'] > 60) &
            (data['bb_position'] > 0.7),
            1,
            0
        )
    )
    
    # Создаем словарь признаков
    features = {}
    
    # Базовые признаки (все, что уже рассчитано)
    for col in data.columns:
        if col not in ['target', 'market_type']:
            features[col] = data[col]
    
    # Добавляем межмонетные признаки, если они есть
    if 'btc_corr' in data.columns:
        features['btc_corr'] = data['btc_corr']
    if 'rel_strength_btc' in data.columns:
        features['rel_strength_btc'] = data['rel_strength_btc']
    if 'beta_btc' in data.columns:
        features['beta_btc'] = data['beta_btc']
            
    # Добавляем признаки подтверждения объемом, если они есть
    if 'volume_strength' in data.columns:
        features['volume_strength'] = data['volume_strength']
    if 'volume_accumulation' in data.columns:
        features['volume_accumulation'] = data['volume_accumulation']
    
    # Добавляем очищенные от шума признаки, если они есть
    if 'clean_returns' in data.columns:
        features['clean_returns'] = data['clean_returns']
        
    # Преобразуем признаки в DataFrame
    features_df = pd.DataFrame(features)
    
    return data.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

def add_clustering_feature(data):
    features_for_clustering = [
        'close', 'volume', 'rsi', 'macd', 'atr', 'sma_3', 'ema_5', 'ema_8',
        'bb_width', 'macd_diff', 'obv', 'returns', 'log_returns'
    ]
    max_for_kmeans = 200_000
    if len(data) > max_for_kmeans:
        sample_df = data.sample(n=max_for_kmeans, random_state=42)
    else:
        sample_df = data
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(sample_df[features_for_clustering])
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

    # Извлечение признаков
    logging.info(f"Столбцы в data перед extract_features: {list(data.columns)}")
    
    # 🚀 Перед extract_features() проверяем, есть ли clean_returns
    missing_columns = [col for col in ['clean_returns'] if col not in data.columns]
    if missing_columns:
        print(f"🔴 ERROR: Эти колонки пропали перед extract_features(): {missing_columns}")
        print("🔧 Добавляем clean_returns вручную...")
        data['clean_returns'] = 0.0  # Гарантированно создаём

    print(f"✅ Колонки в data перед extract_features: {list(data.columns)}")

    def process_chunk(df_chunk):
        df_chunk = extract_features(df_chunk)
        df_chunk = remove_outliers(df_chunk)
        df_chunk = add_clustering_feature(df_chunk)
        return df_chunk

    # Применяем обработку по чанкам, если датасет очень большой
    data = apply_in_chunks(data, process_chunk, chunk_size=100000)
    logging.info(f"После обработки (извлечение признаков, удаление выбросов, кластеризация): {data.shape}")


    # Список признаков
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
    logging.info("Начало балансировки классов")
    logging.info(f"Размеры данных до балансировки: X={X.shape}, y={y.shape}")
    logging.info(f"Уникальные классы в y: {np.unique(y, return_counts=True)}")
    
    max_for_smote = 300_000
    if len(X) > max_for_smote:
        X_sample = X.sample(n=max_for_smote, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Данные для балансировки пусты. Проверьте исходные данные и фильтры.")

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_sample, y_sample)

    logging.info(f"Размеры данных после балансировки: X={X_resampled.shape}, y={y_resampled.shape}")
    return X_resampled, y_resampled


def check_feature_quality(X, y):
    """
    Вычисляет важность признаков с помощью SelectKBest (f_classif)
    и логирует топ-10 признаков по значимости.
    """
    logging.info("Проверка качества признаков...")
    logging.info(f"Форма X: {X.shape}")
    logging.info(f"Количество пропущенных значений в X: {np.isnan(X).sum()}")
    
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
    """
    Обучает XGBoost классификатор на эмбеддингах, извлечённых из нейросети.
    Предполагается, что целевая переменная имеет 3 класса.
    """
    logging.info("Обучение XGBoost на эмбеддингах...")
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1
    )
    xgb_model.fit(X_emb, y)
    logging.info("XGBoost обучен.")
    return xgb_model

def ensemble_predict(nn_model, xgb_model, feature_extractor, X_seq, weight_nn=0.5, weight_xgb=0.5):
    """
    Выполняет предсказание на последовательностях X_seq, комбинируя прогнозы нейросети и XGBoost
    через взвешенное голосование.
    
    Параметры:
      - nn_model: обученная нейросеть (возвращает вероятности для каждого класса).
      - xgb_model: обученная XGBoost-модель.
      - feature_extractor: модель, извлекающая эмбеддинги из nn_model.
      - X_seq: входные последовательности для предсказания.
      - weight_nn, weight_xgb: веса для объединения (суммируются до 1).
      
    Возвращает:
      - Итоговые предсказания (классы из {0, 1, 2}).
    """
    nn_pred_proba = nn_model.predict(X_seq)
    embeddings = feature_extractor.predict(X_seq)
    xgb_pred_proba = xgb_model.predict_proba(embeddings)
    final_pred_proba = weight_nn * nn_pred_proba + weight_xgb * xgb_pred_proba
    final_pred_classes = np.argmax(final_pred_proba, axis=1)
    return final_pred_classes


def build_flat_neural_network(data, model_filename):
    """
    Обучение нейросети для флэтового рынка.

    Parameters:
        data (pd.DataFrame): Данные для обучения.
        model_filename (str): Имя файла для сохранения модели.

    Returns:
        ensemble_model (dict): Словарь с ключами "nn_model", "xgb_model", "feature_extractor",
                               "ensemble_weight_nn", "ensemble_weight_xgb" и "scaler".
    """
    # Создаем директории для чекпоинтов
    os.makedirs("checkpoints", exist_ok=True)
    
    network_name = "flat_neural_network"
    checkpoint_path_regular = f"checkpoints/{network_name}_checkpoint_epoch_{{epoch:02d}}.h5"
    checkpoint_path_best = f"checkpoints/{network_name}_best_model.h5"
    
    if os.path.exists("flat_neural_network.h5"):
        try:
            model = load_model("flat_neural_network.h5", custom_objects={"custom_profit_loss": custom_profit_loss})
            logging.info("Обученная модель загружена из 'flat_neural_network.h5'. Пропускаем обучение.")
            # Если модель уже загружена, возвращаем её вместе с сохраненным масштабировщиком (если он есть)
            return {"nn_model": model}, None
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            logging.info("Начинаем обучение с нуля.")
    else:
        logging.info("Сохраненная модель не найдена. Начинаем обучение с нуля.")
    
    
    logging.info("Начало обучения нейросети для флэтового рынка")
    
    # Выделяем признаки и целевые переменные
    features = [col for col in data.columns if col not in ['target', 'symbol', 'timestamp'] 
                and pd.api.types.is_numeric_dtype(data[col])]
    
    # Логирование для отладки
    logging.info(f"Выбранные признаки: {features}")
    logging.info(f"Типы данных признаков:\n{data[features].dtypes}")
    
    try:
        X = data[features].astype(float).values
        y = data['target'].astype(float).values
        
        logging.info(f"Форма X: {X.shape}, тип данных X: {X.dtype}")
        logging.info(f"Форма y: {y.shape}, тип данных y: {y.dtype}")
        
        # Проверка на наличие невалидных значений
        X_isvalid = np.isfinite(X)
        if not X_isvalid.all():
            invalid_count = np.sum(~X_isvalid)
            logging.warning(f"Найдено {invalid_count} невалидных значений в данных")
            
        # Удаление NaN и бесконечностей
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        y = y[mask]
        
        logging.info(f"После очистки - форма X: {X.shape}, форма y: {y.shape}")
        
    except Exception as e:
        logging.error(f"Ошибка при подготовке данных: {e}")
        raise

    # Дополнительный контроль качества: вывод статистики и оценка важности признаков
    X_df = pd.DataFrame(X, columns=features)
    logging.info("Статистика признаков до балансировки:")
    logging.info(X_df.describe().to_string())
    check_feature_quality(X, y)
    
    # Балансировка классов
    X_resampled, y_resampled = balance_classes(X, y)
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Масштабирование данных
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_scaled = scaler.transform(X_val).reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # Создание tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
    
    # Использование стратегии (теперь для GPU, без TPU)
    try:
        # Здесь можно заменить на MirroredStrategy для GPU
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Используется GPU стратегия: MirroredStrategy")
    except Exception as e:
        logging.error(f"Ошибка при инициализации GPU стратегии: {e}")
        strategy = tf.distribute.get_strategy()
        logging.info("Используем стандартную стратегию.")
    
    with strategy.scope():
        inputs = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))  # (timesteps, num_features)
        
        # Первая ветвь - анализ ценовых микродвижений
        x1 = LSTM(256, return_sequences=True)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        
        # Вторая ветвь - анализ объемов и волатильности
        x2 = LSTM(256, return_sequences=True)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Третья ветвь - анализ рыночного контекста
        x3 = LSTM(256, return_sequences=True)(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.3)(x3)
        
        # Объединение ветвей
        x = Add()([x1, x2, x3])
        
        # Основной LSTM для принятия решений
        x = LSTM(256, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Dense слои
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Добавляем слой эмбеддингов с именем "embedding_layer" для дальнейшего ансамблирования
        x = Dense(128, activation='relu', name="embedding_layer")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # 3 выхода: 0=HOLD, 1=SELL, 2=BUY
        outputs = Dense(3, activation='softmax')(x)
        
        model = tf.keras.models.Model(inputs, outputs)
        
        # Компилируем с нашей новой функцией потерь
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_profit_loss,
            metrics=[]
        )
        
        # Дополнительная метрика для контроля (flat trading metric)
        def flat_trading_metric(y_true, y_pred):
            true_range = tf.reduce_max(y_true) - tf.reduce_min(y_true)
            pred_range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
            range_error = tf.abs(true_range - pred_range)
            return range_error
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_profit_loss, metrics=[flat_trading_metric])
    
    # Попытка загрузить последний сохранённый промежуточный чекпоинт
    latest_checkpoint = glob.glob(f"checkpoints/{network_name}_checkpoint_epoch_*.h5")
    if latest_checkpoint:
        try:
            model.load_weights(latest_checkpoint[-1])
            logging.info(f"Загружены веса из последнего регулярного чекпоинта: {latest_checkpoint[-1]}")
        except Exception as e:
            logging.error(f"Ошибка загрузки регулярного чекпоинта: {e}")
    
    # Проверяем наличие лучшего чекпоинта
    if os.path.exists(checkpoint_path_best):
        try:
            model.load_weights(checkpoint_path_best)
            logging.info(f"Лучший чекпоинт найден: {checkpoint_path_best}. После обучения промежуточные чекпоинты будут удалены.")
        except:
            logging.info("Лучший чекпоинт пока не создан. Это ожидаемо, если обучение ещё не завершено.")
    
    # Настройка коллбеков
    checkpoint_every_epoch = ModelCheckpoint(
        filepath=checkpoint_path_regular,
        save_weights_only=True,
        save_best_only=False,
        verbose=1
    )
    
    checkpoint_best_model = ModelCheckpoint(
        filepath=checkpoint_path_best,
        save_weights_only=True,
        save_best_only=True,
        monitor='flat_trading_metric',
        mode='min',
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(log_dir=f"logs/{time.time()}")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, mode='min')
    early_stopping = EarlyStopping(
        monitor='val_flat_trading_metric',
        patience=5,
        restore_best_weights=True,
        mode='min'
    )
    
    # Настройка весов классов для несбалансированных данных
    class_weights = {
           0: 1.0,  # HOLD
           1: 2.5,  # SELL
           2: 2.5,  # BUY
    }
    
    # Обучение модели
    history = model.fit(
        train_dataset,
        epochs=1, #200
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=[
            early_stopping,
            checkpoint_every_epoch,
            checkpoint_best_model,
            tensorboard_callback,
            reduce_lr
        ]
    )
    
    logging.info("Очистка промежуточных чекпоинтов...")
    for checkpoint in glob.glob(f"checkpoints/{network_name}_checkpoint_epoch_*.h5"):
        if checkpoint != checkpoint_path_best:
            os.remove(checkpoint)
            logging.info(f"Удалён чекпоинт: {checkpoint}")
    logging.info("Очистка завершена. Сохранена только лучшая модель.")
    
    
    # Этап ансамблирования: извлечение эмбеддингов и обучение XGBoost
    logging.info("Этап ансамблирования: извлечение эмбеддингов и обучение XGBoost.")
    try:
        # Создаем модель-эмбеддингов, используя слой с именем 'embedding_layer'
        feature_extractor = Model(inputs=model.input, outputs=model.get_layer("embedding_layer").output)
        embeddings_train = feature_extractor.predict(X_train_scaled)
        embeddings_val = feature_extractor.predict(X_val_scaled)
        logging.info(f"Эмбеддинги получены: embeddings_train.shape = {embeddings_train.shape}")
        
        # Обучаем XGBoost классификатор на эмбеддингах
        xgb_model = train_xgboost_on_embeddings(embeddings_train, y_train)
        logging.info("XGBoost классификатор успешно обучен на эмбеддингах.")
        
        # На валидационной выборке объединяем прогнозы
        nn_pred_proba = model.predict(X_val_scaled)
        xgb_pred_proba = xgb_model.predict_proba(embeddings_val)
        ensemble_pred_proba = 0.5 * nn_pred_proba + 0.5 * xgb_pred_proba
        ensemble_pred_classes = np.argmax(ensemble_pred_proba, axis=1)
        ensemble_f1 = f1_score(y_val, ensemble_pred_classes, average='weighted')
        logging.info(f"Этап ансамблирования: F1-score ансамбля на валидации = {ensemble_f1:.4f}")
        
        # Сохраняем XGBoost модель отдельно
        xgb_save_path = os.path.join("/workspace/saved_models", "xgb_model_flat.pkl")
        os.makedirs(os.path.dirname(xgb_save_path), exist_ok=True)
        joblib.dump(xgb_model, xgb_save_path)
        logging.info(f"XGBoost-модель сохранена в '{xgb_save_path}'")

        
        # Формируем итоговый ансамбль в виде словаря
        ensemble_model = {
            "nn_model": model,
            "xgb_model": xgb_model,
            "feature_extractor": feature_extractor,
            "ensemble_weight_nn": 0.5,
            "ensemble_weight_xgb": 0.5
        }
    except Exception as e:
        logging.error(f"Ошибка на этапе ансамблирования: {e}")
        ensemble_model = {"nn_model": model}
        feature_extractor = None
    
    # Сохраняем финальную модель нейросети
    try:
        model_save_path = os.path.join("/workspace/saved_models", "flat_neural_network.h5")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logging.info(f"Модель успешно сохранена в '{model_save_path}'")
        output_dir = os.path.join("/workspace/output", "flat_neural_network")
        copy_output("Neural_Flat", output_dir)
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели: {e}")

    
    # Возвращаем итоговый ансамбль и масштабировщик
    return {"ensemble_model": ensemble_model, "scaler": scaler}



if __name__ == "__main__":
    # Инициализация стратегии (TPU или CPU/GPU)
    strategy = initialize_strategy()
    
    symbols = ['BTCUSDC', 'ETHUSDC']

    # Периоды флэтового рынка
    flat_periods = [
        {"start": "2020-01-01", "end": "2020-02-29"},
    ]




    # Преобразуем даты из строк в datetime объекты
    start_date = datetime.strptime(flat_periods[0]["start"], "%Y-%m-%d")
    end_date = datetime.strptime(flat_periods[0]["end"], "%Y-%m-%d")

    data = load_flat_data(symbols, flat_periods, interval="1m")

    # Проверяем, что data — это словарь, и он не пуст
    if not isinstance(data, dict) or not data:
        raise ValueError("Ошибка: load_flat_data() вернула пустой словарь!")

    # Объединяем все DataFrame из словаря data в один общий DataFrame
    data = pd.concat(data.values(), ignore_index=False)

    # Проверяем наличие 'timestamp' в колонках
    if 'timestamp' not in data.columns:
        logging.warning("'timestamp' отсутствует, проверяем индекс.")
        if isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = data.index
            logging.info("Индекс преобразован в колонку 'timestamp'.")
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")

    # Применяем предобработку для флэтового рынка,
    # чтобы гарантированно создать необходимые признаки, включая 'clean_returns'
    data = detect_anomalies(data)
    data = validate_volume_confirmation(data)
    data = remove_noise(data)
    # Извлечение признаков
    data = extract_features(data)

    # Удаление NaN
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Обучение модели
    model, scaler = build_flat_neural_network(
        data, 
        model_filename="flat_nn_model.h5"
    )