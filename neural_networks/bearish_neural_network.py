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

# Имя файла для сохранения модели
nn_model_filename = os.path.join("/workspace/saved_models", 'bearish_nn_model.h5')
log_file = os.path.join("/workspace/logs", "training_log_bearish_nn.txt")


network_name = "bearish_neural_network"  # Имя модели
checkpoint_path_regular = os.path.join("/workspace/checkpoints", f"{network_name}_checkpoint_epoch_{{epoch:02d}}.h5")
checkpoint_path_best = os.path.join("/workspace/checkpoints", f"{network_name}_best_model.h5")

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
    data['volume_zscore'] = ((data['volume'] - data['volume'].rolling(50).mean()) / 
                            data['volume'].rolling(50).std())
    data['price_zscore'] = ((data['close'] - data['close'].rolling(50).mean()) / 
                           data['close'].rolling(50).std())
    
    # Более строгие критерии для падений
    data['is_anomaly'] = ((abs(data['volume_zscore']) > 3) & (data['close'] < data['close'].shift(1)) | 
                         (abs(data['price_zscore']) > 3))
    return data

def validate_volume_confirmation(data):
    """
    Добавляет признаки подтверждения движений объемом.
    """
    # Объемное подтверждение тренда
    data['volume_trend_conf'] = np.where(
        (data['close'] > data['close'].shift(1)) & 
        (data['volume'] > data['volume'].rolling(20).mean()),
        1,
        np.where(
            (data['close'] < data['close'].shift(1)) & 
            (data['volume'] > data['volume'].rolling(20).mean()),
            -1,
            0
        )
    )
    
    # Сила объемного подтверждения
    data['volume_strength'] = (data['volume'] / 
                             data['volume'].rolling(20).mean() * 
                             data['volume_trend_conf'])
    
    # Накопление объема
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(5).sum()
    
    return data

def remove_noise(data):
    """
    Улучшенная фильтрация шума с комбинацией фильтра Калмана и экспоненциального сглаживания.
    """
    # Создаем копию данных
    data = data.copy()
    
    # Kalman filter для сглаживания цены
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    kf.R = 5
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])
    
    # Применяем фильтр Калмана
    smoothed_prices = []
    for price in data['close']:
        kf.predict()
        kf.update(price)
        smoothed_prices.append(float(kf.x[0]))
    
    # Используем фильтр Калмана для основного сглаживания
    data['smoothed_close'] = smoothed_prices
    
    # Дополнительное сглаживание EMA для определения выбросов
    ema_smooth = data['close'].ewm(span=10, min_periods=1, adjust=False).mean()
    
    # Вычисляем стандартное отклонение для определения выбросов
    rolling_std = data['close'].rolling(window=20).std()
    rolling_mean = data['close'].rolling(window=20).mean()
    
    # Определяем выбросы (z-score > 3)
    data['is_anomaly'] = abs(data['close'] - rolling_mean) > (3 * rolling_std)
    data['is_anomaly'] = data['is_anomaly'].astype(int)  # Преобразуем в 0 и 1
    
    # Вычисляем "чистые" движения с учетом аномалий
    data['clean_returns'] = data['smoothed_close'].pct_change() * (1 - data['is_anomaly'])
    
    # Заполняем NaN значения
    data = data.ffill().bfill()
    
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
    y_true = tf.cast(y_true, dtype=tf.float32)
    
    # Основная разница
    diff = y_pred - y_true
    
    # Логарифмический фактор для усиления больших ошибок (оставляем эту важную часть)
    log_factor = tf.math.log1p(tf.abs(diff) + 1e-7)
    
    # Штрафы для медвежьего рынка на минутном таймфрейме
    false_long_penalty = 2.5  # Сильный штраф за ложные сигналы покупки
    false_short_penalty = 1.5  # Умеренный штраф за ложные сигналы продажи
    missed_drop_penalty = 2.0  # Штраф за пропуск сильного падения
    
    # Расчет потерь с учетом специфики медвежьего рынка
    loss = tf.where(
        tf.logical_and(y_true == 0, y_pred > 0.5),  # Ложный сигнал на покупку
        false_long_penalty * tf.abs(diff) * log_factor,  # Умножаем на log_factor
        tf.where(
            tf.logical_and(y_true == 2, y_pred < 0.5),  # Пропуск сильного падения
            missed_drop_penalty * tf.abs(diff) * log_factor,  # Умножаем на log_factor
            tf.where(
                tf.logical_and(y_true == 0, y_pred < 0.2),  # Ложный сигнал на продажу
                false_short_penalty * tf.abs(diff) * log_factor,  # Умножаем на log_factor
                tf.abs(diff) * log_factor  # Умножаем на log_factor
            )
        )
    )
    
    # Добавляем штраф за неуверенные предсказания
    uncertainty_penalty = tf.where(
        tf.logical_and(y_pred > 0.3, y_pred < 0.7),
        0.5 * tf.abs(diff) * log_factor,  # Умножаем на log_factor
        0.0
    )
    
    # Добавить штраф за задержку реакции
    time_penalty = 0.1 * tf.abs(diff) * tf.cast(tf.range(tf.shape(diff)[0]), tf.float32) / tf.cast(tf.shape(diff)[0], tf.float32)
    
    # Добавить компонент для учета транзакционных издержек
    transaction_cost = 0.001 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]))
    
    total_loss = tf.reduce_mean(loss + uncertainty_penalty)
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
            executor.submit(get_historical_data, [symbol], bearish_periods, interval, save_path): symbol
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
    data = remove_noise(data)

    # 1. Улучшенная целевая переменная с градацией силы сигналов
    returns = data['close'].pct_change()
    volume_ratio = data['volume'] / data['volume'].rolling(10).mean()
    price_acceleration = returns.diff()  # Скорость изменения цены
    
    # 2. Динамические пороги на основе волатильности
    def calculate_dynamic_thresholds(window=10):
        volatility = returns.rolling(window).std()
        avg_volatility = volatility.rolling(100).mean()  # Долгосрочная средняя волатильность
        volatility_ratio = volatility / avg_volatility
        
        # Базовые пороги для 1-минутных свечей на медвежьем рынке
        base_strong = -0.001  # 0.1%
        base_medium = -0.0005  # 0.05%
        
        # Адаптация порогов
        strong_threshold = base_strong * np.where(
            volatility_ratio > 1.5, 1.5,  # Ограничиваем максимальную адаптацию
            np.where(volatility_ratio < 0.5, 0.5, volatility_ratio)
        )
        medium_threshold = base_medium * np.where(
            volatility_ratio > 1.5, 1.5,
            np.where(volatility_ratio < 0.5, 0.5, volatility_ratio)
        )
        
        return strong_threshold, medium_threshold

    # 3. Создание улучшенной целевой переменной
    strong_threshold, medium_threshold = calculate_dynamic_thresholds()
    
    # Учитываем не только цену, но и объем и скорость падения
    data['target'] = np.where(
        (returns.shift(-1) < strong_threshold) & 
        (volume_ratio > 1.2) & 
        (price_acceleration < 0) &
        (data['volume'] > data['volume'].rolling(20).mean()), 
        2,  # Сильный сигнал
        np.where(
            (returns.shift(-1) < medium_threshold) & 
            (volume_ratio > 1) & 
            (price_acceleration < 0),
            1,  # Средний сигнал
            0   # Нет сигнала
        )
    )

    # 2. Базовые характеристики
    data['returns'] = returns
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # 3. Анализ объемов и давления продаж (специфика медвежьего рынка)
    data['volume_ma'] = data['volume'].rolling(10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['selling_pressure'] = data['volume'] * (data['close'] - data['open']).abs() * \
                              np.where(data['close'] < data['open'], 1, 0)
    data['buying_pressure'] = data['volume'] * (data['close'] - data['open']).abs() * \
                             np.where(data['close'] > data['open'], 1, 0)
    data['pressure_ratio'] = data['selling_pressure'] / data['buying_pressure'].replace(0, 1)

    # 4. Динамические индикаторы волатильности
    data['volatility'] = returns.rolling(10).std()
    data['volatility_ma'] = data['volatility'].rolling(20).mean()
    data['volatility_ratio'] = data['volatility'] / data['volatility_ma']

    # 5. Трендовые индикаторы с адаптивными периодами
    for period in [3, 5, 8, 13, 21]:  # Числа Фибоначчи для лучшего охвата
        data[f'sma_{period}'] = SMAIndicator(data['smoothed_close'], window=period).sma_indicator()
        data[f'ema_{period}'] = data['smoothed_close'].ewm(span=period, adjust=False).mean()

    # 6. MACD с настройкой под минутные свечи
    macd = MACD(data['smoothed_close'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    data['macd_slope'] = data['macd_diff'].diff()  # Скорость изменения MACD

    # 7. Объемные индикаторы с фокусом на медвежий рынок
    data['obv'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    data['cmf'] = ChaikinMoneyFlowIndicator(data['high'], data['low'], data['close'], data['volume']).chaikin_money_flow()
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

    # 8. Осцилляторы с адаптивными периодами
    for period in [7, 14, 21]:
        data[f'rsi_{period}'] = RSIIndicator(data['close'], window=period).rsi()
    data['stoch_k'] = StochasticOscillator(data['high'], data['low'], data['close'], window=7).stoch()
    data['stoch_d'] = StochasticOscillator(data['high'], data['low'], data['close'], window=7).stoch_signal()
    
    # 9. Индикаторы уровней поддержки/сопротивления
    data['support_level'] = data['low'].rolling(20).min()
    data['resistance_level'] = data['high'].rolling(20).max()
    data['price_to_support'] = data['close'] / data['support_level']
    
    # 10. Паттерны свечей и их силы
    data['candle_body'] = abs(data['close'] - data['open'])
    data['upper_shadow'] = data['high'] - np.maximum(data['close'], data['open'])
    data['lower_shadow'] = np.minimum(data['close'], data['open']) - data['low']
    data['body_to_shadow_ratio'] = data['candle_body'] / (data['upper_shadow'] + data['lower_shadow']).replace(0, 0.001)

    # 11. Ценовые уровни и их прорывы
    data['price_level_breach'] = np.where(
        data['close'] < data['support_level'].shift(1), -1,
        np.where(data['close'] > data['resistance_level'].shift(1), 1, 0)
    )

    # 12. Индикаторы скорости движения
    data['price_acceleration'] = returns.diff()
    data['volume_acceleration'] = data['volume_change'].diff()
    
    # 13. Волатильность (с адаптивными периодами)
    bb = BollingerBands(data['smoothed_close'], window=20)
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()
    data['bb_width'] = bb.bollinger_wband()
    data['bb_position'] = (data['close'] - data['bb_low']) / (data['bb_high'] - data['bb_low'])
    
    # 14. ATR с разными периодами для оценки волатильности
    for period in [5, 10, 20]:
        data[f'atr_{period}'] = AverageTrueRange(
            data['high'], data['low'], data['close'], window=period
        ).average_true_range()

    # 4. Добавляем специальные признаки для высокочастотной торговли
    data['micro_trend'] = np.where(
        data['smoothed_close'] > data['smoothed_close'].shift(1), 1,
        np.where(data['smoothed_close'] < data['smoothed_close'].shift(1), -1, 0)
    )
    
    # Считаем микро-тренды последних 5 минут
    data['micro_trend_sum'] = data['micro_trend'].rolling(5).sum()
    
    # Добавляем признак ускорения объемов
    data['volume_acceleration_5m'] = (
        data['volume'].diff() / data['volume'].rolling(5).mean()
    ).fillna(0)

    # 5. Признаки для определения силы медвежьего движения
    data['bearish_strength'] = np.where(
        (data['close'] < data['open']) &  # Медвежья свеча
        (data['volume'] > data['volume'].rolling(20).mean() * 1.5) &  # Объем выше среднего
        (data['close'] == data['low']) &  # Закрытие на минимуме
        (data['clean_returns'] < 0),  # Используем очищенные returns
        3,  # Сильное медвежье движение
        np.where(
            (data['close'] < data['open']) &
            (data['volume'] > data['volume'].rolling(20).mean()) &
            (data['clean_returns'] < 0),
            2,  # Среднее медвежье движение
            np.where(data['close'] < data['open'], 1, 0)  # Слабое/нет движения
        )
    )
    
    # Создаем словарь признаков
    features = {}
    
    features['target'] = data['target']  # Добавляем целевую переменную
    
    # Базовые признаки (все, что уже рассчитано)
    for col in data.columns:
        if col not in ['market_type']:
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
    
    # Добавить в конец функции перед return
    logging.info(f"Количество признаков: {len(features_df.columns)}")
    logging.info(f"Проверка на NaN: {features_df.isna().sum().sum()}")
    logging.info(f"Распределение целевой переменной:\n{features_df['target'].value_counts()}")
    
    logging.info(f"✅ Итоговые признаки: {list(data.columns)}")
    num_nans = data.isna().sum().sum()
    if num_nans > 0:
        logging.warning(f"⚠ Найдено {num_nans} пропущенных значений. Заполняем...")
        data.fillna(0, inplace=True)


    return features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]


def add_clustering_feature(data):
    features_for_clustering = [
        'close', 'volume', 'rsi', 'macd', 'atr', 'sma_10', 'sma_30', 'ema_50', 'ema_200',
        'bb_width', 'macd_diff', 'obv', 'returns', 'log_returns'
    ]
    kmeans = KMeans(n_clusters=5, random_state=42)
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
    data = extract_features(data)
    logging.info(f"После извлечения признаков: {data.shape}")
    
    # Удаление выбросов
    data = remove_outliers(data)
    logging.info(f"После удаления выбросов: {data.shape}")
    
    # Добавление кластеризационного признака
    data = add_clustering_feature(data)
    logging.info(f"После кластеризации: {data.shape}")
    
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

def balance_classes(X, y):
    logging.info("Начало балансировки классов")
    logging.info(f"Размеры данных до балансировки: X={X.shape}, y={y.shape}")
    logging.info(f"Уникальные классы в y: {np.unique(y, return_counts=True)}")

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Данные для балансировки пусты. Проверьте исходные данные и фильтры.")

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    logging.info(f"Размеры данных после балансировки: X={X_resampled.shape}, y={y_resampled.shape}")
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


def build_bearish_neural_network(data):
    """
    Обучает нейронную сеть для медвежьего рынка с корректной обработкой временной метки.
    
    Здесь в самом начале данные передаются через prepare_timestamp_column,
    которая сбрасывает индекс и создаёт (или обновляет) столбец 'timestamp'. 
    Далее происходит выбор признаков, балансировка, разделение выборок, масштабирование и обучение модели,
    сохраняя весь продвинутый функционал (архитектуру LSTM, ансамблирование с XGBoost и пр.).
    
    Важно: теперь никакие reset_index() не выполняются после вызова этой функции.
    """
    logging.info("Начало обучения нейронной сети для медвежьего рынка.")
    
    # Гарантированно создаём столбец 'timestamp'
    data = prepare_timestamp_column(data)
    
    # Выбираем числовые признаки (исключая 'target' и 'timestamp')
    selected_features = [
        col for col in data.columns 
        if col not in ['target', 'timestamp'] and pd.api.types.is_numeric_dtype(data[col])
    ]
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
    
    X_resampled, y_resampled = balance_classes(X, y)
    logging.info(f"Размеры после балансировки: X_resampled={X_resampled.shape}, y_resampled={y_resampled.shape}")
    logging.info(f"Распределение классов после балансировки:\n{pd.Series(y_resampled).value_counts()}")
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    logging.info(f"Размеры тренировочных данных: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Размеры валидационных данных: X_val={X_val.shape}, y_val={y_val.shape}")
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
    
    def hft_metrics(y_true, y_pred):
        reaction_time = tf.reduce_mean(tf.abs(y_pred[1:] - y_pred[:-1]))
        signal_stability = tf.reduce_mean(tf.abs(y_pred[2:] - 2 * y_pred[1:-1] + y_pred[:-2]))
        return reaction_time, signal_stability
    
    def profit_ratio(y_true, y_pred):
        successful_shorts = tf.reduce_sum(tf.where(tf.logical_and(y_true >= 1, y_pred >= 0.5), 1.0, 0.0))
        false_signals = tf.reduce_sum(tf.where(tf.logical_and(y_true == 0, y_pred >= 0.5), 1.0, 0.0))
        return successful_shorts / (false_signals + K.epsilon())
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            strategy = tf.distribute.MirroredStrategy()
            logging.info("GPU инициализированы с использованием MirroredStrategy")
        except RuntimeError as e:
            logging.error(f"Ошибка при инициализации GPU: {e}")
            strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.get_strategy()
        logging.info("GPU не найдены, используется стратегия по умолчанию")
    
    logging.info("Начинаем создание модели для медвежьего рынка...")
    with strategy.scope():
        inputs = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))
        x1 = LSTM(256, return_sequences=True)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        x2 = LSTM(256, return_sequences=True)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        x3 = LSTM(256, return_sequences=True, name='market_context')(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.3)(x3)
        x = Add()([x1, x2, x3])
        x = LSTM(256, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu', name="embedding_layer")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(3, activation='softmax')(x)
        model = tf.keras.models.Model(inputs, outputs)
    
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss=custom_profit_loss,
                      metrics=[hft_metrics, profit_ratio])
    
        try:
            model.load_weights(checkpoint_path_regular.format(epoch=0))
            logging.info(f"Загружены веса модели из {checkpoint_path_regular.format(epoch=0)}")
        except FileNotFoundError:
            logging.info("Контрольная точка не найдена. Начинаем обучение с нуля.")
    
        regular_checkpoints = sorted(glob.glob(f"checkpoints/{network_name}_checkpoint_epoch_*.h5"))
        if regular_checkpoints:
            latest_checkpoint = regular_checkpoints[-1]
            try:
                model.load_weights(latest_checkpoint)
                logging.info(f"Загружены веса из последнего регулярного чекпоинта: {latest_checkpoint}")
            except Exception as e:
                logging.error(f"Ошибка загрузки регулярного чекпоинта: {e}")
    
        if os.path.exists(checkpoint_path_best):
            try:
                model.load_weights(checkpoint_path_best)
                logging.info(f"Лучший чекпоинт найден: {checkpoint_path_best}. После обучения промежуточные чекпоинты будут удалены.")
            except Exception as e:
                logging.info("Лучший чекпоинт пока не создан. Это ожидаемо, если обучение ещё не завершено.")
    
        checkpoint_every_epoch = ModelCheckpoint(filepath=checkpoint_path_regular,
                                                 save_weights_only=True,
                                                 save_best_only=False,
                                                 verbose=1)
        checkpoint_best_model = ModelCheckpoint(filepath=checkpoint_path_best,
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                verbose=1)
        tensorboard_callback = TensorBoard(log_dir=f"logs/{time.time()}")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=5, min_lr=1e-5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                       restore_best_weights=True, mode='min')
    
        history = model.fit(train_dataset,
                            epochs=1, #200
                            validation_data=val_dataset,
                            class_weight={0: 1.0, 1: 2.0, 2: 3.0},
                            verbose=1,
                            callbacks=[early_stopping, checkpoint_every_epoch,
                                       checkpoint_best_model, tensorboard_callback,
                                       reduce_lr])
    
        for checkpoint in glob.glob(f"checkpoints/{network_name}_checkpoint_epoch_*.h5"):
            if checkpoint != checkpoint_path_best:
                os.remove(checkpoint)
                logging.info(f"Удалён чекпоинт: {checkpoint}")
        logging.info("Очистка завершена. Сохранена только лучшая модель.")
    
        try:
            model_save_path = os.path.join("/workspace/saved_models", "bearish_neural_network.h5")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            model.save(model_save_path)
            logging.info(f"Модель успешно сохранена в '{model_save_path}'")
            output_dir = os.path.join("/workspace/output", "bearish_neural_network")
            copy_output("Neural_Bearish", output_dir)
        except Exception as e:
            logging.error(f"Ошибка при сохранении модели: {e}")

    
        logging.info("Этап ансамблирования: извлечение эмбеддингов и обучение XGBoost для медвежьего рынка.")
        try:
            feature_extractor = Model(inputs=model.input, outputs=model.get_layer("embedding_layer").output)
            embeddings_train = feature_extractor.predict(X_train_scaled)
            embeddings_val = feature_extractor.predict(X_val_scaled)
            logging.info(f"Эмбеддинги получены: embeddings_train.shape = {embeddings_train.shape}")
    
            xgb_model = train_xgboost_on_embeddings(embeddings_train, y_train)
            logging.info("XGBoost классификатор успешно обучен на эмбеддингах.")
    
            nn_val_pred = model.predict(X_val_scaled)
            xgb_val_pred = xgb_model.predict_proba(embeddings_val)
            ensemble_val_pred = 0.5 * nn_val_pred + 0.5 * xgb_val_pred
            ensemble_val_pred_class = np.argmax(ensemble_val_pred, axis=1)
            ensemble_f1 = f1_score(y_val, ensemble_val_pred_class, average='weighted')
            logging.info(f"Этап ансамблирования: F1-score ансамбля на валидации = {ensemble_f1:.4f}")
    
            xgb_save_path = os.path.join("/workspace/saved_models", "xgb_model_bearish.pkl")
            joblib.dump(xgb_model, xgb_save_path)
            logging.info(f"XGBoost-модель сохранена в '{xgb_save_path}'")

    
            ensemble_model = {"nn_model": model,
                              "xgb_model": xgb_model,
                              "feature_extractor": feature_extractor,
                              "ensemble_weight_nn": 0.5,
                              "ensemble_weight_xgb": 0.5}
        except Exception as e:
            logging.error(f"Ошибка на этапе ансамблирования: {e}")
            ensemble_model = {"nn_model": model}
    
        return {"ensemble_model": ensemble_model, "scaler": scaler}

if __name__ == "__main__":
    try:
        strategy = initialize_strategy()
        
        symbols = ['BTCUSDC', 'ETHUSDC']
        
        bearish_periods = [
            {"start": "2021-05-12", "end": "2021-06-12"},
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
