import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime, timedelta
from binance.client import Client
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
import pandas_ta as ta
import requests
from scipy.stats import zscore
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import l2
import time
from binance.exceptions import BinanceAPIException
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import tensorflow.keras.backend as K
import xgboost as xgb
from tensorflow.keras.layers import Layer, GRU, Input, Concatenate
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.backend import clear_session
import glob
import requests
import zipfile
from io import BytesIO
from threading import Lock
from ta.trend import SMAIndicator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from scipy.stats import zscore
from scipy.stats import mode



# ✅ Использование GPU, если доступно
def initialize_strategy():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("✅ Используется GPU!")
            return tf.distribute.MirroredStrategy()
        except RuntimeError as e:
            logging.warning(f"⚠ Ошибка при инициализации GPU: {e}")
            return tf.distribute.get_strategy()
    else:
        logging.info("❌ GPU не найден, используем CPU")
        return tf.distribute.get_strategy()

    

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            
            

class Attention(Layer):
    """Attention-механизм для выделения важных временных точек"""
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return x * a
    
class MarketClassifier:
    def __init__(self, model_path="market_condition_classifier.h5", scaler_path="scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/"
        
        
    def apply_in_chunks(df, func, chunk_size=100000):
        """
        Применяет функцию func к DataFrame по чанкам заданного размера.
        Если df не является DataFrame, возвращает func(df).
        """
        if not isinstance(df, pd.DataFrame):
            return func(df)
        if len(df) <= chunk_size:
            return func(df)
        # Разбиваем DataFrame на чанки
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        # Применяем функцию к каждому чанку
        processed_chunks = [func(chunk) for chunk in chunks]
        # Объединяем обработанные чанки в один DataFrame
        return pd.concat(processed_chunks)
    

    def fetch_binance_data(self, symbol, interval, start_date, end_date):
        """
        Скачивает исторические данные с Binance без API-ключа (архив Binance) для заданного символа.
        Возвращает DataFrame с колонками: timestamp, open, high, low, close, volume.
        """
        base_url_monthly = "https://data.binance.vision/data/spot/monthly/klines"
        logging.info(f"📡 Загрузка данных с Binance для {symbol} ({interval}) c {start_date} по {end_date}...")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        all_data = []
        downloaded_files = set()
        download_lock = Lock()  # Глобальная блокировка скачивания, чтобы избежать дублирования

        def download_and_process(date):
            year, month = date.year, date.month
            month_str = f"{month:02d}"
            file_name = f"{symbol}-{interval}-{year}-{month_str}.zip"
            file_url = f"{base_url_monthly}/{symbol}/{interval}/{file_name}"

            # Проверка на уже загруженные файлы
            with download_lock:
                if file_name in downloaded_files:
                    logging.info(f"⏩ Пропуск {file_name}, уже загружено.")
                    return None

                logging.info(f"🔍 Проверка наличия {file_url}...")
                response = requests.head(file_url, timeout=5)
                if response.status_code != 200:
                    logging.warning(f"⚠ Файл не найден: {file_url}")
                    return None

                logging.info(f"📥 Скачивание {file_url}...")
                response = requests.get(file_url, timeout=15)
                if response.status_code != 200:
                    logging.warning(f"⚠ Ошибка загрузки {file_url}: Код {response.status_code}")
                    return None

                logging.info(f"✅ Успешно загружен {file_name}")
                downloaded_files.add(file_name)

            try:
                with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                    csv_file = file_name.replace('.zip', '.csv')
                    with zip_file.open(csv_file) as file:
                        df = pd.read_csv(
                            file, header=None, 
                            names=[
                                "timestamp", "open", "high", "low", "close", "volume",
                                "close_time", "quote_asset_volume", "number_of_trades",
                                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                            ],
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
                            },
                            low_memory=False
                        )
                        # Преобразуем timestamp в datetime
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        # Выбираем только необходимые колонки
                        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        # Приводим числовые колонки к типу float, не затрагивая timestamp
                        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                        # Устанавливаем timestamp в качестве индекса для агрегации
                        df.set_index("timestamp", inplace=True)
                        return df
            except Exception as e:
                logging.error(f"❌ Ошибка обработки {symbol} за {date.strftime('%Y-%m')}: {e}")
                return None

        # Запускаем скачивание в многопоточном режиме
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(download_and_process, pd.date_range(start=start_date, end=end_date, freq='MS')))

        # Собираем загруженные данные
        all_data = [df for df in results if df is not None]

        if not all_data:
            raise ValueError(f"❌ Не удалось загрузить ни одного месяца данных для {symbol}.")

        df = pd.concat(all_data)
        logging.info(f"📊 Итоговая форма данных: {df.shape}")

        # Если вдруг колонка 'timestamp' отсутствует как столбец, сбрасываем индекс
        if "timestamp" not in df.columns:
            df.reset_index(inplace=True)

        # Гарантируем, что timestamp имеет правильный тип
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.set_index("timestamp", inplace=True)

        # Агрегация до 1-минутного таймфрейма
        df = df.resample('1min').ffill()

        # Сброс индекса, чтобы timestamp стал обычной колонкой
        df.reset_index(inplace=True)

        # Обработка пропущенных значений
        num_nans = df.isna().sum().sum()
        if num_nans > 0:
            if num_nans / len(df) > 0.05:  # Если более 5% данных пропущены
                logging.warning("⚠ Пропущено слишком много данных! Пропускаем эти свечи.")
                df.dropna(inplace=True)
            else:
                df.fillna(method='ffill', inplace=True)

        logging.info(f"✅ Данные успешно загружены: {len(df)} записей")
        return df

        

    def add_indicators(self, data):
        """Расширенный набор индикаторов для классификации с использованием pandas_ta"""
        # Базовые индикаторы
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=7)
        data['adx'] = ta.adx(data['high'], data['low'], data['close'], length=7)['ADX_7']
        
        # Множественные MA для определения тренда
        for period in [5, 10, 15, 20, 50]:
            data[f'sma_{period}'] = ta.sma(data['close'], length=period)
            data[f'ema_{period}'] = ta.ema(data['close'], length=period)
        
        # Импульсные индикаторы
        data['rsi'] = ta.rsi(data['close'], length=7)
        macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        data['macd'] = macd['MACD_12_26_9']
        data['macd_signal'] = macd['MACDs_12_26_9']
        data['macd_hist'] = macd['MACDh_12_26_9']
        data['willr'] = ta.willr(data['high'], data['low'], data['close'], length=14)
        
        # Волатильность: группируем rolling для close с окном 10 для Bollinger Bands
        bb = ta.bbands(data['close'], length=10, std=2)
        data['bb_upper']  = bb[f'BBU_10_2.0']
        data['bb_middle'] = bb[f'BBM_10_2.0']
        data['bb_lower']  = bb[f'BBL_10_2.0']
        data['bb_width']  = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']

        
        # Объемные индикаторы: группируем rolling для volume с окном 20
        vs = data['volume'].rolling(window=20)
        data['volume_sma'] = vs.mean()
        data['volume_std'] = vs.std()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Трендовые индикаторы
        data['trend_strength'] = abs(data['close'].pct_change().rolling(window=20).sum())
        data['price_momentum'] = data['close'].diff(periods=10) / data['close'].shift(10)
        
        # Динамические уровни поддержки/сопротивления
        data['support_level'] = data['low'].rolling(window=20).min()
        data['resistance_level'] = data['high'].rolling(window=20).max()
        
        return data


    
    def compute_scaler(self, data_path, sample_size=100000, chunk_size=10000):
        """
        Выполняет предварительный проход по CSV-файлу, читая данные чанками, 
        и вычисляет параметры RobustScaler на представительном подмножестве данных.
        Возвращает обученный масштабировщик и список признаков.
        """
        samples = []
        total_samples = 0
        features = None
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            chunk.dropna(inplace=True)
            # Если индикаторы ещё не добавлены, вызываем существующую функцию
            if 'atr' not in chunk.columns:
                chunk = self.add_indicators(chunk)
            chunk = self.remove_outliers(chunk)
            # Преобразование меток рынка
            label_mapping = {'bullish': 0, 'bearish': 1, 'flat': 2}
            chunk['market_type'] = chunk['market_type'].map(label_mapping)
            chunk.dropna(subset=['market_type'], inplace=True)
            if features is None:
                features = [col for col in chunk.columns if col not in ['market_type', 'symbol', 'timestamp']]
            X_chunk = chunk[features].values
            samples.append(X_chunk)
            total_samples += X_chunk.shape[0]
            if total_samples >= sample_size:
                break
        X_sample = np.concatenate(samples, axis=0)
        scaler = RobustScaler().fit(X_sample)
        logging.info(f"Масштабировщик обучен на {X_sample.shape[0]} выборках.")
        return scaler, features

    def data_generator_split(self, data_path, scaler, features, batch_size, chunk_size=10000, 
                             split='train', train_fraction=0.8, random_seed=42):
        """
        Генератор, который читает CSV-файл чанками, выполняет обработку данных и делит их 
        на тренировочную и валидационную выборки согласно train_fraction.
        """
        chunk_counter = 0
        # Функция для дополнительной обработки каждого подчанка
        def process_subchunk(sub_df):
            # Добавляем индикаторы и удаляем выбросы для подчанка
            sub_df = self.add_indicators(sub_df)
            sub_df = self.remove_outliers(sub_df)
            return sub_df

        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            chunk.dropna(inplace=True)
            # Если данные в чанке все ещё очень большие, делим их дополнительно
            # Здесь вызываем apply_in_chunks для дополнительной обработки
            chunk = apply_in_chunks(chunk, process_subchunk, chunk_size=5000)
            
            label_mapping = {'bullish': 0, 'bearish': 1, 'flat': 2}
            chunk['market_type'] = chunk['market_type'].map(label_mapping)
            chunk.dropna(subset=['market_type'], inplace=True)
            
            X_chunk = chunk[features].values
            y_chunk = chunk['market_type'].values.astype(int)
            n_samples = X_chunk.shape[0]
            
            r = np.random.RandomState(random_seed + chunk_counter)
            random_vals = r.rand(n_samples)
            mask = random_vals < train_fraction if split == 'train' else random_vals >= train_fraction
            X_selected = X_chunk[mask]
            y_selected = y_chunk[mask]
            if X_selected.shape[0] == 0:
                chunk_counter += 1
                continue
            
            X_scaled = scaler.transform(X_selected)
            indices = np.arange(X_scaled.shape[0])
            r_shuffle = np.random.RandomState(random_seed + chunk_counter + 1000)
            r_shuffle.shuffle(indices)
            X_scaled = X_scaled[indices]
            y_selected = y_selected[indices]
            
            n_sel = X_scaled.shape[0]
            for i in range(0, n_sel, batch_size):
                batch_idx = slice(i, i + batch_size)
                X_batch = X_scaled[batch_idx]
                y_batch = y_selected[batch_idx]
                X_batch = np.expand_dims(X_batch, axis=1)
                yield X_batch.astype(np.float32), y_batch.astype(np.int32)
            
            chunk_counter += 1


    def prepare_training_dataset(self, data_path, scaler, features, batch_size, chunk_size=10000, 
                                 split='train', train_fraction=0.8, random_seed=42):
        """
        Создаёт tf.data.Dataset на основе генератора data_generator_split.
        """
        gen = lambda: self.data_generator_split(
            data_path, scaler, features, batch_size, chunk_size, split, train_fraction, random_seed
        )
        # Получаем один батч для определения формы выходных данных
        sample_gen = self.data_generator_split(
            data_path, scaler, features, batch_size, chunk_size, split, train_fraction, random_seed
        )
        sample = next(sample_gen)
        output_shapes = (tf.TensorShape([None, 1, sample[0].shape[-1]]), tf.TensorShape([None]))
        output_types = (tf.float32, tf.int32)
        dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    
    def validate_predictions(self, data, prediction, window=5):
        """
        Валидация предсказаний классификатора с помощью мультипериодного анализа
        """
        # Проверяем последние N свечей для подтверждения сигнала
        recent_data = data.tail(window)
        
        # 1. Проверка консистентности тренда
        price_direction = recent_data['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        trend_consistency = abs(price_direction.sum()) / window
        
        # 2. Проверка объемного подтверждения
        volume_trend = recent_data['volume'] > recent_data['volume_sma']
        volume_confirmation = volume_trend.mean()
        
        # 3. Проверка импульса
        momentum_confirmation = (
            (recent_data['rsi'] > 60).all() or  # Сильный бычий импульс
            (recent_data['rsi'] < 40).all() or  # Сильный медвежий импульс
            (recent_data['rsi'].between(45, 55)).all()  # Стабильный флэт
        )
        
        # 4. Подтверждение через множественные таймфреймы
        mtf_confirmation = (
            recent_data['adx'].mean() > 25 and  # Сильный тренд
            abs(recent_data['macd_hist']).mean() > recent_data['macd_hist'].std()  # Сильный MACD
        )
        
        # Вычисление общего скора достоверности
        confidence_score = (
            0.3 * trend_consistency +
            0.3 * volume_confirmation +
            0.2 * momentum_confirmation +
            0.2 * mtf_confirmation
        )
        
        # Проверка соответствия предсказания и подтверждений
        if prediction == 'bullish':
            prediction_valid = (
                trend_consistency > 0.6 and
                volume_confirmation > 0.5 and
                recent_data['rsi'].mean() > 55
            )
        elif prediction == 'bearish':
            prediction_valid = (
                trend_consistency > 0.6 and
                volume_confirmation > 0.5 and
                recent_data['rsi'].mean() < 45
            )
        else:  # flat
            prediction_valid = (
                trend_consistency < 0.4 and
                0.3 < volume_confirmation < 0.7 and
                40 < recent_data['rsi'].mean() < 60
            )
        
        return {
            'prediction': prediction,
            'is_valid': prediction_valid,
            'confidence': confidence_score,
            'confirmations': {
                'trend_consistency': trend_consistency,
                'volume_confirmation': volume_confirmation,
                'momentum_confirmation': momentum_confirmation,
                'mtf_confirmation': mtf_confirmation
            }
        }
    
    
    def classify_market_conditions(self, data, window=20):
        """
        Улучшенная классификация рынка с валидацией и множественными подтверждениями
        """
        if len(data) < window:
            logging.warning(f"Недостаточно данных для классификации: {len(data)} < {window}")
            return 'flat'  # Возвращаем flat вместо uncertain как безопасное значение
            
        if not hasattr(self, 'previous_market_type'):
            self.previous_market_type = 'flat'  # Инициализация значением flat

        # Базовые сигналы с подробным логированием
        adx = data['adx'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        macd_hist = data['macd_hist'].iloc[-1]
        willr = data['willr'].iloc[-1]
        volume_ratio = data['volume_ratio'].iloc[-1]
        price = data['close'].iloc[-1]
        support = data['support_level'].iloc[-1]
        resistance = data['resistance_level'].iloc[-1]
        
        # Расчет расстояния до уровней в процентах
        distance_to_support = ((price - support) / price) * 100
        distance_to_resistance = ((resistance - price) / price) * 100
        
        logging.info(f"""
        Текущие значения индикаторов:
        ADX: {adx:.2f}
        RSI: {rsi:.2f}
        MACD Histogram: {macd_hist:.2f}
        Williams %R: {willr:.2f}
        Volume Ratio: {volume_ratio:.2f}
        Цена: {price:.2f}
        Поддержка: {support:.2f} (расстояние: {distance_to_support:.2f}%)
        Сопротивление: {resistance:.2f} (расстояние: {distance_to_resistance:.2f}%)
        """)

        # Подтверждение тренда через MA с логированием
        ma_trends = []
        for period in [10, 20, 50]:
            is_above = price > data[f'sma_{period}'].iloc[-1]
            ma_trends.append(is_above)
            logging.info(f"Цена выше SMA{period}: {is_above}")
        trend_confirmation = sum(ma_trends)
        logging.info(f"Общее подтверждение тренда: {trend_confirmation}/3")

        # Валидационные метрики на последних свечах
        recent_data = data.tail(window)
        price_direction = recent_data['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        trend_consistency = abs(price_direction.sum()) / window
        volume_confirmation = (recent_data['volume'] > recent_data['volume_sma']).mean()
        momentum_strength = abs(recent_data['rsi'] - 50).mean() / 50

        logging.info(f"""
        Метрики тренда:
        Консистентность тренда: {trend_consistency:.2f}
        Подтверждение объёмом: {volume_confirmation:.2f}
        Сила импульса: {momentum_strength:.2f}
        """)

        # Определение типа рынка с более мягкими условиями
        market_type = 'uncertain'
        confidence_score = 0

        # Проверка бычьего рынка
        if (adx > 20 and rsi > 45 and volume_ratio > 1.0 and
            trend_confirmation >= 1 and macd_hist > 0 and
            distance_to_resistance > 0.5):  # Есть пространство для роста
            
            bullish_confidence = (
                0.25 * (trend_consistency if price_direction.sum() > 0 else 0) +
                0.25 * volume_confirmation +
                0.25 * (momentum_strength if rsi > 45 else 0) +
                0.25 * (distance_to_resistance / 5)  # Нормализуем расстояние до сопротивления
            )
            
            logging.info(f"Проверка бычьего рынка. Уверенность: {bullish_confidence:.2f}")
            
            if bullish_confidence > 0.5:
                market_type = 'bullish'
                confidence_score = bullish_confidence

        # Проверка медвежьего рынка
        elif (adx > 20 and rsi < 55 and volume_ratio > 1.0 and
              trend_confirmation <= 2 and macd_hist < 0 and
              distance_to_support > 0.5):  # Есть пространство для падения
            
            bearish_confidence = (
                0.25 * (trend_consistency if price_direction.sum() < 0 else 0) +
                0.25 * volume_confirmation +
                0.25 * (momentum_strength if rsi < 55 else 0) +
                0.25 * (distance_to_support / 5)  # Нормализуем расстояние до поддержки
            )
            
            logging.info(f"Проверка медвежьего рынка. Уверенность: {bearish_confidence:.2f}")
            
            if bearish_confidence > 0.5:
                market_type = 'bearish'
                confidence_score = bearish_confidence

        # Проверка флэта
        elif (adx < 25 and 35 < rsi < 65 and
              abs(macd_hist) < 0.2 * data['macd_hist'].std() and
              0.7 < volume_ratio < 1.3 and
              support < price < resistance and
              max(distance_to_support, distance_to_resistance) < 2):  # Цена между уровнями
            
            flat_confidence = (
                0.25 * (1 - trend_consistency) +
                0.25 * (1 - abs(volume_ratio - 1)) +
                0.25 * (1 - momentum_strength) +
                0.25 * (1 - max(distance_to_support, distance_to_resistance) / 2)
            )
            
            logging.info(f"Проверка флэта. Уверенность: {flat_confidence:.2f}")
            
            if flat_confidence > 0.5:
                market_type = 'flat'
                confidence_score = flat_confidence

        # Если сложная логика не сработала, используем упрощённую классификацию
        if market_type == 'uncertain':
            # Учитываем уровни поддержки и сопротивления в упрощённой логике
            if rsi > 60 and distance_to_resistance > 0.5:
                market_type = 'bullish'
            elif rsi < 40 and distance_to_support > 0.5:
                market_type = 'bearish'
            else:
                market_type = 'flat'
            
            logging.info(f"Используем упрощённую классификацию: {market_type} (RSI: {rsi:.2f}, " +
                        f"Расст. до сопр.: {distance_to_resistance:.2f}%, " +
                        f"Расст. до подд.: {distance_to_support:.2f}%)")
        
        self.previous_market_type = market_type
        return market_type
    

    def remove_outliers(self, data, z_threshold=3):
        """
        Удаляет выбросы на основе метода Z-score.
        """
        z_scores = zscore(data[['open', 'high', 'low', 'close', 'volume']])
        mask = (np.abs(z_scores) < z_threshold).all(axis=1)
        filtered_data = data[mask]
        removed_count = len(data) - len(filtered_data)
        logging.info(f"Удалено выбросов: {removed_count}")
        return filtered_data

    def fetch_market_events(self, api_key, start_date, end_date):
        """
        Получает форс-мажорные события через API.
        """
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "market crash OR economic crisis OR volatility",
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": api_key,
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                events = []
                for article in articles:
                    published_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                    events.append({
                        "start": published_date - timedelta(days=1),
                        "end": published_date + timedelta(days=1),
                        "event": article['title']
                    })
                return events
            else:
                logging.error(f"Ошибка API: {response.status_code} {response.text}")
                return []
        except Exception as e:
            logging.error(f"Ошибка получения событий: {e}")
            return []

    def flag_market_events(self, data, events):
        """
        Добавляет флаги для форс-мажорных рыночных событий.
        """
        data['market_event'] = 'None'
        for event in events:
            mask = (data.index >= event['start']) & (data.index <= event['end'])
            data.loc[mask, 'market_event'] = event['event']
        logging.info(f"Автоматические флаги рыночных событий добавлены.")
        return data

    def fetch_and_label_all(self, symbols, start_date, end_date, save_path="labeled_data"):
        """
        Загружает и размечает данные для нескольких торговых пар без использования API.
        """
        os.makedirs(save_path, exist_ok=True)
        all_data = []

        for symbol in symbols:
            try:
                logging.info(f"Загрузка данных для {symbol}")
                df = self.fetch_binance_data(symbol, "1m", start_date, end_date)  # ✅ Убрано использование API
                df = self.add_indicators(df)
                df['market_type'] = self.classify_market_conditions(df)
                df['symbol'] = symbol
                file_path = os.path.join(save_path, f"{symbol}_data.csv")
                df.to_csv(file_path)
                logging.info(f"Данные для {symbol} сохранены в {file_path}")
                all_data.append(df)
            except Exception as e:
                logging.error(f"Ошибка при обработке {symbol}: {e}")

        if not all_data:
            raise ValueError("Не удалось собрать данные ни для одного символа.")
        
        return pd.concat(all_data, ignore_index=True)


    
    def prepare_training_data(self, data_path):
        """
        Загружает и обрабатывает данные для обучения с использованием чанков.
        Функциональность сохраняется полностью: проверка необходимых столбцов,
        удаление NaN, добавление индикаторов (через self.add_indicators), удаление выбросов,
        преобразование меток рынка в числовой формат.
        Данные читаются по частям (чанками) для экономии памяти.
        """
        logging.info(f"Загрузка данных из файла {data_path}")
        chunks = []
        try:
            # Читаем CSV чанками по 10_000 строк
            for chunk in pd.read_csv(data_path, index_col=0, chunksize=10000):
                # Удаляем строки с пропущенными значениями
                chunk.dropna(inplace=True)

                # Проверка наличия необходимых столбцов
                required_columns = [
                    'open', 'high', 'low', 'close', 'volume',
                    'atr', 'adx', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'willr', 'bb_width', 'volume_ratio', 'trend_strength',
                    'market_type'
                ]
                for col in required_columns:
                    if col not in chunk.columns:
                        raise ValueError(f"Отсутствует необходимый столбец: {col}")

                # Логгируем уникальные значения market_type до преобразования
                logging.info(f"Уникальные значения market_type в чанке до преобразования: {chunk['market_type'].unique()}")

                # Удаление выбросов (единожды вызывается для чанка)
                logging.info("Удаление выбросов из чанка...")
                chunk = self.remove_outliers(chunk)
                logging.info(f"Размер чанка после удаления выбросов: {chunk.shape}")

                # Если индикаторы ещё не добавлены – добавляем (функция add_indicators должна быть реализована в классе)
                if 'atr' not in chunk.columns:
                    chunk = self.add_indicators(chunk)

                # Преобразование меток рынка в числовые значения
                label_mapping = {'bullish': 0, 'bearish': 1, 'flat': 2}
                chunk['market_type'] = chunk['market_type'].map(label_mapping)

                # Удаление строк с NaN после преобразования market_type
                if chunk['market_type'].isna().any():
                    bad_values = chunk[chunk['market_type'].isna()]['market_type'].unique()
                    logging.error(f"Обнаружены NaN значения в market_type! Исходные значения: {bad_values}")
                    chunk.dropna(subset=['market_type'], inplace=True)

                chunks.append(chunk)
        except FileNotFoundError:
            logging.error(f"Файл {data_path} не найден.")
            raise

        # Объединяем все обработанные чанки в один DataFrame
        data = pd.concat(chunks, ignore_index=True)
        features = [col for col in data.columns if col not in ['market_type', 'symbol', 'timestamp']]
        X = data[features].values
        y = data['market_type'].values.astype(int)

        logging.info(f"Форма X: {X.shape}")
        logging.info(f"Форма y: {y.shape}")
        return X, y



    def balance_classes(self, y):
        # Приводим y к numpy-массиву, если это ещё не сделано
        y = np.array(y)
        # Вычисляем веса для классов, присутствующих в y
        present_classes = np.unique(y)
        computed_weights = compute_class_weight(
            class_weight='balanced',
            classes=present_classes,
            y=y
        )
        # Формируем словарь для тех классов, которые есть
        class_weights = {int(cls): weight for cls, weight in zip(present_classes, computed_weights)}
        
        # Гарантируем, что словарь содержит все три класса: 0, 1 и 2
        for cls in [0, 1, 2]:
            if cls not in class_weights:
                # Если класс отсутствует в y, то его вес взять равным 1.0
                # (в дальнейшем на полном наборе данных этот случай возникать не должен)
                class_weights[cls] = 1.0
        return class_weights





    def build_lstm_gru_model(self, input_shape):
        inputs = Input(shape=input_shape)

        # LSTM блок (уменьшаем размер слоёв)
        lstm_out = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(lstm_out)

        # GRU блок (уменьшаем размер слоёв)
        gru_out = GRU(128, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
        gru_out = Dropout(0.3)(gru_out)
        gru_out = GRU(64, return_sequences=True, kernel_regularizer=l2(0.01))(gru_out)

        # Объединяем выходы LSTM и GRU
        combined = Concatenate()([lstm_out, gru_out])

        # Attention-механизм
        attention = Attention()(combined)

        x = LSTM(64, return_sequences=False)(attention)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(attention)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', name="embedding_layer", kernel_regularizer=l2(0.01))(x)
        outputs = Dense(3, activation='softmax')(x)

        model = tf.keras.models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model




    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Обучает XGBoost на эмбеддингах LSTM + GRU, или возвращает DummyClassifier, если в y_train только один класс."""
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logging.warning("В обучающем наборе XGB обнаружен только один класс. Используем DummyClassifier.")
            dummy = DummyClassifier(strategy='constant', constant=unique_classes[0])
            dummy.fit(X_train, y_train)
            return dummy
        else:
            booster = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                learning_rate=0.1,
                n_estimators=10,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=1,
                use_label_encoder=False
            )
            # Здесь вместо X_val, y_val можно передать эмбеддинги тестового набора, например, X_test_features и y_test.
            booster.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_val, y_val)], verbose=False)
            return booster


    def build_ensemble(X_train, y_train):
        """Финальный ансамбль: LSTM + GRU + XGBoost"""
        lstm_gru_model = build_lstm_gru_model((X_train.shape[1], X_train.shape[2]))

        # Обучаем LSTM-GRU
        lstm_gru_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)#epochs=100

        # Извлекаем эмбеддинги из последнего слоя перед softmax
        feature_extractor = tf.keras.models.Model(
            inputs=lstm_gru_model.input, outputs=lstm_gru_model.layers[-3].output
        )
        X_features = feature_extractor.predict(X_train)

        # Обучаем XGBoost на эмбеддингах
        xgb_model = self.train_xgboost(X_features, y_train)

        # Ансамбль моделей через VotingClassifier
        ensemble = VotingClassifier(
            estimators=[
                ('lstm_gru', lstm_gru_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )
        return ensemble


    def train_market_condition_classifier(self, data_path, model_path='market_condition_classifier.h5',
                                            scaler_path='scaler.pkl', checkpoint_path='market_condition_checkpoint.h5',
                                            epochs=100, steps_per_epoch=100, validation_steps=20):
        """
        Обучает и сохраняет ансамблевую модель классификации рыночных условий с кросс-валидацией,
        используя LSTM + GRU + Attention + XGBoost. Данные считываются чанками через генераторы
        (tf.data.Dataset), чтобы не загружать весь датасет в оперативную память.
        """
        # Вычисляем масштабировщик на представительном подмножестве (без объединения всех чанков)
        scaler, features = self.compute_scaler(data_path, sample_size=100000, chunk_size=10000)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Масштабировщик сохранён в {scaler_path}.")

        # Создаем тренировочный и валидационный dataset
        train_dataset = self.prepare_training_dataset(data_path, scaler, features, batch_size=32,
                                                        chunk_size=10000, split='train', train_fraction=0.8)
        val_dataset = self.prepare_training_dataset(data_path, scaler, features, batch_size=32,
                                                      chunk_size=10000, split='val', train_fraction=0.8)

        # Создание модели с входной формой (1, число признаков)
        input_shape = (1, len(features))
        with strategy.scope():
            final_model = self.build_lstm_gru_model(input_shape=input_shape)
            final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        ]

        history = final_model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        # Для финальной части: собираем небольшой тестовый набор из валидационного dataset
        X_test_list, y_test_list = [], []
        for X_batch, y_batch in val_dataset.take(validation_steps):
            X_test_list.append(X_batch.numpy())
            y_test_list.append(y_batch.numpy())
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # Извлекаем эмбеддинги из предпоследнего слоя (перед softmax)
        feature_extractor = tf.keras.models.Model(
            inputs=final_model.input,
            outputs=final_model.get_layer("embedding_layer").output
        )
        X_test_features = feature_extractor.predict(X_test)
        X_test_features = np.squeeze(X_test_features, axis=1)  # если нужно убрать лишнюю ось


        # Обучаем XGBoost на эмбеддингах; передаем X_test_features и y_test как eval_set
        xgb_model = self.train_xgboost(X_test_features, y_test, X_val=X_test_features, y_val=y_test)

        # Оценка финальной модели
        # Получаем предсказания от модели LSTM+GRU
        y_pred_nn_probs = final_model.predict(X_test)
        # Если выход имеет лишнюю размерность (например, (N, 1, 3)), удаляем её:
        if y_pred_nn_probs.ndim == 3:
            y_pred_nn_probs = np.squeeze(y_pred_nn_probs, axis=1)
        # Преобразуем вероятности в классы:
        y_pred_nn = np.argmax(y_pred_nn_probs, axis=1)

        # Получаем эмбеддинги с помощью feature_extractor
        X_test_features = feature_extractor.predict(X_test)
        # Если эмбеддинги имеют лишнюю размерность (например, (N, 1, d)), удаляем её:
        if X_test_features.ndim == 3:
            X_test_features = np.squeeze(X_test_features, axis=1)

        # Получаем предсказания от XGBoost
        y_pred_xgb_probs = xgb_model.predict_proba(X_test_features)
        y_pred_xgb = np.argmax(y_pred_xgb_probs, axis=1)

        # Проверяем, что оба предсказания имеют одинаковое число примеров
        assert y_pred_nn.shape[0] == y_pred_xgb.shape[0], f"Mismatch: {y_pred_nn.shape[0]} vs {y_pred_xgb.shape[0]}"

        # Объединяем предсказания (например, голосованием через mode)
        y_pred_ensemble = mode(np.vstack([y_pred_nn, y_pred_xgb]), axis=0)[0].flatten()


        y_pred_classes = y_pred_ensemble
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')


        logging.info(f"""
            Метрики финальной модели:
            Accuracy: {accuracy:.4f}
            Precision: {precision:.4f}
            Recall: {recall:.4f}
            F1-Score: {f1:.4f}
        """)
        
        
        # Сохранение модели только при высоком качестве
        '''
        if f1 >= 0.80:
            # Папка, где будут лежать модели (внутри контейнера RunPod)
            saved_models_dir = "/workspace/saved_models/Market_Classifier"
            os.makedirs(saved_models_dir, exist_ok=True)

            # Путь для LSTM-GRU модели
            model_path = os.path.join(saved_models_dir, "final_model.h5")
            final_model.save(model_path)
                
            # Путь для XGBoost-модели
            xgb_path = os.path.join(saved_models_dir, "classifier_xgb_model.pkl")
            joblib.dump(xgb_model, xgb_path)

            logging.info(f"Финальная модель LSTM-GRU сохранена в {model_path}")
            logging.info(f"XGBoost-модель сохранена в {xgb_path}")
            return final_model
        else:
            logging.warning("Финальное качество модели ниже порогового (80% F1-score). Модель не сохранена.")
            return None
            '''
        # Папка, где будут лежать модели (внутри контейнера RunPod)
        saved_models_dir = "/workspace/saved_models/Market_Classifier"
        os.makedirs(saved_models_dir, exist_ok=True)

        # Путь для LSTM-GRU модели
        model_path = os.path.join(saved_models_dir, "final_model.h5")
        final_model.save(model_path)
                
        # Путь для XGBoost-модели
        xgb_path = os.path.join(saved_models_dir, "classifier_xgb_model.pkl")
        joblib.dump(xgb_model, xgb_path)

        logging.info(f"Финальная модель LSTM-GRU сохранена в {model_path}")
        logging.info(f"XGBoost-модель сохранена в {xgb_path}")
        return final_model     




if __name__ == "__main__":
    # Инициализация стратегии (TPU или CPU/GPU)
    strategy = initialize_strategy()
    
    symbols = ['BTCUSDC', 'ETHUSDC']
    
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 1, 30)
    
    data_path = os.path.join("/workspace/data", "labeled_market_data.csv")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    model_path = os.path.join("/workspace/saved_models", "market_condition_classifier.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    scaler_path = os.path.join("/workspace/saved_models", "scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Создание экземпляра классификатора
    classifier = MarketClassifier()

    # Загрузка и разметка данных
    try:
        logging.info("Начало загрузки и разметки данных.")
        labeled_data = classifier.fetch_and_label_all(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            save_path="labeled_data"
        )
        labeled_data.to_csv(data_path, index=True)
        logging.info(f"Данные успешно сохранены в {data_path}.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        exit(1)

    # Обучение классификатора
    try:
        logging.info("Начало обучения классификатора.")
        classifier.train_market_condition_classifier(
            data_path=data_path,
            model_path=model_path,
            scaler_path=scaler_path
        )
        logging.info("Обучение завершено успешно.")
    except Exception as e:
        logging.error(f"Ошибка в процессе обучения: {e}")
        exit(1)

