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
from utils_output import ensure_directory, copy_output, save_model_output



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
        """Расширенный набор индикаторов для точной классификации с использованием pandas_ta"""
        # Базовые индикаторы
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
        data['adx'] = ta.adx(data['high'], data['low'], data['close'], length=14)['ADX_14']
        
        # Множественные MA для определения силы тренда
        for period in [10, 20, 50, 100]:
            data[f'sma_{period}'] = ta.sma(data['close'], length=period)
            data[f'ema_{period}'] = ta.ema(data['close'], length=period)
        
        # Импульсные индикаторы
        data['rsi'] = ta.rsi(data['close'], length=14)
        macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        data['macd'], data['macd_signal'], data['macd_hist'] = macd['MACD_12_26_9'], macd['MACDs_12_26_9'], macd['MACDh_12_26_9']
        data['willr'] = ta.willr(data['high'], data['low'], data['close'], length=14)
        
        # Волатильность
        bb = ta.bbands(data['close'], length=20, std=2)
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = bb['BBU_20_2.0'], bb['BBM_20_2.0'], bb['BBL_20_2.0']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Объемные индикаторы
        data['obv'] = ta.obv(data['close'], data['volume'])
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['volume_std'] = data['volume'].rolling(window=20).std()
        
        # Трендовые индикаторы
        data['trend_strength'] = abs(data['close'].pct_change().rolling(window=20).sum())
        data['price_momentum'] = data['close'].diff(periods=10) / data['close'].shift(10)
        
        # Динамические уровни поддержки/сопротивления
        data['support_level'] = data['low'].rolling(window=20).min()
        data['resistance_level'] = data['high'].rolling(window=20).max()
        
        return data

    
    
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
        adx = data['adx'].shift(1).iloc[-1]
        rsi = data['rsi'].shift(1).iloc[-1]
        macd_hist = data['macd_hist'].shift(1).iloc[-1]
        willr = data['willr'].shift(1).iloc[-1]
        volume_ratio = data['volume_ratio'].iloc[-1]
        price = data['close'].iloc[-1]
        support = data['support_level'].shift(1).iloc[-1]
        resistance = data['resistance_level'].shift(1).iloc[-1]
        
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
        Загружает и обрабатывает данные для обучения.
        """
        logging.info(f"Загрузка данных из файла {data_path}")
        try:
            data = pd.read_csv(data_path, index_col=0)
        except FileNotFoundError:
            logging.error(f"Файл {data_path} не найден.")
            raise

        # Проверка наличия необходимых столбцов
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'adx', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'willr', 'bb_width', 'volume_ratio', 'trend_strength',
            'market_type'
        ]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Отсутствует необходимый столбец: {col}")

        # Удаление строк с пропущенными значениями
        data.dropna(inplace=True)

        # Проверка значений market_type до обработки
        logging.info(f"Уникальные значения market_type до преобразования: {data['market_type'].unique()}")

        # Удаление выбросов (только один вызов!)
        logging.info("Удаление выбросов из данных...")
        data = self.remove_outliers(data)
        logging.info(f"Размер данных после удаления выбросов: {data.shape}")

        # Преобразование меток рынка в числовые значения
        label_mapping = {'bullish': 0, 'bearish': 1, 'flat': 2}
        data['market_type'] = data['market_type'].map(label_mapping)

        # Удаление строк с NaN после преобразования
        if data['market_type'].isna().any():
            bad_values = data[data['market_type'].isna()]['market_type'].unique()
            logging.error(f"Обнаружены NaN значения в market_type! Исходные значения: {bad_values}")
            data.dropna(subset=['market_type'], inplace=True)

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
        """Создаёт мощный ансамбль LSTM + GRU с Attention"""
        inputs = Input(shape=input_shape)

        # LSTM блок
        lstm_out = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(lstm_out)

        # GRU блок
        gru_out = GRU(256, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
        gru_out = Dropout(0.3)(gru_out)
        gru_out = GRU(128, return_sequences=True, kernel_regularizer=l2(0.01))(gru_out)
        gru_out = Dropout(0.3)(gru_out)
        gru_out = GRU(64, return_sequences=True, kernel_regularizer=l2(0.01))(gru_out)

        # Объединяем выходы LSTM и GRU
        combined = Concatenate()([lstm_out, gru_out])

        # Attention-механизм
        attention = Attention()(combined)

        # Финальные Dense-слои
        x = LSTM(64, return_sequences=False)(attention)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        outputs = Dense(3, activation='softmax')(x)  # 3 класса: bullish, bearish, flat

        model = tf.keras.models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


    def train_xgboost(self, X_train, y_train):
        """Обучает XGBoost на эмбеддингах LSTM + GRU, или возвращает DummyClassifier, если в y_train только один класс."""
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logging.warning("В обучающем наборе XGB обнаружен только один класс. Используем DummyClassifier.")
            dummy = DummyClassifier(strategy='constant', constant=unique_classes[0])
            dummy.fit(X_train, y_train)
            return dummy
        else:
            booster = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                learning_rate=0.1,
                n_estimators=10,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=1
            )
            booster.fit(X_train, y_train)
            return booster


    def build_ensemble(X_train, y_train):
        """Финальный ансамбль: LSTM + GRU + XGBoost"""
        lstm_gru_model = build_lstm_gru_model((X_train.shape[1], X_train.shape[2]))

        # Обучаем LSTM-GRU
        lstm_gru_model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)#epochs=100

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
                                          scaler_path='scaler.pkl', checkpoint_path='market_condition_checkpoint.h5'):
        """
        Обучает и сохраняет ансамблевую модель классификации рыночных условий с кросс-валидацией.
        Теперь использует LSTM + GRU + Attention + XGBoost и требует минимум 85% точности перед финальным обучением.
        """
        # Подготовка данных
        X, y = self.prepare_training_data(data_path)

        # Масштабирование данных
        scaler = RobustScaler()  # ✅ Исправлено (теперь RobustScaler правильно импортируется)
        if os.path.exists(scaler_path):
            logging.info(f"Загружается существующий масштабировщик из {scaler_path}.")
            scaler = joblib.load(scaler_path)
        else:
            logging.info("Создаётся новый масштабировщик.")
            scaler.fit(X)
            joblib.dump(scaler, scaler_path)
            logging.info(f"Масштабировщик сохранён в {scaler_path}.")

        X_scaled = scaler.transform(X)

        # Кросс-валидация для оценки модели
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Подготовка данных для LSTM
            X_train_fold = np.expand_dims(X_train_fold, axis=1)
            X_val_fold = np.expand_dims(X_val_fold, axis=1)

            logging.info(f"Обучение на фолде {fold + 1}")
            logging.info(f"Shape X_train_fold: {X_train_fold.shape}")
            logging.info(f"Shape y_train_fold: {y_train_fold.shape}")

            # Создание и обучение модели для текущего фолда
            with strategy.scope():
                model_fold = self.build_lstm_gru_model(input_shape=(X_train_fold.shape[1], X_train_fold.shape[2]))  # ✅ Используем обновленную модель с GRU + Attention

                model_fold.fit(
                    X_train_fold, y_train_fold,
                    epochs=50,#50
                    batch_size=64,
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=1
                )

            # Оценка на валидации
            y_val_pred = model_fold.predict(X_val_fold)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)

            val_acc = accuracy_score(y_val_fold, y_val_pred_classes)
            val_f1 = f1_score(y_val_fold, y_val_pred_classes, average='weighted')

            fold_scores.append(val_acc)
            f1_scores.append(val_f1)

        # Финальная проверка перед обучением последней версии модели
        avg_accuracy = np.mean(fold_scores)
        avg_f1_score = np.mean(f1_scores)

        logging.info(f"Средняя точность на кросс-валидации: {avg_accuracy:.4f}")
        logging.info(f"Средний F1-score на кросс-валидации: {avg_f1_score:.4f}")

        if avg_accuracy < 0.85 or avg_f1_score < 0.80:
            logging.warning("Качество модели ниже 85% точности или 80% F1-score. Доработайте архитектуру.")
            return None  # Если качество низкое, не продолжаем

        # Финальное обучение на всей выборке
        X_scaled = np.expand_dims(X_scaled, axis=1)  # Добавляем временную ось для LSTM

        # Разделение данных для финальной модели
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Балансировка классов
        class_weights = self.balance_classes(y_train)

        # Создание финальной модели
        with strategy.scope():
            final_model = self.build_lstm_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))  # ✅ Используем LSTM + GRU + Attention

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
            ]

            history = final_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=200,#200
                batch_size=64,
                class_weight=class_weights,
                callbacks=callbacks
            )

            # **Обучаем XGBoost на эмбеддингах LSTM + GRU**
            feature_extractor = tf.keras.models.Model(
                inputs=final_model.input, outputs=final_model.layers[-3].output  # ✅ Берем эмбеддинги перед softmax
            )
            X_train_features = feature_extractor.predict(X_train)
            X_test_features = feature_extractor.predict(X_test)

            xgb_model = self.train_xgboost(X_train_features, y_train)  # ✅ Обучаем XGBoost

            # **Оценка финальной модели**
            y_pred_lstm_gru = final_model.predict(X_test)
            y_pred_xgb = xgb_model.predict(X_test_features)

            # **Ансамбль голосованием**
            y_pred_classes = np.argmax(y_pred_lstm_gru, axis=1) * 0.5 + y_pred_xgb * 0.5
            y_pred_classes = np.round(y_pred_classes).astype(int)

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
            if f1 >= 0.80:
                # Папка, где будут лежать модели (внутри контейнера RunPod)
                saved_models_dir = "/workspace/saved_models/Market_Classifier"
                os.makedirs(saved_models_dir, exist_ok=True)

                # Путь для LSTM-GRU модели
                model_path = os.path.join(saved_models_dir, "final_model.h5")
                final_model.save(model_path)
                
                # Путь для XGBoost-модели
                xgb_path = os.path.join(saved_models_dir, "bullish_xgb_model.pkl")
                joblib.dump(xgb_model, xgb_path)

                logging.info(f"Финальная модель LSTM-GRU сохранена в {model_path}")
                logging.info(f"XGBoost-модель сохранена в {xgb_path}")
                return final_model
            else:
                logging.warning("Финальное качество модели ниже порогового (80% F1-score). Модель не сохранена.")
                return None



if __name__ == "__main__":
    # Инициализация стратегии (TPU или CPU/GPU)
    strategy = initialize_strategy()
    
    
    symbols = ['BTCUSDC', 'ETHUSDC']
    
    start_date = datetime(2019, 7, 1)
    end_date = datetime(2024, 8, 1)
    
    data_path = os.path.join("/workspace/data", "labeled_market_data.csv")
    model_path = os.path.join("/workspace/saved_models", "market_condition_classifier.h5")
    scaler_path = os.path.join("/workspace/saved_models", "scaler.pkl")


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


