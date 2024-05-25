import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
import pandas_datareader.data as pdr
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import ModelCheckpoint, LambdaCallback
from backtesting import Backtest, Strategy
import io
import contextlib



st.title('LSTM Model Stock Backtesting')
st.header('期末專題報告Demo · 第12組')
st.markdown('組長：  \n徐睿延 110099029')
st.markdown(f"組員：  \n陳冠倫 110072250  \n陳亮廷 110072224  \n宋宇然 110072206  \n張稚婕 111042013  \n賀守聖 111042038")

# import
code = '''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
import pandas_datareader.data as pdr

# 指標選擇用
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# 模型資料處理用
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump, load

#模型窗口設置用
from collections import deque

# 模型建立與訓練用
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import ModelCheckpoint, LambdaCallback

# 回測用
from backtesting import Backtest, Strategy

#建立Web應用程式用
import streamlit as st
'''
st.header("Importing Modules", divider='grey')
st.code(code, language='python')

#讀stock
#concat是將所有的股票數據按行合併成一個DF
#progress代表讀的時候不要有進度條
#assign是pandas對於DF的一個添加方法，添加股票代碼進到DF裡面
#選用2021-2023兩年資料進行建模，後續再用2023整年資料回測
with st.echo():
          stocks = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'META', 'GOOG', 'BRK-B', 
                    'LLY', 'JPM', 'AVGO', 'XOM', 'UNH', 'V', 'TSLA', 'PG', 'MA', 
                    'JNJ', 'HD', 'MRK', 'COST', 'ABBV', 'CVX', 'CRM', 'BAC', 'NFLX']
          
          startdate = "2021-01-01"
          enddate = "2023-01-01"
          
          data = pd.concat(
              [yf.download(stock, start=startdate, end=enddate, progress=False).assign(Stock=stock)
               for stock in stocks],
              axis=0
          )
st.header("Reading Stock Data for Modeling", divider='grey')
st.markdown("Choosing Top25 Companies in S&P500 by Index Weight")
st.dataframe(data)





#計算指標
code = '''
# RSI 相對強弱指標
data['Close_delta'] = data['Close'].diff()
data['Up'] = data['Close_delta'].clip(lower=0)
data['Down'] = -1 * data['Close_delta'].clip(upper=0)
data['Ma_up'] = data['Up'].transform(lambda x: x.rolling(window=14).mean())
data['Ma_down'] = data['Down'].transform(lambda x: x.rolling(window=14).mean())
data['RSI'] = data['Ma_up'] / (data['Ma_up'] + data['Ma_down']) * 100

# MACD 平滑異同移動平均線指標 (背離指標)
data['Exp1'] = data['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
data['Exp2'] = data['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
data['Macd'] = data['Exp1'] - data['Exp2'] # Moving Average Convergence Divergence
data['Macdsig'] = data['Macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean()) # MACD Signal Line
data['Macdhist'] = data['Macd'] - data['Macdsig'] # MACD Histogram

# Momentum 動能指標
data['Momentum'] = data['Close'].transform(lambda x: x.diff(periods=15))

# ATR 真實波動幅度均值
data['High_low'] = data['High'] - data['Low']
data['High_close'] = abs(data['High'] - data['Close'].shift())
data['Low_close'] = abs(data['Low'] - data['Close'].shift())
data['Tr'] = data[['High_low', 'High_close', 'Low_close']].max(axis=1)
data['ATR'] = data['Tr'].rolling(window=14).mean()

# ROC 價格變動率
data['ROC'] = data['Close'].pct_change(periods=10) * 100

# DMI 動向指標
data['+DM'] = np.where((data['High'] - data['High'].shift()) > (data['Low'].shift() - data['Low']), data['High'] - data['High'].shift(), 0)
data['-DM'] = np.where((data['Low'].shift() - data['Low']) > (data['High'] - data['High'].shift()), data['Low'].shift() - data['Low'], 0)
data['+DM'] = data['+DM'].where(data['+DM'] > data['-DM'], 0)
data['-DM'] = data['-DM'].where(data['-DM'] > data['+DM'], 0)
data['TR'] = data[['High_low', 'High_close', 'Low_close']].max(axis=1)
data['+DI'] = 100 * data['+DM'].rolling(window=14).sum() / data['TR'].rolling(window=14).sum()
data['-DI'] = 100 * data['-DM'].rolling(window=14).sum() / data['TR'].rolling(window=14).sum()
data['ADX'] = abs((data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']) * 100).rolling(window=14).mean()

# VWAP 成交量加權平均
data['Cumulative_Volume'] = data['Volume'].cumsum()
data['Cumulative_Volume_Price'] = (data['Close'] * data['Volume']).cumsum()
data['VWAP'] = data['Cumulative_Volume_Price'] / data['Cumulative_Volume']

# AD-line 漲跌趨勢線指標
data['Advance'] = data.groupby('Stock')['Close'].diff().apply(lambda x: 1 if x > 0 else 0)
data['Decline'] = data.groupby('Stock')['Close'].diff().apply(lambda x: 1 if x < 0 else 0)
data['AD_Line_pre'] = data['Advance'] - data['Decline']
data['AD_Line'] = data['AD_Line_pre'].cumsum()

# EMA 指數平滑移動平均線
data['EMA_3'] = data['Close'].ewm(span=3, adjust=False).mean()
data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
# 前一天的EMA
data['EMA_9_prev'] = data['EMA_9'].shift(1)
data['EMA_12_prev'] = data['EMA_12'].shift(1)
data['EMA_20_prev'] = data['EMA_20'].shift(1)
data['EMA_50_prev'] = data['EMA_50'].shift(1)
data['EMA_26_prev'] = data['EMA_26'].shift(1)
#信號
data['912EMA_Signal'] = 0
data['2050EMA_Signal'] = 0
data['1226EMA_Signal'] = 0
# 計算 EMA 9 和 EMA 12 之間的交叉信號
data.loc[(data['EMA_9_prev'] < data['EMA_12_prev']) & (data['EMA_9'] > data['EMA_12']), '912EMA_Signal'] = 1
data.loc[(data['EMA_9_prev'] > data['EMA_12_prev']) & (data['EMA_9'] < data['EMA_12']), '912EMA_Signal'] = -1
# 計算 EMA 20 和 EMA 50 之間的交叉信號
data.loc[(data['EMA_20_prev'] < data['EMA_50_prev']) & (data['EMA_20'] > data['EMA_50']), '2050EMA_Signal'] = 1
data.loc[(data['EMA_20_prev'] > data['EMA_50_prev']) & (data['EMA_20'] < data['EMA_50']), '2050EMA_Signal'] = -1
# 計算 EMA 12 和 EMA 26 之間的交叉信號
data.loc[(data['EMA_12_prev'] < data['EMA_26_prev']) & (data['EMA_12'] > data['EMA_26']), '1226EMA_Signal'] = 1
data.loc[(data['EMA_12_prev'] > data['EMA_26_prev']) & (data['EMA_12'] < data['EMA_26']), '1226EMA_Signal'] = -1


# Stochastic Oscillator 隨機指標(KD)
data['Low_n'] = data['Low'].rolling(window=14).min() #常用週期14天
data['High_n'] = data['High'].rolling(window=14).max()
data['%K'] = 100 * ((data['Close'] - data['Low_n']) / (data['High_n'] - data['Low_n']))
data['%D3'] = data['%K'].rolling(window=3).mean() #短三日SMA
data['%D5'] = data['%K'].rolling(window=5).mean() #短五日SMA
data['KD_Signal_3'] = data['KD_Signal_5'] = 0
data.loc[(data['%K'].shift(1) < data['%D3'].shift(1)) & (data['%K'] > data['%D3']), 'KD_Signal_3'] = 1
data.loc[(data['%K'].shift(1) > data['%D3'].shift(1)) & (data['%K'] < data['%D3']), 'KD_Signal_3'] = -1
data.loc[(data['%K'].shift(1) < data['%D5'].shift(1)) & (data['%K'] > data['%D5']), 'KD_Signal_5'] = 1
data.loc[(data['%K'].shift(1) > data['%D5'].shift(1)) & (data['%K'] < data['%D5']), 'KD_Signal_5'] = -1
'''
st.header("Calculating Indicators", divider='grey')
st.code(code, language='python')

#加權指標
# 由於指標會出現NA，所以要處理掉 (比如十天平均，前九天就不會有資料)
# 給label，如果明天股價高於今天，就給label=1，反之=0
code='''
# Cleaning NA
data.ffill(inplace=True)
data.dropna(inplace=True)

# Labeling
data['Label'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
y = data['Label']

# Scaling
indicator_candidate = data[['Volume', 'RSI', 'Macdhist','Momentum','ATR',
                            'ROC','ADX','VWAP','AD_Line','EMA_9','EMA_50']]
indicator_scaler = StandardScaler()
indicator_candidate_scaled = indicator_scaler.fit_transform(indicator_candidate)
'''
st.header("Scaling Indicators", divider='grey')
st.code(code, language='python')


#指標隨機森林
#selected_features = ['Volume', 'RSI', 'Macdhist', 'EMA_3', 'EMA_50']
code = '''
indicator_model = RandomForestClassifier()
indicator_model.fit(indicator_candidate_scaled, y)
feature_importances = indicator_model.feature_importances_
feature_names = indicator_candidate.columns

features = list(zip(feature_names, feature_importances))
sorted_features = sorted(features, key=lambda x: x[1], reverse=True)

for feature, importance in sorted_features:
    print(f'Feature: {feature}, Importance: {importance}')

feature_data = data[selected_features+ ['Label']]

'''
st.header("Picking Indicators using Random Forest", divider='grey')
st.code(code, language='python')


#pre modeling
code = '''
def pre_model_data_processing(feature_data, mem_days):
    scaler = StandardScaler()
    sca_X = scaler.fit_transform(feature_data.iloc[:,:-1])
    dump(scaler, 'scaler.joblib')

    mem = mem_days
    deq = deque(maxlen=mem)

    X = []
    for i in sca_X:
        deq.append(list(i))
        if len(deq)==mem:
            X.append(list(deq))

    y = feature_data["Label"].values[mem-1:]
    X=np.array(X)
    y=np.array(y)
    return X,y
'''
st.header("Pre-model Data Process", divider='grey')
st.code(code, language='python')

#modeling
code = '''
mem_days = [10,15,20,25,30] # 模型內存天數
lstm_layers = [1,2] # LSTM層數量
dense_layers = [1,2] # Dense層數量
units = [32,64,128,256] # 每一層的神經元數量
dropouts = [0.1,0.2,0.3] # Dropout率
batch_sizes = [32,64] # 批量處理大小


# 訓練計數器(查找目前作業進度用)
total_iterations = len(mem_days)*len(lstm_layers)*len(dense_layers)*len(units)*len(dropouts)*len(batch_sizes)
current_iteration = 0
folder = 'folder_path/'
best_accuracy = 0.0
best_filepath = ""

for the_mem_day in mem_days:
    for the_lstm_layer in lstm_layers:
        for the_dense_layer in dense_layers:
            for the_unit in units:
                for dropout_rate in dropouts:
                    for the_batch_size in batch_sizes:

                        # print目前作業進度
                        current_iteration += 1
                        print(f"Progress: {current_iteration}/{total_iterations}")

                        # 輸入整理好的xy資料，訓練80%測試20%
                        X, y = pre_model_data_processing(feature_data, the_mem_day)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

                        # 建立模型，指定初始輸入層X
                        model = Sequential()
                        model.add(Input(shape=X.shape[1:]))

                        # LSTM層循環
                        for i in range(the_lstm_layer):
                            # return sequence，如果最後一層就不return了
                            return_sequences = True if i < the_lstm_layer - 1 else False
                            model.add(LSTM(the_unit, activation='tanh', return_sequences=return_sequences))
                            # 添加dropout防止over-fitting
                            model.add(Dropout(dropout_rate))

                        # Dense層循環
                        for i in range(the_dense_layer):
                            model.add(Dense(the_unit, activation='relu'))
                            # 添加dropout防止over-fitting
                            model.add(Dropout(dropout_rate))

                        # 輸出層，使用sigmoid將結果進行二分
                        model.add(Dense(1, activation='sigmoid'))

                        # 指定模型之優化器、損失函數和評估指標
                        model.compile(optimizer='adam', loss='Focal Loss', metrics=['accuracy'])

                    
                        # 配置模型保存的路徑，這裡先暫存，稍後重命名
                        checkpoint_path = os.path.join(folder, "temp_best_model.keras")
                        checkpoint = ModelCheckpoint(
                            filepath=checkpoint_path,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max',
                            save_weights_only=False
                        )
                        
                        
                        # 以不同的批量訓練模型
                        history = model.fit(X_train, y_train, batch_size=the_batch_size, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpoint])
                        
                        best_val_acc = max(history.history['val_accuracy'])
                        if best_val_acc > best_accuracy:
                            best_accuracy = best_val_acc
                            new_path = os.path.join(folder, f"val{best_val_acc:.4f}_mem{the_mem_day}_lstm{the_lstm_layer}_dense{the_dense_layer}_unit{the_unit}_dropout{dropout_rate:.2f}_batch{the_batch_size}.keras")
                            os.rename(checkpoint_path, new_path)
                            best_filepath = new_path                       
                        
                        
print("------------End of Training------------")
'''
st.header("LSTM modeling", divider='grey')
st.code(code, language='python')

#loading model

code = '''
model_path = "val05584_mem25_lstm1_dense2_unit256_dropout010_batch32.keras"
model = load_model(model_path)
model.summary()
'''
st.header("Loading Model for Backtest", divider='grey')
st.code(code, language='python')

model_path = "val05584_mem25_lstm1_dense2_unit256_dropout010_batch32.keras"
try:
    model = load_model(model_path)    
    string_io = io.StringIO()
    with contextlib.redirect_stdout(string_io):
        model.summary()
    summary_string = string_io.getvalue()
    st.text(summary_string)
    
except ValueError as e:
    st.error(f"Error loading the model: {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")


#def stock indicator
code = '''
def calculate_selected_indicators(data):
    delta = data['Close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.rolling(window=14).mean()
    ma_down = down.rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + ma_up / ma_down))

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    data['Macdhist'] = macd - macd.ewm(span=9, adjust=False).mean()

    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    data.dropna(inplace=True)
    return data
'''
st.header("Backtesting Stocks' Indicators", divider='grey')
st.code(code, language='python')


#BTEST
code = '''
class LSTMBasedStrategy(Strategy):
    def init(self):
        self.prediction = self.I(lambda x: x, full_predictions)

    def next(self):
        if self.prediction[-1] == 1 and not self.position.is_long:
            self.buy()
        elif self.prediction[-1] == 0 and not self.position.is_short:
            self.sell()
'''
st.header("Model Backtesting", divider='grey')
st.code(code, language='python')

#testing all stocks
code = '''
stocks = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'META', 'GOOG', 'BRK-B', 
          'LLY', 'JPM', 'AVGO', 'XOM', 'UNH', 'V', 'TSLA', 'PG', 'MA', 
          'JNJ', 'HD', 'MRK', 'COST', 'ABBV', 'CVX', 'CRM', 'BAC','NFLX']
startdate = "2023-01-15"
enddate = "2024-01-15"
mem_days = 25
results_df=pd.DataFrame([])

for stock in stocks: 
    df = yf.download(stock, start=startdate, end=enddate, progress=False)
    df = calculate_selected_indicators(df)
    df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI', 'Macdhist', 'EMA_9', 'EMA_50', 'Volume']  

    Backtest_scaler = StandardScaler()
    Backtest_scaler.fit(df[features])
    scaled_features = Backtest_scaler.transform(df[features])

    X = np.array([scaled_features[i:i + mem_days] for i in range(len(scaled_features) - mem_days + 1)])

    predictions = model.predict(X)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    full_predictions = np.zeros(len(df))
    full_predictions[mem_days-1:mem_days-1+len(predicted_classes)] = predicted_classes

    bt = Backtest(df, LSTMBasedStrategy, cash=10000, commission=.0425)
    results = bt.run()
    new_df = pd.DataFrame([results])
    new_df['ID'] = stock
    new_df.insert(0, 'ID', new_df.pop('ID'))
    results_df = pd.concat([results_df, new_df], ignore_index=True)

'''
st.header("Backtesting Stocks", divider='grey')
st.code(code, language='python')


# model = load_model("val05584_mem25_lstm1_dense2_unit256_dropout010_batch32.keras")
# def calculate_selected_indicators(data):
#     delta = data['Close'].diff()
#     up, down = delta.clip(lower=0), -delta.clip(upper=0)
#     ma_up = up.rolling(window=14).mean()
#     ma_down = down.rolling(window=14).mean()
#     data['RSI'] = 100 - (100 / (1 + ma_up / ma_down))

#     exp1 = data['Close'].ewm(span=12, adjust=False).mean()
#     exp2 = data['Close'].ewm(span=26, adjust=False).mean()
#     macd = exp1 - exp2
#     data['Macdhist'] = macd - macd.ewm(span=9, adjust=False).mean()

#     data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
#     data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

#     data.dropna(inplace=True)
#     return data

# class LSTMBasedStrategy(Strategy):
#     def init(self):
#         self.prediction = self.I(lambda x: x, full_predictions)

#     def next(self):
#         if self.prediction[-1] == 1 and not self.position.is_long:
#             self.buy()
#         elif self.prediction[-1] == 0 and not self.position.is_short:
#             self.sell()

# stock = 'MSFT'
# startdate = "2023-01-15"
# enddate = "2024-01-15"
# mem_days = 25
# results_df = pd.DataFrame([])

# df = yf.download(stock, start=startdate, end=enddate, progress=False)
# df = calculate_selected_indicators(df)
# df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
# features = ['RSI', 'Macdhist', 'EMA_9', 'EMA_50', 'Volume']  

# Backtest_scaler = StandardScaler()
# Backtest_scaler.fit(df[features])
# scaled_features = Backtest_scaler.transform(df[features])

# X = np.array([scaled_features[i:i + mem_days] for i in range(len(scaled_features) - mem_days + 1)])

# predictions = model.predict(X)
# predicted_classes = (predictions > 0.5).astype(int).flatten()

# full_predictions = np.zeros(len(df))
# full_predictions[mem_days-1:mem_days-1+len(predicted_classes)] = predicted_classes

# bt = Backtest(df, LSTMBasedStrategy, cash=10000, commission=.0425)
# results = bt.run()
# btplt = bt.plot()

# st.bokeh_chart(btplt)



