import streamlit as st

st.title('LSTM Model Stock Backtesting')
st.header('期末專題報告Demo · 第12組')
st.markdown('組長：  \n徐睿延 110099029')
st.markdown(f"組員：  \n陳冠倫 110072250  \n陳亮廷 110072224  \n宋宇然 110072206  \n張稚婕 111042013  \n賀守聖 111042038")

# import
st.header("Importing Modules", divider='grey')
with st.echo():
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import yfinance as yf
    yf.pdr_override()
    import pandas_datareader.data as pdr
    import io
    
    # 指標選擇用
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    
    # 模型資料處理用
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    #模型窗口設置用
    from collections import deque
    
    # 模型建立與訓練用
    from keras.models import Sequential, load_model
    from keras.layers import Dense, LSTM, Dropout, Input
    from keras.callbacks import ModelCheckpoint, LambdaCallback
    import contextlib
    
    # 回測用
    from backtesting import Backtest, Strategy
    

#讀stock
#concat是將所有的股票數據按行合併成一個DF
#progress代表讀的時候不要有進度條
#assign是pandas對於DF的一個添加方法，添加股票代碼進到DF裡面
#選用2021-2023兩年資料進行建模，後續再用2023整年資料回測
st.header("Reading Stock Data for Modeling", divider='grey')
st.markdown("Choosing Top25 Companies in S&P500 by Index Weight")
with st.echo():
    # List of stocks
    stocks = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'META', 'GOOG', 'BRK-B', 
              'LLY', 'JPM', 'AVGO', 'XOM', 'UNH', 'V', 'TSLA', 'PG', 'MA', 
              'JNJ', 'HD', 'MRK', 'COST', 'ABBV', 'CVX', 'CRM', 'BAC', 'NFLX']

    # Start and end dates
    startdate = "2021-01-01"
    enddate = "2023-01-01"

    data = pd.concat(
        [yf.download(stock, start=startdate, end=enddate, progress=False).assign(Stock=stock)
         for stock in stocks],
        axis=0
    )
st.code('print(data)',language='python')
st.dataframe(data)





#計算指標
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
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

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
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

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
st.code(code,language='python')
st.code('print(data)',language='python')
st.dataframe(data)


#加權指標
# 由於指標會出現NA，所以要處理掉 (比如十天平均，前九天就不會有資料)
# 給label，如果明天股價高於今天，就給label=1，反之=0
st.header("Scaling Indicators", divider='grey')
with st.echo():
# Cleaning NA
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    
    # Labeling
    data['Label'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    y = data['Label']
    
    # Scaling
    indicator_candidate = data[['Volume', 'RSI', 'Macdhist','Momentum','ATR',
                                'ROC','ADX','VWAP','AD_Line','EMA_3','EMA_50',
                                'KD_Signal_3','KD_Signal_5']]

    indicator_scaler = StandardScaler()
    indicator_candidate_scaled = indicator_scaler.fit_transform(indicator_candidate)
st.code('print(indicator_candidate_scaled)',language='python')
st.dataframe(indicator_candidate_scaled)


#指標隨機森林
#selected_features = ['Volume', 'RSI', 'Macdhist', 'EMA_3', 'EMA_50']
code = '''
indicator_model = RandomForestClassifier()

# 使用 RFE 選取前五個最重要的特徵
selector = RFE(indicator_model, n_features_to_select=5, step=1)
selector = selector.fit(indicator_candidate_scaled, y)
selected_features = selector.support_
print("Selected features:", indicator_candidate.columns[selected_features])
feature_data = data[indicator_candidate.columns[selected_features].tolist() + ['Label']]

'''
st.header("Picking Indicators using Random Forest", divider='grey')
st.code(code, language='python')
st.code(f"Selected features: Index(['Volume', 'RSI', 'Macdhist', 'EMA_3', 'EMA_50'], dtype='object')")


#pre modeling
code = '''
def pre_model_data_processing(feature_data, mem_days):
    scaler = StandardScaler()
    sca_X = scaler.fit_transform(feature_data.iloc[:,:-1])

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

                        # 輸入整理好的xy資料，訓練70%
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
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                    
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
                        history = model.fit(X_train, y_train, batch_size=the_batch_size, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])
                        
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
#設定標的
stocks = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'META', 'GOOG', 'BRK-B', 
          'LLY', 'JPM', 'AVGO', 'XOM', 'UNH', 'V', 'TSLA', 'PG', 'MA', 
          'JNJ', 'HD', 'MRK', 'COST', 'ABBV', 'CVX', 'CRM', 'BAC','NFLX']
#設定日期範圍
startdate = "2023-01-15"
enddate = "2024-01-15"

'''
st.header("load stocks to test", divider='grey')
st.code(code, language='python')


code='''
#計算指標放進data
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
    return data
'''
st.header("Backtesting Stocks' Indicators", divider='grey')
st.code(code, language='python')


#BTEST
code = '''
class LSTMBasedStrategy(Strategy):
    def init(self):
        self.consecutive_buys = 0 
        self.consecutive_sells = 0 
        self.total_invested = 0 # 記錄總投資金額
        self.total_shorted = 0 #  記錄總做空金額
        self.prediction = self.I(lambda x: x, full_predictions)

    def next(self):
        current_price = self.data.Close[-1]
        available_equity = self.equity * 0.2  # 可交易資金(總資產的20%)
        initial_position_size = max(1, int(self.equity / 10 / current_price))  # Ensure size is at least 1

        if self.prediction[-1] == 1:
            if self.position.is_long:
                additional_size = int(min(self.position.size * 0.2, (available_equity - self.total_invested) / current_price))
                if additional_size > 0 and self.equity - self.total_invested >= additional_size * current_price:
                    self.buy(size=additional_size)
                    self.total_invested += additional_size * current_price
            else:
                if initial_position_size > 0 and self.equity - self.total_invested >= initial_position_size * current_price:
                    self.buy(size=initial_position_size)
                    self.total_invested = initial_position_size * current_price
                    self.consecutive_buys = 1
            self.consecutive_sells = 0
            self.total_shorted = 0

        elif self.prediction[-1] == 0:
            if self.position.is_short:
                additional_size = int(min(abs(self.position.size) * 0.2, (available_equity - self.total_shorted) / current_price))
                if additional_size > 0 and self.equity - self.total_shorted >= additional_size * current_price:
                    self.sell(size=additional_size)
                    self.total_shorted += additional_size * current_price
            else:
                if initial_position_size > 0 and self.equity - self.total_shorted >= initial_position_size * current_price:
                    self.sell(size=initial_position_size)
                    self.total_shorted = initial_position_size * current_price
                    self.consecutive_sells = 1
            self.consecutive_buys = 0
            self.total_invested = 0

        else:
            # Reset counters and investment totals on signals that are not buy or sell
            self.consecutive_buys = 0
            self.consecutive_sells = 0
            self.total_invested = 0
            self.total_shorted = 0

        # Close positions if opposite signals or conditions require
        if self.position.is_long and self.prediction[-1] == 0:
            self.sell(size=self.position.size)
        elif self.position.is_short and self.prediction[-1] == 1:
            self.buy(size=abs(self.position.size))
'''
st.header("Model Backtesting", divider='grey')
st.code(code, language='python')

#testing all stocks
code = '''
results_df=pd.DataFrame([])
for stock in stocks:
    df = pd.concat(
        [yf.download(stock, start=startdate, end=enddate, progress=False).assign(Stock=stock)
        ],
        axis=0)

    df = calculate_selected_indicators(df)

    df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    features = ['RSI', 'Macdhist', 'EMA_9', 'EMA_50', 'Volume']  

    scaler = StandardScaler()
    scaler.fit(df[features])
    scaled_features = scaler.transform(df[features])

    mem_days = 25  

    X = np.array([scaled_features[i:i + mem_days] for i in range(len(scaled_features) - mem_days + 1)])

    predictions = model.predict(X)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    full_predictions = np.zeros(len(df))
    full_predictions[mem_days-1:mem_days-1+len(predicted_classes)]
    
    bt = Backtest(df, LSTMBasedStrategy, cash=10000, commission=.002)
        results = bt.run()
        
        #將個別的結果放入results_df
        new_df = pd.DataFrame([results])
        new_df['ID']=stock
        results_df = pd.concat([results_df, new_df], ignore_index=True)
        print(results_df)
'''
st.header("Backtesting Stocks", divider='grey')
st.code(code, language='python')
df = pd.read_csv('results.csv')
st.dataframe(df)





