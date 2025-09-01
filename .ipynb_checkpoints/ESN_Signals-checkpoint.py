import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyESN import ESN

def esn_signals(train_df: pd.DataFrame, test_df: pd.DataFrame, Technical_Signals: list,
                n_reservoir: int = 200, spectral_radius: float = 0.95, sparsity: float = 0.1,
                signal_threshold: float = 0.5, random_state: int = 42):
    """
    ESN 모델을 학습하고 매수/매도 신호를 생성합니다.

    Args:
        train_df (pd.DataFrame): ESN 학습에 사용할 학습 데이터 (Technical_Signals 포함).
        test_df (pd.DataFrame): ESN이 신호를 생성할 테스트 데이터 (Technical_Signals 포함).
                                백테스팅의 실제 기간 데이터와 일치해야 합니다.
        Technical_Signals (list): ESN 입력으로 사용할 기술적 신호 컬럼 이름 리스트.
        n_reservoir (int): ESN의 Reservoir 크기.
        spectral_radius (float): Reservoir의 Spectral Radius.
        sparsity (float): Reservoir의 희소성.
        signal_threshold (float): ESN 예측값을 신호로 변환할 임계값.
        random_state (int): 난수 시드.

    Returns:
        pd.DataFrame: ESN 모델이 test_df에 대해 생성한 매수/매도 신호 DataFrame.
                      'Close', 'Predicted_Signals' 컬럼을 포함합니다.
    """

    # 1. 학습 데이터 준비
    features = Technical_Signals + ['Close']

    train_df_copy = train_df.copy()
    train_df_copy['Target_cpm_point_type'] = train_df_copy['cpm_point_type'].shift(-1)

    df_esn_train = train_df_copy.dropna(subset=features + ['Target_cpm_point_type'])

    # 학습 데이터가 충분하지 않을 경우 즉시 빈 DataFrame 반환
    if df_esn_train.empty:
        return pd.DataFrame(columns=['Close', 'Predicted_Signals']), {}

    X_train_raw = df_esn_train[features].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    y_train = df_esn_train['Target_cpm_point_type'].values.astype(float)

    # 2. ESN 모델 초기화 및 학습
    n_inputs = X_train.shape[1]
    n_outputs = 1

    esn_model = ESN(n_inputs=n_inputs, n_outputs=n_outputs, n_reservoir=n_reservoir,
                    spectral_radius=spectral_radius, sparsity=sparsity,
                    input_scaling=1.0, # MA_Signals가 -1,0,1 이므로 1.0으로 둡니다.
                    teacher_scaling=1.0, # 목표도 -1,0,1 이므로 1.0으로 둡니다.
                    teacher_shift=0.0,
                    random_state=random_state, silent=True)

    esn_model.fit(X_train, y_train)

    # 3. 테스트 데이터 준비 및 예측
    test_df_copy = test_df.copy()

    # 예측에 사용할 피처 데이터 준비 (test_df의 마지막 행도 포함될 수 있도록 Target_cpm_point_type 드롭은 하지 않음)
    # ESN은 현재 X로 다음 Y를 예측하므로, test_df의 모든 X를 사용합니다.
    df_esn_test = test_df_copy.dropna(subset=features) # Target_cpm_point_type은 예측에 직접 사용되지 않음

    # 테스트 데이터가 충분하지 않을 경우 즉시 빈 DataFrame 반환
    if df_esn_test.empty:
        return pd.DataFrame(columns=['Close', 'Predicted_Signals']), {}

    X_test_raw = df_esn_test[features].values
    X_test = scaler.transform(X_test_raw)

    test_indices = df_esn_test.index
    test_close_prices = df_esn_test['Close']

    esn_predictions = esn_model.predict(X_test, continuation=False)
    esn_predictions = esn_predictions.flatten()

    # 4. 예측값을 매수/매도 신호로 변환
    esn_signals_df = pd.DataFrame(index=test_indices)
    esn_signals_df['Prediction'] = esn_predictions
    esn_signals_df['Close'] = test_close_prices

    esn_signals_df['Type_Num'] = 0 # 기본값 HOLD (0)
    esn_signals_df.loc[esn_signals_df['Prediction'] > signal_threshold, 'Type_Num'] = 1 # SELL
    esn_signals_df.loc[esn_signals_df['Prediction'] < -signal_threshold, 'Type_Num'] = -1 # BUY

    # 백테스팅을 위해 필요한 컬럼만 추출
    backtest_signals = esn_signals_df[['Close', 'Type_Num']].rename(columns={'Type_Num': 'Predicted_Signals'})

    return backtest_signals