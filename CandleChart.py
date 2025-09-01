import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal1 import calculate_total_fitness_optimized

# DEAP creator 객체는 한 번만 생성되어야 합니다.
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def generate_synergy_signals(data: pd.DataFrame, a: float, d: float, e: float, f: float, g: float,
                             window: int) -> pd.DataFrame:
    """
    망치형/교수형 패턴(초기 신호)과 장악형 패턴(확인 신호)의 순차적 발생을 기반으로
    매매 신호를 생성합니다.

    Args:
        data (pd.DataFrame): 'Open', 'High', 'Low', 'Close' 컬럼을 포함한 주가 데이터.
        a, d, e, f, g (float): 캔들스틱 패턴을 정의하는 최적화 파라미터.
        window (int): 초기 신호와 확인 신호 사이의 최대 기간.

    Returns:
        pd.DataFrame: 'Index', 'Close', 'Type' 컬럼을 포함한 매매 신호 데이터프레임.
    """
    df_copy = data.copy()

    # 캔들스틱 몸통 및 꼬리 길이, 추세 계산
    df_copy['Body'] = abs(df_copy['Open'] - df_copy['Close'])
    df_copy['Wick'] = df_copy['High'] - df_copy['Low'] - df_copy['Body']
    df_copy['Trend'] = talib.SMA(df_copy['Close'], timeperiod=10)
    df_filtered = df_copy.dropna().copy()

    initial_signals = []  # 망치형/교수형 초기 신호를 저장할 리스트
    confirmed_signals = []  # 최종 확정 신호를 저장할 리스트

    # 1. 초기 신호(망치형/교수형) 포착
    for i in range(1, len(df_filtered)):
        curr_candle = df_filtered.iloc[i]
        prev_candle = df_filtered.iloc[i - 1]

        if curr_candle['Body'] > 0 and curr_candle['Wick'] / curr_candle['Body'] > a:
            is_uptrend = prev_candle['Close'] > prev_candle['Trend']
            signal_type = 'SELL' if is_uptrend else 'BUY'
            initial_signals.append({'Index': curr_candle.name, 'Type': signal_type, 'Close': curr_candle['Close']})

    # 2. 확인 신호(장악형) 포착 및 최종 신호 생성
    initial_signals_df = pd.DataFrame(initial_signals).set_index('Index')

    for i in range(1, len(df_filtered)):
        curr_candle = df_filtered.iloc[i]
        prev_candle = df_filtered.iloc[i - 1]

        is_uptrend = prev_candle['Close'] > prev_candle['Trend']

        # 장악형 패턴 조건
        is_bullish_engulfing = (prev_candle['Close'] < prev_candle['Open']) and \
                               (curr_candle['Close'] > curr_candle['Open']) and \
                               (curr_candle['Open'] < prev_candle['Close']) and \
                               (curr_candle['Close'] > prev_candle['Open'])

        is_bearish_engulfing = (prev_candle['Close'] > prev_candle['Open']) and \
                               (curr_candle['Close'] < curr_candle['Open']) and \
                               (curr_candle['Open'] > prev_candle['Close']) and \
                               (curr_candle['Close'] < prev_candle['Open'])

        if is_bullish_engulfing and not is_uptrend:
            # Bullish Engulfing 패턴 파라미터 조건 만족
            if abs(curr_candle['Close'] - prev_candle['Open']) > d and abs(
                    prev_candle['Close'] - curr_candle['Open']) > e:
                # 윈도우 내 초기 신호 확인
                start_window = df_filtered.index[max(0, i - window)]
                initial_buy_signals_in_window = initial_signals_df.loc[start_window:curr_candle.name]
                if not initial_buy_signals_in_window.loc[initial_buy_signals_in_window['Type'] == 'BUY'].empty:
                    confirmed_signals.append({'Index': curr_candle.name, 'Close': curr_candle['Close'], 'Type': 'BUY'})

        elif is_bearish_engulfing and is_uptrend:
            # Bearish Engulfing 패턴 파라미터 조건 만족
            if abs(curr_candle['Open'] - prev_candle['Close']) > f and abs(
                    curr_candle['Close'] - prev_candle['Open']) > g:
                # 윈도우 내 초기 신호 확인
                start_window = df_filtered.index[max(0, i - window)]
                initial_sell_signals_in_window = initial_signals_df.loc[start_window:curr_candle.name]
                if not initial_sell_signals_in_window.loc[initial_sell_signals_in_window['Type'] == 'SELL'].empty:
                    confirmed_signals.append({'Index': curr_candle.name, 'Close': curr_candle['Close'], 'Type': 'SELL'})

    if not confirmed_signals:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    signals_df = pd.DataFrame(confirmed_signals)
    signals_df = signals_df.sort_values(by='Index').drop_duplicates(subset=['Index'], keep='last')
    return signals_df


def evaluate_synergy_individual(individual, df_data, expected_trading_points_df):
    """
    캔들스틱 시너지 전략의 개체(파라미터 조합)를 평가하는 적합도 함수.
    """
    a, d, e, f, g, window = individual

    # 파라미터 유효성 검사 및 보정
    a, d, e, f, g = [max(0.01, p) for p in [a, d, e, f, g]]
    window = int(max(1, window))

    suggested_signals_df = generate_synergy_signals(df_data, a, d, e, f, g, window)
    fitness = calculate_total_fitness_optimized(df_data, expected_trading_points_df, suggested_signals_df)

    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)

    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)


def run_candlestick_synergy_ga_optimization(df_input: pd.DataFrame, generations: int = 50, population_size: int = 50,
                                            seed: int = None):
    """
    캔들스틱 시너지 전략 파라미터를 유전 알고리즘으로 최적화합니다.

    Args:
        df_input (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'cpm_point_type' 컬럼을 포함한 데이터.
        generations (int): 유전 알고리즘 세대 수.
        population_size (int): 개체군 크기.
        seed (int): 난수 시드.

    Returns:
        tuple: 최적의 파라미터, 최적의 적합도, 그리고 최적화된 신호가 추가된 데이터프레임.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df_data = df_input.copy()
    if not all(col in df_data.columns for col in ['Open', 'High', 'Low', 'Close', 'cpm_point_type']):
        raise ValueError("입력 DataFrame에 'Open', 'High', 'Low', 'Close', 'cpm_point_type' 컬럼이 반드시 포함되어야 합니다.")

    signal_rows = df_data.loc[df_data['cpm_point_type'] != 0].copy()
    signal_rows['Type'] = signal_rows['cpm_point_type'].map({-1: 'BUY', 1: 'SELL'})

    expected_trading_points_df = pd.DataFrame({
        'Index': signal_rows.index,
        'Type': signal_rows['Type'],
        'Close': signal_rows['Close']
    })

    toolbox = base.Toolbox()

    # 파라미터 등록: a (망치형/교수형), d, e, f, g (장악형), window (기간)
    toolbox.register("attr_a", random.uniform, 1.0, 5.0)
    toolbox.register("attr_d", random.uniform, 0.01, 5.0)
    toolbox.register("attr_e", random.uniform, 0.01, 5.0)
    toolbox.register("attr_f", random.uniform, 0.01, 5.0)
    toolbox.register("attr_g", random.uniform, 0.01, 5.0)
    toolbox.register("attr_window", random.randint, 1, 5)

    # 개체는 6개의 파라미터로 구성
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_d, toolbox.attr_e,
                      toolbox.attr_f, toolbox.attr_g, toolbox.attr_window), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 적합도 평가 함수 등록
    toolbox.register("evaluate", evaluate_synergy_individual, df_data=df_data,
                     expected_trading_points_df=expected_trading_points_df)

    # 유전 연산자 등록
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=[0] * 6, sigma=[0.5, 0.2, 0.2, 0.2, 0.2, 1], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("캔들스틱 시너지 유전 알고리즘 실행 중...")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- 캔들스틱 시너지 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (a, d, e, f, g, window): {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_a, final_d, final_e, final_f, final_g, final_window = best_individual

    suggested_signals_from_best_params = generate_synergy_signals(
        df_data, final_a, final_d, final_e, final_f, final_g, int(final_window)
    )

    df_data['Synergy_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['Synergy_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)

    return best_individual, best_fitness, df_data