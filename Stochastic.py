import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import multiprocessing
from eval_signal1 import calculate_total_fitness_optimized

# DEAP creator 객체는 한 번만 생성되어야 합니다.
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def generate_stochastic_signals(data: pd.DataFrame, a: float, b: float, c: float, d: float,
                                period: int = 14) -> pd.DataFrame:
    df_copy = data.copy()

    # 1. HH, LL 계산 (지정된 기간 동안의 최고가와 최저가)
    df_copy['HH'] = df_copy['High'].rolling(window=period).max()
    df_copy['LL'] = df_copy['Low'].rolling(window=period).min()

    df_filtered = df_copy.dropna().copy()

    # 2. H3, L3 계산 (논문 공식: 3기간 동안의 합)
    # H3: 3기간 (C - LL)의 합
    df_filtered['C-LL'] = df_filtered['Close'] - df_filtered['LL']
    df_filtered['H3'] = df_filtered['C-LL'].rolling(window=3).sum()

    # L3: 3기간 (HH - LL)의 합
    df_filtered['HH-LL'] = df_filtered['HH'] - df_filtered['LL']
    df_filtered['L3'] = df_filtered['HH-LL'].rolling(window=3).sum()

    # 3. %K 계산 (논문 공식)
    df_filtered['%K'] = 100 * (df_filtered['H3'] / df_filtered['L3']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. H'3, L'3 계산 (논문 공식: 3기간 동안의 H3, L3 합)
    df_filtered['H\'3'] = df_filtered['H3'].rolling(window=3).sum()
    df_filtered['L\'3'] = df_filtered['L3'].rolling(window=3).sum()

    # 5. %D 계산 (논문 공식)
    df_filtered['%D'] = 100 * (df_filtered['H\'3'] / df_filtered['L\'3']).replace([np.inf, -np.inf], np.nan).fillna(0)

    df_final = df_filtered.dropna().copy()
    signals = []

    # 6. 매매 신호 생성
    for i in range(1, len(df_final)):
        prev_k = df_final['%K'].iloc[i - 1]
        prev_d = df_final['%D'].iloc[i - 1]
        curr_k = df_final['%K'].iloc[i]
        curr_d = df_final['%D'].iloc[i]

        # 매수 신호 조건
        if prev_k <= prev_d and curr_k > curr_d:  # %K가 %D를 상향 돌파
            if curr_k < a and curr_k - curr_d < b:
                signals.append({'Index': df_final.index[i], 'Close': df_final['Close'].iloc[i], 'Type': 'BUY'})

        # 매도 신호 조건
        elif prev_k >= prev_d and curr_k < curr_d:  # %K가 %D를 하향 돌파
            if curr_k > c and curr_d - curr_k < d:
                signals.append({'Index': df_final.index[i], 'Close': df_final['Close'].iloc[i], 'Type': 'SELL'})

    if not signals:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    signals_df = pd.DataFrame(signals)
    return signals_df.sort_values(by='Index').reset_index(drop=True)


def evaluate_stochastic_individual(individual, df_data, expected_trading_points_df):
    a, b, c, d = individual

    # 파라미터 유효성 검사 및 보정
    a = min(max(50, a), 100)  # a는 50~100 사이의 overbought 영역 값
    b = max(0.01, b)  # b는 양수
    c = min(max(0, c), 50)  # c는 0~50 사이의 oversold 영역 값
    d = max(0.01, d)  # d는 양수

    suggested_signals_df = generate_stochastic_signals(df_data, a, b, c, d)
    fitness = calculate_total_fitness_optimized(df_data, expected_trading_points_df, suggested_signals_df)

    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)

    if fitness == float('inf'):
        return (1000000000.0,)  # 매우 큰 페널티 값

    return (fitness,)


def run_stochastic_ga_optimization(df_input: pd.DataFrame, generations: int = 50, population_size: int = 50,
                                   seed: int = None):
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

    # 파라미터 등록: a, b, c, d
    toolbox.register("attr_a", random.uniform, 50, 90)  # 매수 신호 상한선 (%K < a)
    toolbox.register("attr_b", random.uniform, 0, 10)  # 매수 신호 조건 (%K - %D < b)
    toolbox.register("attr_c", random.uniform, 10, 50)  # 매도 신호 하한선 (%K > c)
    toolbox.register("attr_d", random.uniform, 0, 10)  # 매도 신호 조건 (%D - %K < d)

    # 개체는 4개의 파라미터로 구성
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_b, toolbox.attr_c, toolbox.attr_d), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 적합도 평가 함수 등록
    toolbox.register("evaluate", evaluate_stochastic_individual, df_data=df_data,
                     expected_trading_points_df=expected_trading_points_df)

    # 유전 연산자 등록
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=[0] * 4, sigma=[5, 0.5, 5, 0.5], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("스토캐스틱 유전 알고리즘 실행 중...")
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

    print("\n--- 스토캐스틱 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (a, b, c, d): {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_a, final_b, final_c, final_d = best_individual

    suggested_signals_from_best_params = generate_stochastic_signals(
        df_data, final_a, final_b, final_c, final_d
    )

    df_data['Stochastic_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['Stochastic_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)

    return best_individual, best_fitness, df_data