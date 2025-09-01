import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal1 import calculate_total_fitness_optimized

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_MA_signals(data: pd.DataFrame, N: int, n: int, a: float, b: float, c: float) -> pd.DataFrame:
    df_copy = data.copy()

    close_values = df_copy['Close'].values

    ma_N_values = talib.SMA(close_values, timeperiod=N)
    ma_n_values = talib.SMA(close_values, timeperiod=n)

    df_copy['MA_N'] = pd.Series(ma_N_values, index=df_copy.index)
    df_copy['MA_n'] = pd.Series(ma_n_values, index=df_copy.index)
    
    df_copy['Zt'] = df_copy['MA_N'] - df_copy['MA_n']
    
    df_filtered = df_copy.dropna(subset=['MA_N', 'MA_n', 'Zt']).copy()
    
    signals = []
    
    last_golden_cross_idx = None
    last_dead_cross_idx = None

    filtered_indices = df_filtered.index.tolist()
    
    for i in range(1, len(filtered_indices)):
        current_idx = filtered_indices[i]
        prev_idx = filtered_indices[i-1]
        
        current_zt = df_filtered.loc[current_idx, 'Zt']
        prev_zt = df_filtered.loc[prev_idx, 'Zt']

        # (a) 매수 신호 조건: 골든 크로스 발생 및 조건 만족
        if prev_zt <= 0 and current_zt > 0: # 골든 크로스 발생
            last_golden_cross_idx = current_idx
        
        if last_golden_cross_idx is not None and current_zt >= 0:
            t1_internal_idx_in_filtered = df_filtered.index.get_loc(last_golden_cross_idx)
            current_internal_idx_in_filtered = df_filtered.index.get_loc(current_idx)
            
            Mzt = df_filtered['Zt'].iloc[t1_internal_idx_in_filtered : current_internal_idx_in_filtered + 1].max()

            if abs(Mzt) > b * c and current_zt < min(abs(Mzt) / a, c):
                signals.append({'Index': current_idx, 
                                'Close': df_filtered.loc[current_idx, 'Close'], 
                                'Type': 'BUY'})

        # (b) 매도 신호 조건: 데드 크로스 발생 및 조건 만족
        if prev_zt >= 0 and current_zt < 0: # 데드 크로스 발생
            last_dead_cross_idx = current_idx
        
        if last_dead_cross_idx is not None and current_zt < 0:
            k1_internal_idx_in_filtered = df_filtered.index.get_loc(last_dead_cross_idx)
            current_internal_idx_in_filtered = df_filtered.index.get_loc(current_idx)
            
            Mwk = df_filtered['Zt'].iloc[k1_internal_idx_in_filtered : current_internal_idx_in_filtered + 1].apply(abs).max()
            current_wk = abs(current_zt) # 현재 시점의 wk

            if abs(Mwk) > b * c and current_wk < min(abs(Mwk) / a, c):
                signals.append({'Index': current_idx, 
                                'Close': df_filtered.loc[current_idx, 'Close'], 
                                'Type': 'SELL'})

    signals_df = pd.DataFrame(signals)

    if not signals_df.empty:
        signals_df = signals_df.sort_values(by='Index').reset_index(drop=True)
    else:
        print("빈 데이터프레임 반환")
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])
    
    return signals_df

def evaluate_MA_individual(individual, df_data, expected_trading_points_df):
    N, n, a, b, c = individual
    N = int(max(1, N))
    n = int(max(1, n))
    a = max(0.01, a)
    b = max(0.0001, b)
    c = max(0.01, c)

    if N <= n or N < 5 or n < 2:
        return (float('inf'),)

    suggested_signals_df = generate_MA_signals(df_data, N, n, a, b, c)
    fitness = calculate_total_fitness_optimized(df_data, expected_trading_points_df, suggested_signals_df)

    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)
    
    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)


def run_MA_ga_optimization(df_input: pd.DataFrame, generations: int = 50, population_size: int = 50, seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df_data = df_input.copy()
    if isinstance(df_input, pd.Series):
        df_data = pd.DataFrame(df_input)
        df_data.columns = ['Close']
    elif 'Close' not in df_data.columns:
        raise ValueError("입력 DataFrame에 'Close' 컬럼이 반드시 포함되어야 합니다.")

    if 'cpm_point_type' not in df_data.columns:
        raise ValueError("입력 DataFrame에 'cpm_point_type' 컬럼이 반드시 포함되어야 합니다.")

    signal_rows = df_data.loc[df_data['cpm_point_type'] != 0].copy()
    signal_rows['Type'] = signal_rows['cpm_point_type'].map({-1: 'BUY', 1: 'SELL'})

    expected_trading_points_df = pd.DataFrame({
        'Index': signal_rows.index,
        'Type': signal_rows['Type'],
        'Close': signal_rows['Close']
    })

    toolbox = base.Toolbox()

    # 파라미터 등록: N (장기 MA 기간), n (단기 MA 기간), a, b, c (논문 파라미터)
    toolbox.register("attr_N", random.randint, 41, 150) # 장기 MA 기간
    toolbox.register("attr_n", random.randint, 5, 40)   # 단기 MA 기간
    toolbox.register("attr_a", random.uniform, 0.1, 10.0)
    toolbox.register("attr_b", random.uniform, 0.01, 1.0)
    toolbox.register("attr_c", random.uniform, 0.1, 5.0)

    # 개체는 5개의 파라미터 (N, n, a, b, c)로 구성
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_N, toolbox.attr_n, 
                      toolbox.attr_a, toolbox.attr_b, toolbox.attr_c), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 적합도 평가 함수 등록
    toolbox.register("evaluate", evaluate_MA_individual, df_data=df_data, expected_trading_points_df=expected_trading_points_df)

    # 유전 연산자 등록
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, 
                     mu=[0]*5, sigma=[5, 2, 0.5, 0.05, 0.1], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("이동평균 유전 알고리즘 실행 중...")
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

    print("\n--- 이동평균 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (N, n, a, b, c): {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_N, final_n, final_a, final_b, final_c = best_individual
    final_N = int(final_N)
    final_n = int(final_n)
    
    suggested_signals_from_best_params = generate_MA_signals(
        df_data, final_N, final_n, final_a, final_b, final_c
    )

    df_data['MA_Signals'] = 0

    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['MA_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    else:
        df_data['MA_Signals'] = 0
    
    return best_individual, best_fitness, df_data