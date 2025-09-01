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

def generate_roc_signals(data: pd.DataFrame, period: int, threshold: float) -> pd.DataFrame:
    """
    ROC(Rate of Change) 지표를 기반으로 매수/매도 신호를 생성합니다.
    Args:
        data (pd.DataFrame): 시계열 데이터 (종가 'Close' 포함).
        period (int): ROC 계산 기간.
        threshold (float): 신호 발생을 위한 기준선.
    Returns:
        pd.DataFrame: 생성된 매매 신호 DataFrame.
    """
    df_copy = data.copy()
    
    # ROC 계산
    df_copy['ROC'] = talib.ROC(df_copy['Close'], timeperiod=period)
    
    df_filtered = df_copy.dropna().copy()
    
    signals = []
    
    # 신호 생성 로직
    # 매수 신호: ROC가 threshold를 상향 돌파할 때
    buy_signals = df_filtered[(df_filtered['ROC'].shift(1) <= threshold) & 
                              (df_filtered['ROC'] > threshold)]
    for index, row in buy_signals.iterrows():
        signals.append({'Index': index, 'Close': row['Close'], 'Type': 'BUY'})

    # 매도 신호: ROC가 -threshold를 하향 돌파할 때
    sell_signals = df_filtered[(df_filtered['ROC'].shift(1) >= -threshold) & 
                               (df_filtered['ROC'] < -threshold)]
    for index, row in sell_signals.iterrows():
        signals.append({'Index': index, 'Close': row['Close'], 'Type': 'SELL'})

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df = signals_df.sort_values(by='Index').reset_index(drop=True)
        
    return signals_df

def evaluate_roc_individual(individual, df_data, expected_trading_points_df):
    """유전 알고리즘의 각 개체를 평가하는 함수."""
    period, threshold = individual
    period = int(max(2, period))
    threshold = max(0, threshold)

    if period < 5:
        return (float('inf'),)

    suggested_signals_df = generate_roc_signals(df_data, period, threshold)
    
    fitness = calculate_total_fitness_optimized(df_data, expected_trading_points_df, suggested_signals_df)

    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)

def run_roc_ga_optimization(df_input: pd.DataFrame, generations: int = 50, population_size: int = 50, seed: int = None):
    """
    유전 알고리즘을 실행하여 ROC 파라미터를 최적화합니다.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df_data = df_input.copy()

    signal_rows = df_data.loc[df_data['cpm_point_type'] != 0].copy()
    signal_rows['Type'] = signal_rows['cpm_point_type'].map({-1: 'BUY', 1: 'SELL'})
    expected_trading_points_df = pd.DataFrame({
        'Index': signal_rows.index,
        'Type': signal_rows['Type'],
        'Close': signal_rows['Close']
    })

    toolbox = base.Toolbox()
    toolbox.register("attr_period", random.randint, 5, 50)  # 기간 (5 ~ 50)
    toolbox.register("attr_threshold", random.uniform, 0, 10) # 기준선 (0 ~ 10)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_period, toolbox.attr_threshold), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_roc_individual, df_data=df_data, expected_trading_points_df=expected_trading_points_df)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=[0, 0], sigma=[5, 1], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("ROC 유전 알고리즘 실행 중...")
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- ROC 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (period, threshold): {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_period, final_threshold = best_individual
    
    suggested_signals_from_best_params = generate_roc_signals(df_data, int(final_period), final_threshold)
    
    df_data['ROC_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['ROC_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return best_individual, best_fitness, df_data