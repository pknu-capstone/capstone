import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from eval_signal1 import calculate_total_fitness_optimized
import multiprocessing
import talib

def calculate_stochastic(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df_copy = df.copy()

    if 'High' not in df_copy.columns:
        df_copy['High'] = df_copy['Close']
    if 'Low' not in df_copy.columns:
        df_copy['Low'] = df_copy['Close']

    df_copy['%K'], df_copy['%D'] = talib.STOCH(
        df_copy['High'], df_copy['Low'], df_copy['Close'],
        fastk_period=n,          # n은 %K를 계산하는 기간 (FastK Period)
        slowk_period=3,          # %K의 이동평균 기간 (SlowK Period)
        slowk_matype=0,          # %K의 이동평균 타입 (0 = SMA)
        slowd_period=3,          # %D (SlowD Period)
        slowd_matype=0           # %D의 이동평균 타입 (0 = SMA)
    )

    return df_copy

def generate_stochastic_signals(data: pd.DataFrame, n: int, a: float, b: float, c: float, d: float) -> pd.DataFrame:
    data_with_stochastic = calculate_stochastic(data.copy(), n)

    if len(data_with_stochastic.dropna(subset=['%K', '%D'])) < 1:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    df_filtered = data_with_stochastic.dropna(subset=['%K', '%D']).copy()

    prev_k = df_filtered['%K'].shift(1)
    prev_d = df_filtered['%D'].shift(1)
    current_k = df_filtered['%K']
    current_d = df_filtered['%D']

    buy_conditions = (prev_k <= prev_d) & \
                     (current_k > current_d) & \
                     (current_k < a) & \
                     ((current_k - current_d) < b)

    signals_df_buy = df_filtered[buy_conditions].copy()
    signals_df_buy['Type'] = 'BUY'

    sell_conditions = (prev_k >= prev_d) & \
                      (current_k < current_d) & \
                      (current_k > c) & \
                      ((current_d - current_k) < d)

    signals_df_sell = df_filtered[sell_conditions].copy()
    signals_df_sell['Type'] = 'SELL'

    signals_df = pd.concat([signals_df_buy, signals_df_sell]).sort_index()

    if not signals_df.empty:
        signals_df = signals_df[['Close', 'Type']].reset_index(names=['Index'])
    else:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    return signals_df

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate_individual(individual, df_data, expected_trading_points_df):
    n, a, b, c, d = individual
    n = int(max(1, n))

    if n < 5 or a >= c or b <= 0 or d <= 0 or a < 0 or b < 0 or c < 0 or d < 0:
         return (float('inf'),)

    suggested_signals_df = generate_stochastic_signals(df_data, n, a, b, c, d)
    
    fitness = calculate_total_fitness_optimized(df_data, expected_trading_points_df, suggested_signals_df)

    if expected_trading_points_df.empty:
        return (1000000000.0,)

    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)

def run_stochastic_ga_optimization(df_input: pd.DataFrame, generations: int = 50, population_size: int = 50, seed: int = None):
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

    toolbox.register("attr_n", random.randint, 5, 20)
    toolbox.register("attr_a", random.uniform, 10.0, 40.0)
    toolbox.register("attr_b", random.uniform, 0.1, 5.0)
    toolbox.register("attr_c", random.uniform, 60.0, 90.0)
    toolbox.register("attr_d", random.uniform, 0.1, 5.0)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n, toolbox.attr_a, toolbox.attr_b, toolbox.attr_c, toolbox.attr_d), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, df_data=df_data, expected_trading_points_df=expected_trading_points_df)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=[0, 0, 0, 0, 0], sigma=[2, 5, 0.5, 5, 0.5], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("유전 알고리즘 실행 중...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                         stats=stats, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (n, a, b, c, d): {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_n, final_a, final_b, final_c, final_d = best_individual
    final_n = int(final_n)
    suggested_signals_from_best_params = generate_stochastic_signals(df_data, final_n, final_a, final_b, final_c, final_d)

    df_data['ST_Signals'] = 0

    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['ST_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    else:
        df_data['ST_Signals'] = 0
    
    return best_individual, best_fitness, df_data