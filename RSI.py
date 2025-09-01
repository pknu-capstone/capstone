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

def generate_RSI_signals(data: pd.DataFrame, x: int, overbought_level: float, oversold_level: float, p: float, q: float) -> pd.DataFrame:
    df_copy = data.copy()
    print(len(df_copy), x)
    df_copy['RSI'] = talib.RSI(df_copy['Close'].values, timeperiod=x)
    df_filtered = df_copy.dropna(subset=['RSI']).copy()

    signals = []
    
    # RSI 사선 전략을 위한 상태 변수 초기화
    # RSI가 과매수/과매도 영역에 진입한 후 사선을 그리기 시작하는 시점의 인덱스와 RSI 값
    overbought_entry_idx = None
    oversold_entry_idx = None

    filtered_indices = df_filtered.index.tolist()
    
    for i in range(1, len(filtered_indices)):
        current_idx = filtered_indices[i]
        prev_idx = filtered_indices[i-1]
        
        current_rsi = df_filtered.loc[current_idx, 'RSI']
        prev_rsi = df_filtered.loc[prev_idx, 'RSI']
        current_close = df_filtered.loc[current_idx, 'Close']

        # --- 매도 신호 로직 (과매수 + 사선 돌파) ---
        # RSI가 과매수 수준 위로 상승 (사선 그리기 시작점 포착)
        if prev_rsi <= overbought_level and current_rsi > overbought_level:
            overbought_entry_idx = current_idx

        # 과매수 영역 진입 후, RSI가 사선 'ab'를 하향 돌파하는지 확인
        if overbought_entry_idx is not None:
            # 사선 ab의 y값 = 과매수 진입 시점의 RSI + 기울기 p * (현재 시점의 날짜 인덱스 - 과매수 진입 시점의 날짜 인덱스)
            start_pos = df_filtered.index.get_loc(overbought_entry_idx)
            current_pos = df_filtered.index.get_loc(current_idx)
            
            # 사선 ab의 Y값: 시작점 RSI + 기울기 * 경과 기간
            oblique_line_ab_y = df_filtered.loc[overbought_entry_idx, 'RSI'] + p * (current_pos - start_pos)

            # 매도 조건: RSI가 과매수 영역에 있고, 현재 RSI가 사선 ab 아래로 하락
            if current_rsi > overbought_level and current_rsi < oblique_line_ab_y:
                signals.append({'Index': current_idx, 
                                'Close': current_close, 
                                'Type': 'SELL'})
                overbought_entry_idx = None # 신호 발생 후 사선 초기화

        # --- 매수 신호 로직 (과매도 + 사선 돌파) ---
        # RSI가 과매도 수준 아래로 하락 (사선 그리기 시작점 포착)
        if prev_rsi >= oversold_level and current_rsi < oversold_level:
            oversold_entry_idx = current_idx
        
        # 과매도 영역 진입 후, RSI가 사선 'a0b0'를 상향 돌파하는지 확인
        if oversold_entry_idx is not None:
            start_pos = df_filtered.index.get_loc(oversold_entry_idx)
            current_pos = df_filtered.index.get_loc(current_idx)

            # 사선 a0b0의 Y값: 시작점 RSI + 기울기 q * (현재 시점의 날짜 인덱스 - 과매도 진입 시점의 날짜 인덱스)
            oblique_line_a0b0_y = df_filtered.loc[oversold_entry_idx, 'RSI'] + q * (current_pos - start_pos)

            # 매수 조건: RSI가 과매도 영역에 있고, 현재 RSI가 사선 a0b0 위로 상승
            if current_rsi < oversold_level and current_rsi > oblique_line_a0b0_y:
                signals.append({'Index': current_idx, 
                                'Close': current_close, 
                                'Type': 'BUY'})
                oversold_entry_idx = None # 신호 발생 후 사선 초기화

    signals_df = pd.DataFrame(signals)

    if not signals_df.empty:
        signals_df = signals_df.sort_values(by='Index').reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])
    
    return signals_df

def evaluate_RSI_individual(individual, df_data, expected_trading_points_df):
    x, overbought_level, oversold_level, p, q = individual
    x = int(max(1, x)) # RSI 기간은 최소 1 이상의 정수

    # x: RSI 기간은 최소 2 이상 (talib의 RSI 최소 기간)
    # overbought_level, oversold_level: 0-100 범위, overbought > oversold
    # p: 매도 사선 기울기 (음수 값으로 하락 추세 반영)
    # q: 매수 사선 기울기 (양수 값으로 상승 추세 반영)
    if x < 2:
        print("x < 2")
        return (float('inf'),)
    elif not (0 < oversold_level < overbought_level < 100):
        print("0 < oversold_level < overbought_level < 100")
        return (float('inf'),)
    elif p >= 0:
        print("p >= 0")
        return (float('inf'),)
    elif q <= 0:
        print("q <= 0")
        return (float('inf'),)

    suggested_signals_df = generate_RSI_signals(df_data, x, overbought_level, oversold_level, p, q)
    fitness = calculate_total_fitness_optimized(df_data, expected_trading_points_df, suggested_signals_df)

    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)
    
    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)

def custom_mutate(individual, mu, sigma, indpb):
    tools.mutGaussian(individual, mu, sigma, indpb)

    # 인덱스 0: x (RSI 기간, 5~30 정수)
    individual[0] = int(max(5, min(30, individual[0])))

    # 인덱스 1: overbought_level (과매수 수준, 70.0~100.0 실수)
    individual[1] = max(70.0, min(100.0, individual[1]))

    # 인덱스 2: oversold_level (과매도 수준, 0.0~30.0 실수)
    individual[2] = max(0.0, min(30.0, individual[2]))

    # 인덱스 3: p (매도 사선 기울기, -10.0~-0.1 실수)
    individual[3] = max(-10.0, min(-0.1, individual[3]))

    # 인덱스 4: q (매수 사선 기울기, 0.1~10.0 실수)
    individual[4] = max(0.1, min(10.0, individual[4]))

    # overbought_level은 oversold_level보다 항상 커야 함
    if individual[1] <= individual[2]:
        # overbought_level을 oversold_level보다 살짝 크게 조정 (예: 0.1 이상 차이)
        if individual[2] >= 99.9: # overbought_level의 최대치에 가까우면
             individual[2] = individual[1] - 0.1 # oversold_level을 overbought_level보다 작게
             individual[2] = max(10.0, individual[2]) # oversold_level 최소값 유지
        else:
            individual[1] = individual[2] + 0.1 # overbought_level을 oversold_level보다 크게
            individual[1] = min(100.0, individual[1]) # overbought_level 최대값 유지

    return individual,

def run_RSI_ga_optimization(df_input: pd.DataFrame, generations: int = 50, population_size: int = 50, seed: int = None):
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

    toolbox.register("attr_x", random.randint, 5, 30) # RSI 기간 (일반적으로 14, 9 등)
    toolbox.register("attr_overbought", random.uniform, 70.0, 100.0) # 과매수 수준 (80 근처)
    toolbox.register("attr_oversold", random.uniform, 0.0, 30.0) # 과매도 수준 (20 근처)
    toolbox.register("attr_p", random.uniform, -10.0, -0.1) # 매도 사선 기울기 (음수여야 함)
    toolbox.register("attr_q", random.uniform, 0.1, 10.0) # 매수 사선 기울기 (양수여야 함)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_x, toolbox.attr_overbought,
                      toolbox.attr_oversold, toolbox.attr_p, toolbox.attr_q), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_RSI_individual, df_data=df_data, expected_trading_points_df=expected_trading_points_df)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    toolbox.register("mutate", custom_mutate, 
                     mu=[0]*5, sigma=[1, 1, 1, 0.2, 0.2], indpb=0.1)

    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("RSI 유전 알고리즘 실행 중...")
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

    print("\n--- RSI 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (x, overbought_level, oversold_level, p, q): {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_x, final_overbought, final_oversold, final_p, final_q = best_individual
    final_x = int(final_x)
    
    suggested_signals_from_best_params = generate_RSI_signals(
        df_data, final_x, final_overbought, final_oversold, final_p, final_q
    )

    df_data['RSI_Signals'] = 0

    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data.loc[signal_map.index, 'RSI_Signals'] = signal_map.fillna(0).astype(int)
    else:
        df_data['RSI_Signals'] = 0
    
    return best_individual, best_fitness, df_data