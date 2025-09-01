import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from backtesting import Backtest, Strategy
from ESN_Signals import esn_signals
import warnings

warnings.filterwarnings('ignore')

# GA 최적화 과정에서 사용될 전략
class PredictedSignalStrategy(Strategy):
    def init(self):
        self.signal = self.I(lambda x: x, self.data.Predicted_Signals, name='signal')

    def next(self):
        current_signal = self.signal[-1]
        if current_signal == -1 and not self.position:
            self.buy()
        elif current_signal == 1 and self.position.is_long:
            self.position.close()

# 하이퍼파라미터 범위 설정
PARAM_RANGES = {
    'n_reservoir': {'min': 100, 'max': 1000, 'type': int},
    'spectral_radius': {'min': 0.5, 'max': 1.5, 'type': float},
    'sparsity': {'min': 0.01, 'max': 0.5, 'type': float},
    'signal_threshold': {'min': 0.1, 'max': 0.9, 'type': float}
}
# n_reservoir 고정값
# N_RESERVOIR_FIXED = 1000

try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except RuntimeError:
    pass

def generate_individual():
    n_res = random.randint(PARAM_RANGES['n_reservoir']['min'], PARAM_RANGES['n_reservoir']['max'])
    spec_rad = random.uniform(PARAM_RANGES['spectral_radius']['min'], PARAM_RANGES['spectral_radius']['max'])
    sp = random.uniform(PARAM_RANGES['sparsity']['min'], PARAM_RANGES['sparsity']['max'])
    sig_thresh = random.uniform(PARAM_RANGES['signal_threshold']['min'], PARAM_RANGES['signal_threshold']['max'])
    return [n_res, spec_rad, sp, sig_thresh]

def fitness_function_with_backtesting(params, train_df: pd.DataFrame, test_df: pd.DataFrame, Technical_Signals=None):
    n_reservoir, spectral_radius, sparsity, signal_threshold = params
    
    # n_reservoir = N_RESERVOIR_FIXED
    n_reservoir = int(round(n_reservoir))
    n_reservoir = max(PARAM_RANGES['n_reservoir']['min'], min(n_reservoir, PARAM_RANGES['n_reservoir']['max']))
    spectral_radius = max(PARAM_RANGES['spectral_radius']['min'], min(spectral_radius, PARAM_RANGES['spectral_radius']['max']))
    sparsity = max(PARAM_RANGES['sparsity']['min'], min(sparsity, PARAM_RANGES['sparsity']['max']))
    signal_threshold = max(PARAM_RANGES['signal_threshold']['min'], min(signal_threshold, PARAM_RANGES['signal_threshold']['max']))

    try:
        backtest_signals_df = esn_signals(
            train_df=train_df,
            test_df=test_df,
            Technical_Signals=Technical_Signals,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            signal_threshold=signal_threshold
        )
        if backtest_signals_df.empty or 'Predicted_Signals' not in backtest_signals_df.columns:
            return 0.0,
        
        backtest_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        backtest_data['Predicted_Signals'] = backtest_signals_df['Predicted_Signals'].reindex(backtest_data.index)
        backtest_data['Predicted_Signals'] = backtest_data['Predicted_Signals'].fillna(0)
        
        bt = Backtest(backtest_data, PredictedSignalStrategy,
                      cash=10000, commission=.002, exclusive_orders=True)
        stats = bt.run()
        return_percent = stats['Return [%]']

        if pd.isna(return_percent) or np.isinf(return_percent):
            return 0.0,
        return return_percent,
    except Exception as e:
        print(f"백테스팅 중 오류 발생: {e}")
        return 0.0,

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function_with_backtesting)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian,
                 mu=[0, 0, 0, 0],
                 sigma=[(PARAM_RANGES['n_reservoir']['max'] - PARAM_RANGES['n_reservoir']['min']) * 0.1,
                        (PARAM_RANGES['spectral_radius']['max'] - PARAM_RANGES['spectral_radius']['min']) * 0.1,
                        (PARAM_RANGES['sparsity']['max'] - PARAM_RANGES['sparsity']['min']) * 0.1,
                        (PARAM_RANGES['signal_threshold']['max'] - PARAM_RANGES['signal_threshold']['min']) * 0.1],
                 indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_genetic_algorithm(train_df_ga: pd.DataFrame, test_df_ga: pd.DataFrame, technical_signals_list: list,
                          pop_size: int = 50, num_generations: int = 20, cxpb: float = 0.7, mutpb: float = 0.2,
                          random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    toolbox.evaluate.keywords['train_df'] = train_df_ga
    toolbox.evaluate.keywords['test_df'] = test_df_ga
    toolbox.evaluate.keywords['Technical_Signals'] = technical_signals_list

    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    population, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, num_generations,
                                          stats=stats, halloffame=hof, verbose=True)

    best_individual = hof[0]
    print(f"\nGA 최적화 완료 - 최적 하이퍼파라미터: {best_individual}")
    print(f"GA 최적화 완료 - 최고 Return [%]: {best_individual.fitness.values[0]:.4f}")

    return best_individual, log

def perform_final_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame, best_params: list, technical_signals_list: list,
                           random_state: int = 42):
    n_reservoir, spectral_radius, sparsity, signal_threshold = best_params

    # n_reservoir = N_RESERVOIR_FIXED
    n_reservoir = int(round(n_reservoir))
    n_reservoir = max(PARAM_RANGES['n_reservoir']['min'], min(n_reservoir, PARAM_RANGES['n_reservoir']['max']))
    spectral_radius = max(PARAM_RANGES['spectral_radius']['min'], min(spectral_radius, PARAM_RANGES['spectral_radius']['max']))
    sparsity = max(PARAM_RANGES['sparsity']['min'], min(sparsity, PARAM_RANGES['sparsity']['max']))
    signal_threshold = max(PARAM_RANGES['signal_threshold']['min'], min(signal_threshold, PARAM_RANGES['signal_threshold']['max']))

    print(f"\n--- 최적화된 파라미터로 최종 ESN 학습 및 백테스팅 ---")
    print(f"  n_reservoir: {n_reservoir}")
    print(f"  spectral_radius: {spectral_radius:.4f}")
    print(f"  sparsity: {sparsity:.4f}")
    print(f"  signal_threshold: {signal_threshold:.4f}")

    final_backtest_signals_df = esn_signals(
        train_df=train_df,
        test_df=test_df,
        Technical_Signals=technical_signals_list,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        signal_threshold=signal_threshold,
        random_state=random_state
    )

    if not isinstance(final_backtest_signals_df, pd.DataFrame) or final_backtest_signals_df.empty or 'Predicted_Signals' not in final_backtest_signals_df.columns:
        print("최종 ESN 모델에서 유효한 신호가 생성되지 않았습니다. 백테스팅을 건너뜀.")
        return None, None

    final_backtest_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    final_backtest_data['Predicted_Signals'] = final_backtest_signals_df['Predicted_Signals']
    final_backtest_data['Predicted_Signals'] = final_backtest_data['Predicted_Signals'].fillna(0)

    bt_final = Backtest(final_backtest_data, PredictedSignalStrategy,
                        cash=10000, commission=.002, exclusive_orders=True)
    stats_final = bt_final.run()

    print("\n최종 백테스팅 결과 (최적화된 파라미터):")
    print(stats_final)
    
    bt_final.plot(filename='test_df_backtest_results', open_browser=True)
    
    return stats_final, final_backtest_signals_df

def rolling_forward_split(df: pd.DataFrame, n_splits: int, initial_train_ratio: float = 0.5):
    total_len = len(df)
    initial_train_size = int(total_len * initial_train_ratio)
    remaining_len = total_len - initial_train_size
    
    if n_splits == 0:
        val_size = 0
    else:
        val_size = remaining_len // n_splits
    
    for i in range(n_splits):
        train_end_idx = initial_train_size + i * val_size
        val_end_idx = train_end_idx + val_size

        train_df = df.iloc[:train_end_idx].copy()
        val_df = df.iloc[train_end_idx:val_end_idx].copy()

        if val_df.empty:
            continue
        
        yield train_df, val_df

def esn_rolling_forward(df: pd.DataFrame, technical_signals_list: list, n_splits: int = 5,
                                    pop_size: int = 50, num_generations: int = 20):
    total_returns = []
    bh_returns = []
    
    splits = list(rolling_forward_split(df, n_splits))
    if not splits:
        print("유효한 데이터 분할이 생성되지 않았습니다.")
        return None, None

    print("--- 1. 첫 번째 폴드로 ESN 파라미터 최적화 ---")
    first_train_df, first_val_df = splits[0]
    best_params, _ = run_genetic_algorithm(
        train_df_ga=first_train_df,
        test_df_ga=first_val_df,
        technical_signals_list=technical_signals_list,
        pop_size=pop_size,
        num_generations=num_generations
    )
    
    print("\n--- 2. 최적 파라미터로 모든 폴드에 대해 롤링 포워드 검증 ---")
    for i, (train_df, val_df) in enumerate(splits):
        print(f"\n--- 폴드 {i+1} / {n_splits} ---")
        
        try:
            stats, _ = perform_final_backtest(
                train_df=train_df,
                test_df=val_df,
                best_params=best_params,
                technical_signals_list=technical_signals_list
            )
            if stats is not None:
                total_returns.append(stats['Return [%]'])
                bh_returns.append(stats['Buy & Hold Return [%]'])
        except Exception as e:
            print(f"폴드 {i+1} 백테스팅 중 오류 발생: {e}")
            
    print("\n" + "="*50)
    print("롤링 포워드 교차 검증 최종 결과:")
    print(f"총 {len(total_returns)}개 폴드 결과")
    print(f"각 폴드 Return [%]: {total_returns}")
    print(f"평균 Return [%]: {np.mean(total_returns):.4f}")
    print(f"Buy&Hold 평균 Return [%]: {np.mean(bh_returns):.4f}")
    print(f"표준편차: {np.std(total_returns):.4f}")
    print("="*50)
    
    return best_params, total_returns