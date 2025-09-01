import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from backtesting import Backtest, Strategy
from ESN_Signals import esn_signals
import warnings

# 경고 메시지 무시 설정 (DEAP 및 Backtesting.py에서 발생할 수 있는 경고 방지)
warnings.filterwarnings('ignore')

# PredictedSignalStrategy 클래스 정의
# GA 최적화 과정에서 사용될 전략
class PredictedSignalStrategy(Strategy):
    def init(self):
        # 'Predicted_Signals'는 ESN 모델에서 생성된 신호가 담긴 Series
        self.signal = self.I(lambda x: x, self.data.Predicted_Signals, name='signal')

    def next(self):
        current_signal = self.signal[-1] # 현재 시점의 신호

        # 매수 신호(-1)가 있고, 현재 포지션이 없는 경우 매수
        if current_signal == -1 and not self.position:
            self.buy()

        # 매도 신호(1)가 있고, 현재 롱 포지션이 있는 경우 포지션 청산
        elif current_signal == 1 and self.position.is_long:
            self.position.close()

# 하이퍼파라미터 범위 설정 (유전 알고리즘 초기화 및 변이 시 사용)
PARAM_RANGES = {
    'n_reservoir': {'min': 100, 'max': 1000, 'type': int},
    'spectral_radius': {'min': 0.5, 'max': 1.5, 'type': float},
    'sparsity': {'min': 0.01, 'max': 0.5, 'type': float},
    'signal_threshold': {'min': 0.1, 'max': 0.9, 'type': float}
}

# DEAP를 위한 타입 생성
# 이 부분은 파일 로드 시 한 번만 실행되어야 합니다.
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except RuntimeError: # 이미 생성되어 있을 경우 에러 방지 (주피터에서 셀 재실행 시 발생 가능)
    pass


# 유전자(하이퍼파라미터 조합) 초기화 함수
def generate_individual():
    """
    PARAM_RANGES에 정의된 범위 내에서 무작위 하이퍼파라미터 조합을 생성합니다.
    """
    n_res = random.randint(PARAM_RANGES['n_reservoir']['min'], PARAM_RANGES['n_reservoir']['max'])
    spec_rad = random.uniform(PARAM_RANGES['spectral_radius']['min'], PARAM_RANGES['spectral_radius']['max'])
    sp = random.uniform(PARAM_RANGES['sparsity']['min'], PARAM_RANGES['sparsity']['max'])
    sig_thresh = random.uniform(PARAM_RANGES['signal_threshold']['min'], PARAM_RANGES['signal_threshold']['max'])
    return [n_res, spec_rad, sp, sig_thresh]


# 유전 알고리즘을 위한 적합도 함수
def fitness_function_with_backtesting(params, train_df: pd.DataFrame, test_df: pd.DataFrame, Technical_Signals=None):
    """
    ESN 하이퍼파라미터로 ESN 모델을 학습하고 백테스팅을 수행하여 Return [%]를 반환하는 적합도 함수.

    Args:
        params (tuple): 최적화할 하이퍼파라미터 튜플 (n_reservoir, spectral_radius, sparsity, signal_threshold).
        train_df (pd.DataFrame): ESN 학습에 사용할 훈련 데이터.
        test_df (pd.DataFrame): 백테스팅에 사용할 테스트 데이터.
        Technical_Signals (list): 기술적 신호 컬럼 이름 리스트.

    Returns:
        tuple: 백테스팅 결과의 Return [%] (DEAP 요구사항에 따라 튜플로 반환).
               오류 발생 시 (0.0,) 또는 (-np.inf,) 반환.
    """
    n_reservoir, spectral_radius, sparsity, signal_threshold = params

    # 하이퍼파라미터 타입 변환 및 범위 강제 (GA가 생성한 값 보정)
    n_reservoir = int(round(n_reservoir))
    n_reservoir = max(PARAM_RANGES['n_reservoir']['min'], min(n_reservoir, PARAM_RANGES['n_reservoir']['max']))
    spectral_radius = max(PARAM_RANGES['spectral_radius']['min'], min(spectral_radius, PARAM_RANGES['spectral_radius']['max']))
    sparsity = max(PARAM_RANGES['sparsity']['min'], min(sparsity, PARAM_RANGES['sparsity']['max']))
    signal_threshold = max(PARAM_RANGES['signal_threshold']['min'], min(signal_threshold, PARAM_RANGES['signal_threshold']['max']))

    # ESN 신호 생성
    backtest_signals_df = esn_signals(
        train_df=train_df,
        test_df=test_df,
        Technical_Signals=Technical_Signals,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        signal_threshold=signal_threshold
    )

    # ESN 신호 생성이 실패했을 경우
    if backtest_signals_df.empty or 'Predicted_Signals' not in backtest_signals_df.columns:
        print("ESN 신호 생성 실패")
        return 0.0, # 낮은 적합도 반환

    # backtesting.py를 위한 데이터 준비
    # test_df에서 필요한 OHLCV 컬럼을 복사하고 ESN 신호를 병합
    backtest_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # ESN 신호(Predicted_Signals)를 backtest_data에 추가
    # esn_signals 함수가 반환하는 backtest_signals_df는 test_df와 동일한 인덱스를 가짐
    backtest_data['Predicted_Signals'] = backtest_signals_df['Predicted_Signals'].reindex(backtest_data.index)

    # 신호가 없는 날짜는 0 (HOLD)으로 채움
    backtest_data['Predicted_Signals'] = backtest_data['Predicted_Signals'].fillna(0)

    # Backtest 실행
    try:
        bt = Backtest(backtest_data, PredictedSignalStrategy,
                      cash=10000, commission=.002, exclusive_orders=True)
        stats = bt.run()

        return_percent = stats['Return [%]'] # 키 값 수정: 'Return [%]'

        # NaN 또는 무한대 값 처리
        if pd.isna(return_percent) or np.isinf(return_percent):
            print("NaN 또는 무한대 값 처리")
            return 0.0, # 낮은 적합도 반환
        return return_percent, # DEAP는 튜플을 기대합니다.
    except Exception as e:
        # 백테스팅 중 오류 발생 시 디버깅을 위해 출력할 수 있으나, GA 실행 시 너무 많을 수 있음
        print(f"백테스팅 중 오류 발생: {e}. 파라미터: {params}")
        return 0.0, # 오류 발생 시 낮은 적합도 반환

# DEAP Toolbox 설정
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 평가 함수 등록
toolbox.register("evaluate", fitness_function_with_backtesting)

# 유전 알고리즘 연산자 등록
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian,
                 mu=[0, 0, 0, 0],
                 sigma=[(PARAM_RANGES['n_reservoir']['max'] - PARAM_RANGES['n_reservoir']['min']) * 0.1,
                        (PARAM_RANGES['spectral_radius']['max'] - PARAM_RANGES['spectral_radius']['min']) * 0.1,
                        (PARAM_RANGES['sparsity']['max'] - PARAM_RANGES['sparsity']['min']) * 0.1,
                        (PARAM_RANGES['signal_threshold']['max'] - PARAM_RANGES['signal_threshold']['min']) * 0.1],
                 indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 유전 알고리즘 실행을 위한 메인 함수
def run_genetic_algorithm(train_df_ga: pd.DataFrame, test_df_ga: pd.DataFrame, technical_signals_list: list,
                          pop_size: int = 50, num_generations: int = 20, cxpb: float = 0.7, mutpb: float = 0.2,
                          random_seed: int = 42):
    """
    유전 알고리즘을 실행하여 ESN 하이퍼파라미터를 최적화합니다.

    Args:
        train_df_ga (pd.DataFrame): ESN 학습에 사용할 훈련 데이터.
        test_df_ga (pd.DataFrame): 백테스팅에 사용할 테스트 데이터.
        technical_signals_list (list): ESN 입력으로 사용할 기술적 신호 컬럼 이름 리스트.
        pop_size (int): 집단 크기.
        num_generations (int): 세대 수.
        cxpb (float): 교차 확률.
        mutpb (float): 변이 확률.
        random_seed (int): 난수 시드 (재현성을 위해).

    Returns:
        tuple: 최적화된 하이퍼파라미터 조합 (최고의 적합도를 가진 개체).
        list: 유전 알고리즘의 통계 기록 (logbook).
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 평가 함수에 훈련 데이터, 테스트 데이터, 기술적 신호 리스트 전달
    toolbox.evaluate.keywords['train_df'] = train_df_ga
    toolbox.evaluate.keywords['test_df'] = test_df_ga
    toolbox.evaluate.keywords['Technical_Signals'] = technical_signals_list

    population = toolbox.population(n=pop_size)

    # 통계 수집 객체 설정
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

# 최종 백테스팅을 위한 헬퍼 함수
def perform_final_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame, best_params: list, technical_signals_list: list,
                           random_state: int = 42):
    """
    최적화된 하이퍼파라미터로 최종 ESN 모델을 학습하고 백테스팅을 수행합니다.

    Args:
        train_df (pd.DataFrame): ESN 학습에 사용할 훈련 데이터.
        test_df (pd.DataFrame): 백테스팅에 사용할 테스트 데이터.
        best_params (list): 최적화된 하이퍼파라미터 리스트 [n_reservoir, spectral_radius, sparsity, signal_threshold].
        technical_signals_list (list): 기술적 신호 컬럼 이름 리스트.
        random_state (int): ESN 모델의 난수 시드.

    Returns:
        pandas.Series: 최종 백테스팅 결과 통계.
    """
    n_reservoir, spectral_radius, sparsity, signal_threshold = best_params

    # 하이퍼파라미터 타입 변환 및 범위 강제
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
        return None

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