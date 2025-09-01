import pandas as pd
import numpy as np

def calculate_total_fitness_optimized(
    data: pd.DataFrame,
    expected_trading_points: pd.DataFrame,
    suggested_signals: pd.DataFrame
) -> float:
    total_fitness = 0.0

    expected_trading_points_sorted = expected_trading_points.sort_values(by='Index')
    suggested_signals_sorted = suggested_signals.sort_values(by='Index')

    suggested_signal_indices_arr = suggested_signals_sorted['Index'].values
    data_indices = data.index.values

    for _, row in expected_trading_points_sorted.iterrows():
        ti_original_index = row['Index']
        ti_type = row['Type']

        try:
            ti_data_iloc = data.index.get_loc(ti_original_index)
        except KeyError:
            continue

        ti_close = data['Close'].iloc[ti_data_iloc]

        fitness_val = float('inf')

        search_start_data_idx = max(0, ti_data_iloc - 1)
        search_end_data_idx = min(len(data) - 1, ti_data_iloc + 1)
        
        idx_in_suggested_signals_arr_start = np.searchsorted(
            suggested_signal_indices_arr, 
            data_indices[search_start_data_idx], 
            side='left'
        )
        idx_in_suggested_signals_arr_end = np.searchsorted(
            suggested_signal_indices_arr, 
            data_indices[search_end_data_idx], 
            side='right'
        )
        
        relevant_sjs = suggested_signals_sorted.iloc[
            idx_in_suggested_signals_arr_start:idx_in_suggested_signals_arr_end
        ].copy()

        if not relevant_sjs.empty:
            closest_sj_idx_in_relevant = (relevant_sjs['Index'] - ti_original_index).abs().idxmin()
            Sj = relevant_sjs.loc[closest_sj_idx_in_relevant]
            Sj_close = Sj['Close']
            Sj_type = Sj['Type']

            if ti_type == 'BUY':
                if Sj_type == 'BUY':
                    fitness_val = Sj_close - ti_close
                elif Sj_type == 'SELL':
                    if abs(Sj_close - ti_close) / ti_close < 0.05:
                        max_price_in_range = data['Close'].iloc[
                            max(0, ti_data_iloc-1) : min(len(data), ti_data_iloc+2)
                        ].max()
                        fitness_val = 2 * (max_price_in_range - ti_close)

            elif ti_type == 'SELL':
                if Sj_type == 'SELL':
                    fitness_val = ti_close - Sj_close
                elif Sj_type == 'BUY':
                    if abs(Sj_close - ti_close) / ti_close < 0.05:
                        min_price_in_range = data['Close'].iloc[
                            max(0, ti_data_iloc-1) : min(len(data), ti_data_iloc+2)
                        ].min()
                        fitness_val = 2 * (ti_close - min_price_in_range)
        
        if fitness_val == float('inf'):
            start_for_extremum = max(0, ti_data_iloc - 1)
            end_for_extremum = min(len(data), ti_data_iloc + 2)
            
            price_range_slice = data['Close'].iloc[start_for_extremum : end_for_extremum]
            
            if not price_range_slice.empty:
                if ti_type == 'BUY':
                    fitness_val = price_range_slice.max() - ti_close
                elif ti_type == 'SELL':
                    fitness_val = ti_close - price_range_slice.min()
            else:
                fitness_val = float('inf')
                print("적합도 함수에서 inf")
        total_fitness += fitness_val
        
    # MDD 계산 및 감점 로직
    if suggested_signals_sorted.empty:
        # 시그널 없으면 MDD 계산 불가 → 패널티
        return total_fitness - 5.0
    
    # 1) 전략 포지션 데이터
    data = data.copy()
    data['position'] = 0
    
    # type = buy -> -1, sell = 1
    type_to_pos = {'BUY': -1, 'SELL': 1}
    positions = suggested_signals_sorted['Type'].map(type_to_pos)
    
    # 시그널 인덱스에 포지션 입력
    data.loc[suggested_signals_sorted['Index'],'position'] = positions.values
    
    # 2) 포지션 누적(가장 최근 시그널이 유지된다고 가정)
    data['position'] = data['position'].replace(0, method='ffill').fillna(0)
    
    # 3) 수익률 계산
    data['returns'] = data['Close'].pct_change().fillna(0)
    data['strategy_returns'] = data['returns'] * data['position']
    
    # 4) 누적 수익률과 MDD 계산
    cumulative_returns = (1 + data['strategy_returns']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (running_max - cumulative_returns) / running_max
    max_drawdown = drawdown.max()
    
    # 5) MDD 감점 (절대값 안전 처리)
    total_fitness -= abs(max_drawdown)

    return total_fitness
