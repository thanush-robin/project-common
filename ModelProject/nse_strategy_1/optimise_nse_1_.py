import multiprocessing
import os
import pickle

import pandas as pd
import itertools
import time as time
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime as dt
import multiprocessing as mp
from copy import deepcopy
from jproperties import Properties
from Portfolio_with_sector import Portfolio
from PriceData_ import PriceData

from nse_strategy_1._Backtest_strategy_1 import run_backtest
from nse_strategy_1.nse_strategy_1_strat import nse1_enhanced_strat


def run_optimisation(start_date,price_data, param_dict, number_of_cpu=6, in_sample=True,
                     boxplots=False, params=[], metrics=[]):
    strt = time.time()
    # ********************************************************
    # ********** PREPARING FOR MULTIPROCESSING ****************
    # Partion the possible combinations of parameters reading for multiprocessing
    # 1.
    # get the keys from our original params to optimise dict
    strat_keys = sorted(param_dict)

    # 2.
    # Create a list of dictionarys containing all possible combinations of k,v pairs from our params to optimise
    param_comb = [dict(zip(strat_keys, prod)) for prod in itertools.product(*(params_to_optimise[i]
                                                                              for i in strat_keys))]

    # 3.
    # we then create a nested dictionary labelling each test case with a integer to be used later
    # Param comb dict is closely related to param comb - dict of dicts instead of list of dicts.
    param_combo_dict = {i: param_comb[i] for i in range(len(param_comb))}

    # 4.
    # Then we have to split the number of possible test combinations into N lists, where N is the number of
    # processes (/cores) we will be running.
    # The results idx_split is a list of lists (containing integers)
    idx_split = list(split(len(param_comb), number_of_cpu))

    # ******************************************
    # ******* START OF MULTIPROCESSING *********
    # 1.
    # Create a manager dict to collate results and define the number of CPUs you want to utilise
    # (NOT SURE WHAT IS BEST/OPTIMUM, can we use all the cpus?)
    manager = mp.Manager()
    wealth_dict = manager.dict()
    trade_dict = manager.dict()
    processes = []
    # Create a ETA on completion that requires you to know the time it takes for one run (~on average)
    eta = round(avg_time_one_run * len(param_comb) / number_of_cpu, 2)  # in mins
    print(f'Running {len(param_comb)} backtests across {number_of_cpu} processes...'
          f'\n ETA: {int(eta / 60)}hrs {int(eta % 60)} mins')

    # 2.
    # Split the backtests across multiple cores using the mp.Process class and providing it a targert function
    # which has been written below.
    for i in range(number_of_cpu):
        p = mp.Process(target=mp_optimisation_process,
                       args=(deepcopy(price_data),
                             param_combo_dict.copy(),
                             idx_split[i],
                             wealth_dict,
                             trade_dict,
                             start_date,
                             in_sample))
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()

    print(f'Completed All Backtests in {round((time.time() - strt) / 60, 2)} mintues')

    # trade_log_dict = {i: trade_dict[i].trade_log for i in range(len(param_comb))}
    # wealth_track_dict = {i: return_dict[i].wealth_track for i in range(len(param_comb))}

    process_results(param_combo_dict, boxplots, params, metrics)


def mp_optimisation_process(price_data, params_dict, idx_slice, wealth_dict, trade_dict, start_date ,in_sample=True):
    # This is the target function that runs on each core when we optimise. The role of this function is to loop through
    # our list of in idx_slice that represents unique test parameters. We then take the unique parameters and
    # instantiate a strategy and portfolio for those parameters and then run a backtest. We then take the portfolio
    # after a backtest and save down the trade log and the wealth track.

    start_cash = 100000  # 2 dp. float
    max_share_amnt = 10
    generate_signals = True
    for i in idx_slice:
        # max_share_amnt = params_dict[i]['number_of_slots']
        opt_strat = nse1_enhanced_strat(price_data,
                                         strat_params=params_dict[i],
                                         generate_signals=generate_signals,
                                         starting_capital=start_cash,
                                         max_share_amnt=max_share_amnt)

        opt_portf = Portfolio(price_data,
                              strat_params=params_dict[i],
                              starting_capital=start_cash,
                              max_amnt_shares=max_share_amnt,
                              optimise=True,
                              in_sample_test=in_sample,sector_limit=1,sector_level=1)




        run_backtest(price_data, opt_strat, opt_portf)

        wealth_dict = opt_portf.wealth_track
        trade_dict = opt_portf.trade_log

        ### WE ARE NOW GOING TO REDUCE THE SIZE OF THE OUTPUTS AND SAVE THEM DOWN.
        trade_log_df = pd.DataFrame.from_dict(trade_dict, orient='index')

        temp_dict = {}
        winner_mask = trade_log_df['profit'] > 0
        loser_mask = trade_log_df['profit'] < 0

        # write to summary dict
        temp_dict['test_number'] = i
        for k, v in params_dict[i].items():
            temp_dict[k] = v
        temp_dict['total_profit'] = trade_log_df['profit'].sum().round(2)
        temp_dict['#trades'] = len(trade_log_df)
        temp_dict['avg_trade_length'] = round(trade_log_df['time_in_trade'].mean())
        temp_dict['%_winners'] = round((trade_log_df['profit'] > 0).sum() / len(trade_log_df) * 100, 2)
        temp_dict['win_loss_ratio'] = round(len(trade_log_df[winner_mask]) / len(trade_log_df[loser_mask]), 4)
        temp_dict['avg_profit'] = round(trade_log_df['profit'].mean(), 2)
        temp_dict['avg_profit%'] = round(trade_log_df['profit%'].mean(), 4)
        temp_dict['avg_winner_profit'] = round(trade_log_df[winner_mask]['profit'].mean(), 2)
        temp_dict['avg_winner_profit%'] = round(trade_log_df[winner_mask]['profit%'].mean(), 4)
        temp_dict['avg_winner_length'] = round(trade_log_df[winner_mask]['time_in_trade'].mean())
        temp_dict['avg_loser_profit'] = round(trade_log_df[loser_mask]['profit'].mean(), 2)
        temp_dict['avg_loser_profit%'] = round(trade_log_df[loser_mask]['profit%'].mean(), 4)
        temp_dict['avg_loser_length'] = round(trade_log_df[loser_mask]['time_in_trade'].mean())

        ### INTRODUCING THE USE OF PICKLE FILES
        a_file = open(f'PickleFiles/{i}_wealth_track.pkl', 'wb')
        pickle.dump(wealth_dict, a_file)
        a_file.close()

        b_file = open(f'PickleFiles/{i}_trade_log.pkl', 'wb')
        pickle.dump(temp_dict, b_file)
        b_file.close()

        del a_file
        del b_file
        del opt_portf
        del opt_strat
        del wealth_dict
        del trade_dict
        del trade_log_df
        del temp_dict

    # print(temp_dict)
    # return_dict.update(temp_dict)


def process_results(param_track_dict, boxplot=False, params=[], metrics=[], equity_plot_limit=250):
    # We then take the dictionaries containing all of the combinations of parameters and collection of trade logs
    # and wealth track, loop through and process the performance of each test

    summary_dict = {}
    error_dict = {}
    # bf = Bull_Market_Rally_Fall(bull_path=r'C:\Users\tharu\Desktop\Bull and Bear Rally Fall\BULL_MARKET_SEPERATED.csv'
    #                                   ,bear_path=r'C:\Users\tharu\Desktop\Bull and Bear Rally Fall\BEAR_MARKET_SEPERATED.csv')
    trade_log_dict = {}
    bull_rally_fall ={}
    bear_rally_fall ={}
    for i in param_track_dict.keys():
        # Read in trade pickle and create trade log
        try:
            trade_pkl = open(f'PickleFiles/{i}_trade_log.pkl', 'rb')
            temp_dict = pickle.load(trade_pkl)

            # Read in wealth track pickle and anlayse
            wealth_pkl = open(f'PickleFiles/{i}_wealth_track.pkl', 'rb')
            wealth_track = pickle.load(wealth_pkl)

            equity_df = pd.DataFrame()
            wealth_series = pd.Series(wealth_track)
            equity_df['equity'] = wealth_series.values
            equity_df['year'] = wealth_series.index.year
            equity_df['month'] = wealth_series.index.month
            equity_df['day'] = wealth_series.index.day

            yrly_equity_ret = (((equity_df.groupby(['year'])[['equity']].last() -
                                 equity_df.groupby(['year'])[['equity']].first()) /
                                starting_cash) * 100).round(2)

            # daily_ret = ((wealth_series - wealth_series.shift(1)) / wealth_series.shift(1)) * 100

            equit_val = [x[0] for x in yrly_equity_ret.values]
            temp_dict.update(dict(zip(yrly_equity_ret.index, equit_val)))

            temp_dict = get_drawdown_stats(wealth_series, temp_dict)


            # calc,perc_inc,score,ent = bf.scoring_bull_rally_fall1(wealthTrack=wealth_track)
            # temp_dict.update(dict(zip(bf.bull_market_df['calc_columns'], calc)))
            # temp_dict.update(dict(zip(bf.bull_market_df['perc_inc_columns'], perc_inc)))
            # temp_dict.update(dict(zip(bf.bull_market_df['score_columns'], score)))
            # temp_dict['bull_rally_score'] = ent.rally_score
            # temp_dict['bull_fall_score'] = ent.fall_score
            # temp_dict['bullrallypctchanges'] = ent.rally_pct_change
            # temp_dict['bullfallpctchanges'] = ent.fall_pct_change
            # temp_dict['sumBRminusBf'] = ent.rally_minus_fall
            # temp_dict['sumBullPctRallyFall'] = ent.rally_fall_pct_change
            # temp_dict['sumBullRally'] = ent.rally_profit
            # temp_dict['sumBullFall'] = ent.fall_profit
            #
            #
            #
            # calc1,perc1_inc,score1,ent1 = bf.scoring_bear_rally_fall1(wealthTrack=wealth_track)
            # temp_dict.update(dict(zip(bf.bear_market_df['calc_columns'], calc1)))
            # temp_dict.update(dict(zip(bf.bear_market_df['perc_inc_columns'], perc1_inc)))
            # temp_dict.update(dict(zip(bf.bear_market_df['score_columns'], score1)))
            #
            # temp_dict['bear_rally_score'] = ent1.rally_score
            # temp_dict['bear_fall_score'] = ent1.fall_score
            # temp_dict['bearrallypctchanges'] = ent1.rally_pct_change
            # temp_dict['bearfallpctchanges'] = ent1.fall_pct_change
            # temp_dict['sumBearRminusBearf'] = ent1.rally_minus_fall
            # temp_dict['sumBearPctRallyFall'] = ent1.rally_fall_pct_change
            # temp_dict['sumBearRally'] = ent.rally_profit
            # temp_dict['sumBearFall'] = ent.fall_profit

            trade_log_dict[i] = dict(temp_dict)
            del temp_dict
        except Exception as e:
            print(e)


    trade_log_df = pd.DataFrame.from_dict(trade_log_dict, orient='index')

    # Build the graphs
    top_test_nums = trade_log_df['total_profit'].sort_values(ascending=False).index[:equity_plot_limit]

    # SAVE Outputs
    datecode = dt.datetime.now()
    datecode_str = datecode.strftime('%Y%m%d_%H%M')
    out_loc = f'nse_strategy_1/Outputs/Optimisation/{datecode_str}'
    os.mkdir(out_loc)
    trade_log_df.to_csv(f'{out_loc}/OptimisationReport_{datecode_str}.csv', index=False)

    opt_plot = plot_equity_curves(top_test_nums)
    opt_plot.write_html(f'{out_loc}/OptimisationPlot_{datecode_str}.html')

    if boxplot:
        get_optimisation_boxplots(out_loc, f'OptimisationReport_{datecode_str}', params, metrics)

    return print('Optimisation run finished. \n Remeber to delete pickle files when finished.')


def get_drawdown_stats(wealth_series, temp_dict):
    drawdown = (wealth_series - wealth_series.cummax())
    # DB added for drawdown analysis
    d = drawdown.copy()
    d.replace(0, np.nan, inplace=True)
    sparse_ts = d.astype(pd.SparseDtype('float'))
    block_locs = zip(sparse_ts.values.sp_index.to_block_index().blocs,
                     sparse_ts.values.sp_index.to_block_index().blengths)
    drawdown_dict = {'start_date': [],
                     'end_date': [],
                     'length': [],
                     'max_dd': [],
                     'avg_dd': []
                     }
    for start, length in block_locs:
        if length > 1:
            temp_series = d.iloc[start:(start + length)]
            drawdown_dict['start_date'].append(temp_series.index[0])
            drawdown_dict['end_date'].append(temp_series.index[-1])
            drawdown_dict['length'].append(length)
            drawdown_dict['max_dd'].append(temp_series.min())
            drawdown_dict['avg_dd'].append(temp_series.mean())

    dd_df = pd.DataFrame(drawdown_dict)

    top_ten_df = dd_df.sort_values(["max_dd"]).iloc[:10]
    temp_dict['max_drawdown'] = top_ten_df.iloc[0]['max_dd']
    temp_dict['length_max_drawdown'] = top_ten_df.iloc[0]['length']
    temp_dict['avg_top10_dd'] = round(top_ten_df['max_dd'].mean(), 2)
    temp_dict['avg_length_top10_dd'] = round(top_ten_df['length'].mean())
    return temp_dict


def plot_equity_curves(test_numbers, title=None):  # NEED TO REMOVE THE REFRENCE OF PRICE DATA. PASS IT AS ARG
    """
    Provide some test numbers from the optimisation just run to plot.

    After running an optimisation, whilst you still have the
    data.optimisation_wealth_tracks variable accessible by your editor. This
    function will produce a plotly line graph of the tests that you provide as
    a list.

    Parameters
    ----------
    test_numbers : list
        A list of integers referring to test numbers of an optimsation report.
    title : str, default None
        The title you would like to appear at the top of the plot.

    Returns
    -------
    None
        Automatically opens a .html plot in your default browser.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    fig = go.Figure()
    max_profit = 0
    min_profit = 0

    for i in test_numbers:
        wealth_pkl = open(f'PickleFiles/{i}_wealth_track.pkl', 'rb')
        wealth_track = pickle.load(wealth_pkl)
        wealth_series = pd.Series(wealth_track)
        profit_series = wealth_series - wealth_series[0]
        if max(profit_series) > max_profit:
            max_profit = max(profit_series)
        if min(profit_series) > min_profit:
            min_profit = min(profit_series)
        final_equity = profit_series.iloc[-1]

        fig.add_trace(go.Scatter(x=profit_series.index, y=profit_series, name=str(i) + '_' + str(round(final_equity))))

    if run_in_sample_test:
        in_out_samples = pd.DataFrame(index=pricedata.all_dates.union(pricedata.oos_dates), columns=['IS', 'OOS'])
        in_out_samples['IS'].loc[pricedata.is_dates] = max_profit * 1.05
        in_out_samples['OOS'].loc[pricedata.oos_dates] = max_profit * 1.05
        in_out_samples.fillna(0, inplace=True)

        ### CVE added this 13/10/2021
        in_out_samples_lower = pd.DataFrame(index=pricedata.all_dates.union(pricedata.oos_dates), columns=['IS', 'OOS'])
        in_out_samples_lower['IS'].loc[pricedata.is_dates] = min_profit * 1.05
        in_out_samples_lower['OOS'].loc[pricedata.oos_dates] = min_profit * 1.05
        in_out_samples_lower.fillna(0, inplace=True)
        # 0 Line
        zero_line = pd.DataFrame(index=pricedata.all_dates.union(pricedata.oos_dates), columns=['zero'])
        zero_line.fillna(0, inplace=True)

        fig.add_trace(go.Scatter(x=in_out_samples.index, y=in_out_samples['IS'],
                                 name='In Sample', marker_color='green', fill='tozeroy', line_shape='hv', opacity=0.05))
        fig.add_trace(go.Scatter(x=in_out_samples.index, y=in_out_samples['OOS'],
                                 name='Out of Sample', marker_color='red', fill='tozeroy', line_shape='hv',
                                 opacity=0.05))
        ### CVE added this 13/10/2021
        fig.add_trace(go.Scatter(x=in_out_samples_lower.index, y=in_out_samples_lower['IS'],
                                 name='In Sample_lower', marker_color='green', fill='tozeroy', line_shape='hv',
                                 opacity=0.05))
        fig.add_trace(go.Scatter(x=in_out_samples_lower.index, y=in_out_samples_lower['OOS'],
                                 name='Out of Sample_lower', marker_color='red', fill='tozeroy', line_shape='hv',
                                 opacity=0.05))
        fig.add_trace(go.Scatter(x=in_out_samples_lower.index, y=zero_line['zero'],
                                 name='zero_line', marker_color='white', fill='tozeroy', line_shape='hv', opacity=0.05))
        ###

    # fig.layout.update(template='plotly_dark', title=title)
    fig.update_layout(template='plotly_dark', title=title)
    plot(fig, auto_open=True)

    return fig


def get_optimisation_boxplots(opt_report_path, report_name, params, metrics):
    opt_df = pd.read_csv(f'{opt_report_path}/{report_name}.csv')

    for u in params:
        fig = make_subplots(rows=2, cols=3, subplot_titles=metrics)
        i = 0
        for p in metrics:
            fig.add_trace(go.Box(x=opt_df[u], y=opt_df[p], notched=False, showlegend=False),
                          col=(i % 3) + 1, row=(i // 3) + 1)
            i += 1
        fig.update_layout(title=f'Optimisation of {u}')
        fig.show()
        fig.write_html(f'{opt_report_path}/{u}_boxplot.html')
    return


def split(b, n):
    a = list(np.arange(b))
    k, m = divmod(b, n)

    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == '__main__':
    configs = Properties()
    with open('selectiveapplication.properties', 'rb') as read_prop:
        configs.load(read_prop)

    base_path = r'{}'.format(configs.get("base_path").data)  # Location of bespoke scripts

    # ---------------------- PRICE DATA STATIC
    day, month, year = configs.get("start_date").data.split('-')
    start_date = dt.date(int(year), int(month), int(day))  # The date trading will start

    day, month, year = configs.get("end_date").data.split('-')
    end_date = dt.date(int(year), int(month),
                       int(day))  # The date trading will end - left off last two years so we dont over fit
    rebalance = configs.get("rebalance").data  # 'daily', 'weekly', 'month-end', 'month-start'
    offset = 0
    max_lookback = int(configs.get("max_lookback").data)

    data_source = configs.get("data_source").data  # Either 'Norgate' or 'local_csv
    stock_data_path = r'{}'.format(configs.get("stock_data_path").data)  # folder path
    # FOR NORGATE
    data_fields_needed = ['Open', 'High', 'Low', 'Close',
                          'Unadjusted Close']  # The fields needed. If `check_stop_loss` is used, need OHLC
    data_adjustment = 'TotalReturn'  # The type of adjustment of the data

    # ---------------------- STRATEGY STATIC
    starting_cash = float(configs.get("starting_cash").data)  # 2 dp. float
    max_share_amount = int(configs.get("max_share_amount").data)
    enable_kill_Switch = bool(configs.get("enable_kill_Switch").data)
    # params_to_optimise = {'week_lookback': [7],
    #                       'upthrust_weeks': [2],
    #                       'end_uptrend': [12],
    #                       'stock_hist_percent': [0.2],
    #                       'index_hist_percent': [0.3],
    #                       'exit_weeks': [11],
    #                       'stoploss_pct': [0.2],
    #                       'takeprof_pct': [None],
    #                       'giveback_pct': [None],
    #                       'lt_price_lkback': [52],
    #                       'breakev_thresh': [None],
    #                       'breakev_take': [None],
    #                       'freeze_lookback': [6],
    #                       'freeze_limit': [8000],
    #                       'freeze_out_days': [5]}
    params_to_optimise = {'rsi_entry_lookback': [2],
                       'rsi_exit_lookback': [20],
                       'rsi_entry_level': [10],
                       'rsi_exit_level': [86],
                       'historic_Volatility': [20],
                       'moving_average': [40],
                       'moving_average_126': [126],
                       'moving_average_63': [63],
                       'skip_days_param': [5],
                       'vol_lookback_for_sharpe_param': [126],
                       'first_filter_amount_param': [100],
                       'takeprof_pct': [0.5],
                       'stoploss_pct': [0.1],
                       'exit_maxtime': [98],
                       'unacceptable_profit': [200],
                       'nft_lookback': [25],
                       'nft_limit': [10],
                       'nif_vol_lookback': [5],
                       'nif_vol_ma': [50,75,100,120,150,200],
                       'vol_moving_average': [3],
                       'enable_sector_filteration': [False],
                       }

    # 'freeze_lookback': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
    # 'freeze_limit': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
    # 'freeze_out_days': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # params_to_optimise = {'rsi_lookback_entry': [3],
    #                       'rsi_entry_level': [28],
    #                       'rsi_exit_level': [82, 84],
    #                       'close_level_exit': [5],
    #                       'stoploss_pct': [0.15],
    #                       'max_profit': [None],
    #                       'close_level_entry': [10],
    #                       'h_volatility_lookback': [52],
    #                       'spy_lookback': [13, 17, 21, 25],
    #                       'spy_limit': [11, 13, 15, 18],
    #                       'takeprof_pct': [0.15],
    #                       'giveback_pct': [None],
    #                       'exit_maxtime': [100000],
    #                       'stp_loss_break_time': [21],
    #                       'killSwitch': [False],
    #                       'suspension_period': [0],
    #                       'sample_days': [7],
    #                       'constant_suspension_period': [10],
    #                       'equity_loss_trigger': [6000],
    #                       'perc_inc_btw_slots': [0.2],
    #                       'uptrend_sample_days': [3],
    #                       # 'phantom_slots': [],
    #                       'perc_inc_btw_zero_last': [1],
    #                       'total_percent_increase': [1],
    #                       'countConsecutiveInc': [0],
    #                       'enable_stop_loss_enable': [True],
    #                       }

    in_out_sampling = {'end_trim_percent': 9.5,
                       'random_month_percent': 25}

    # ------------------***     Get PRICEDATA   ***------------------
    start_time = time.time()
    split_time = time.time()
    # 1. create price data object - NEED TO FIX THE IN OUT SAMPLING. ITS HORRIBLE. NEED BETTER LOGIC
    pricedata = PriceData(start_dt=start_date,
                          end_dt=end_date,
                          rebalance=rebalance,
                          offset=offset,
                          max_lkback=max_lookback,
                          data_source=data_source,
                          data_path=stock_data_path,
                          in_out_sampling=in_out_sampling,
                          fields=data_fields_needed,
                          price_adjust=data_adjustment)
    pricedata.get_vix_prices_df(ticker='$NIF', interval='D')
    pricedata.nifty_500 = pd.read_csv(r'D:\Tharun\NSE\prices\Nifty_500_Historical_Data.csv', index_col=['Date'],
                                      parse_dates=True)
    pricedata.nifty_500 = pricedata.nifty_500.sort_index()

    pricedata.get_in_out_sample_dates_fixed()

    print("--- Price Data retreived in %s seconds ---" % (round(time.time() - split_time, 2)))
    split_time = time.time()

    # # hack for daily bars
    # pricedata.weekly_closes = pricedata.daily_closes
    # pricedata.weekly_opens = pricedata.daily_opens
    # pricedata.weekly_lows = pricedata.daily_lows
    # pricedata.weekly_highs = pricedata.daily_highs

    # ------------------***     START of Optimisation SCRIPT    ***------------------

    # Do you want to produce boxplots for metrics shown?
    boxplots = True
    params = ['rsi_entry_lookback',
              'rsi_exit_lookback',
              'rsi_entry_level',
              'rsi_exit_level',
              'historic_Volatility',
              'moving_average',
              ]

    perf_metrics = ['total_profit',
                    '%_winners',
                    'max_drawdown',
                    '#trades',
                    'avg_profit',
                    'length_max_drawdown']
    run_in_sample_test = False

    # ------------------***     MULTIPROCESSING OPTIMISATION  ***------------------
    avg_time_one_run = 1  # in minutes
    run_optimisation(start_date,pricedata, params_to_optimise, number_of_cpu=5, in_sample=run_in_sample_test,
                     boxplots=boxplots, params=params, metrics=perf_metrics)
