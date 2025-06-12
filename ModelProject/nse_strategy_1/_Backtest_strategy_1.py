
from ta import momentum
from tqdm import tqdm
import datetime as dt
import time
import os
from math import floor
import pandas as pd
from ast import literal_eval
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Portfolio_with_sector import Portfolio

from PostAnalysis_ import BOA
from PriceData_ import PriceData
from jproperties import Properties
import logging

import warnings

from nse_strategy_1.nse_strategy_1_strat import nse1_movingcrossover_strat

warnings.filterwarnings('ignore')

logging.basicConfig(filename="logfilename.log", filemode='w', level=logging.WARN)



def updates_before_end_of_trade_day(portf, d):
    portf.lookahead_checks(d)
    portf.mark_to_market(d)

def run_backtest(data_obj, strategy, portf):

    entry_signals, exit_signals =[],[]

    for d in tqdm(data_obj.all_dates, desc='Running Backtest...'):

        # ******---------------- EXECUTION CODE ----------------******
        # ******---------------- MAIN D-LOOP LOGIC STARTS HERE ----------------******
        # This has a catch that when its the last day we close all trades and finish backtest


        if d >=pd.Timestamp(dt.date(2001, 6, 1)):
            if d == pd.Timestamp(dt.date(2009, 5, 15)):
                bla = 1

            if d in data_obj.daily_closes.index:

                portf.execute_exit_signals(exit_signals, d)

                portf.execute_entry_signals(entry_signals, d)


                updates_before_end_of_trade_day(portf, d)



                entry_signals, exit_signals = strategy.update_strat(data=data_obj, trade_date=d)





        if d == data_obj.all_dates[-1]:
            datecode = dt.datetime.now()
            datecode_str = datecode.strftime('%Y%m%d_%H%M')

            portf.end_of_backtest(d)

            if not portf.optimise:
                out_loc = f'/home/tharun/Thanush_new_project_v1/ModelProject/nse_strategy_1/Outputs/Backtests/{datecode_str}'
                os.mkdir(out_loc)

                complete_backtest(portf,out_loc,datecode_str)

                print(f"\n--- Backtesting completed. ---")


    return portf


# def run_exe_instr(d,data_obj, portf,portfolioForPh, entry_tickers, exit_tickers,
#                             killswitch_bool,out_loc,phantom_rentry_success,strategy,kill_Switch_Reason,killswitch_active):
#
#     return


def complete_backtest(portfol,out_loc,datecode_str):
    bear_years = [2001, 2002, 2008, 2011, 2015, 2018]
    bull_years = [2003, 2006, 2009, 2010, 2012, 2013, 2014, 2016, 2017, 2019,2020,2021]
    other_years = [2000, 2004, 2005, 2007]
    # POST BACKTEST ANALYSIS
    boa = BOA(portfol, '')

    # can create an output function which formats summary txt and csv file

    # wrtie new columns to equity curve csv
    benchmark = 'NIFTY'
    equity_df = boa.get_equity_df(pricedata, benchmark=benchmark)
    equity_df.to_csv(f'{out_loc}/equity_df_'+datecode_str+'.csv')



    monthly_perf_fig, annual_perf_df = boa.get_performance_plots(portfol.starting_capital)
    monthly_perf_fig.write_html(f'{out_loc}/MonthlyPerformance'+datecode_str+'.html')



    eq_plot = boa.plot_equity_curve(pricedata, with_benchmark=False)
    eq_plot.write_html(f'{out_loc}/EquityCurve_'+datecode_str+'.html')

    summary_str = boa.get_summary(portfol.strat_params)
    summary_str += (f'\n-------------------------------------------------------------|'
                    f'\n             Annual Performance against Benchmark %'
                    f'\n-------------------------------------------------------------|'
                    f'\n{annual_perf_df}')




    print(summary_str)
    with open(f'{out_loc}/Summary_'+datecode_str+'.txt', 'w') as f:
        f.write(summary_str)

    # vol_plot = boa.plot_vol_break_nif()
    # vol_plot.write_html(f'{out_loc}/vol_plot_Curve_' + datecode_str + '.html')

    # get a trade value dict
    trade_track_dict = dict()
    for k in portfol.trade_log.keys():
        trade_track_dict[k] = portfol.trade_log[k]['value_track']

    # boa.plot_excursion_graphs_v2(trade_track_dict)

    boa.trade_list.drop(['value_track'], axis=1, inplace=True)
    boa.trade_list.insert(2, 'sector', '')
    # boa.trade_list['sector']= boa.trade_list.apply (lambda row: sector_finder(row,portfol.sm), axis=1)
    boa.trade_list.to_csv(f'{out_loc}/trade_list_'+datecode_str+'.csv')

    return






if __name__ == '__main__':

    configs = Properties()
    with open(r'/home/tharun/Thanush_new_project_v1/ModelProject/nse_strategy_1/application.properties', 'rb') as read_prop:
        configs.load(read_prop)

    # base_path = r'C:\Users\Nick_Elmer\Documents\BarnesD\spy_breakout'
    # outpath = r'C:\Tharun\Source\Qlib_Demo'

    # ---------------------- PRICE DATA STATIC
    # The date trading will start
    # The date trading will end - left off last two years so we dont over fit

    strategy_name = configs.get("strategy_name").data
    day, month, year = configs.get("start_date").data.split('-')
    start_date = dt.date(int(year), int(month), int(day))
    day, month, year = configs.get("end_date").data.split('-')
    end_date = dt.date(int(year), int(month), int(day))

    rebalance = configs.get("rebalance").data  # 'daily', 'weekly', 'month-end', 'month-start'

    offset = 0
    max_lookback = int(configs.get("max_lookback").data)

    data_source = configs.get("data_source").data  # Either 'Norgate' or 'local_csv

    stock_data_path = r'{}'.format(configs.get("stock_data_path").data)  # folder path  # folder path
    # FOR NORGATE
    data_fields_needed = ['Open', 'High', 'Low', 'Close',
                          'Unadjusted Close']  # The fields needed. If `check_stop_loss` is used, need OHLC
    data_adjustment = 'TotalReturn'  # The type of adjustment of the data

    # ---------------------- STRATEGY STATIC
    starting_cash = float(configs.get("starting_cash").data)  # 2 dp. float
    max_share_amount = int(configs.get("max_share_amount").data)

    strategy_params = configs.get("strategy_params").data
    strategy_params = literal_eval(strategy_params)

    in_out_sampling = {'end_trim_percent': 10,
                       'random_month_percent': 25}

    # State if we need to generate entry/exit signals or if we can pick them up from somewhere
    generate_signals = True
    signal_path = r''
    # Do you want to run only on insample ***
    run_in_sample_test = False
    execution_instr = True
    execution_lookback = 10

    optimise_enable = configs.get("optimise_enable").data
    enable_kill_Switch = bool(configs.get("enable_kill_Switch").data)

    # ------------------***     START of BACKTEST SCRIPT    ***------------------
    start_time = time.time()
    split_time = time.time()
    # 1. create price data object - NEED TO FIX THE IN OUT SAMPLING. ITS HORRIBLE. NEED BETTER LOGIC
    '''
    PriceData is the class where we will be loading the local_csv or Norgate data for open,high,low,close
    and we will get the valid trading days.
    SO we are initializing it with basic parameters
    '''

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



    if run_in_sample_test:
        pricedata.get_in_out_sample_dates_fixed()

    print("\n--- Price Data retreived in %s seconds ---" % (round(time.time() - split_time, 2)))
    split_time = time.time()

    # 2. Initiate strat class and do required steps (I DONT THINK WE NEED STARTING CASH AND MAX SHARES IN STRAT)
    '''
    This rsi_strat is a place for creating strategy parameters and initializing
      the entry_pass and exit_pass.
    '''

    strat = nse1_movingcrossover_strat(pricedata,
                                     strat_params=strategy_params,
                                     generate_signals=generate_signals,
                                     signal_path=signal_path,
                                     starting_capital=starting_cash,
                                     max_share_amnt=max_share_amount)





    '''
     Portfolio is the replica of Orders class
     1.entering trade
     2.closing trade
     3.Executing Entry and Exit signals
     '''

    portfolio = Portfolio(pricedata,
                          strategy_params,
                          starting_capital=starting_cash,
                          max_amnt_shares=max_share_amount,
                          in_sample_test=run_in_sample_test,sector_limit=1,sector_level=1)

    # 4. Run Backtest
    run_backtest(pricedata, strat, portfolio)
