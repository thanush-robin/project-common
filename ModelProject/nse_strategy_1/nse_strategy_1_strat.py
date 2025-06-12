# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:10:19 2020

@author: Nick Elmer
"""

import pandas as pd
import numpy as np
import datetime as dt
import logging


# from TechnicalIndicators import RSI
# import plotly.graph_objects as go
logging.basicConfig(filename="logfilename.log", filemode='w', level=logging.WARN)


class nse1_movingcrossover_strat:
    def __init__(self, price_data, strat_params, generate_signals=False, signal_path=None,
                 starting_capital=100000, max_share_amnt=10, forensics=False):
        # STATIC DATA
        self.max_share_amnt = max_share_amnt
        self.starting_capital = starting_capital
        self.generate_signals = generate_signals
        self.signal_path = signal_path
        self.price_data = price_data
        self.strategy_params = strat_params

        # ENTRY EXIT PASS
        self.entry_pass_df = None
        self.exit_pass_df = None
        # FORESENIC VARIABLES
        self.running_forensics = False
        self.forensic_df = None

        self.foresensic_spy_rank = {}
        if generate_signals:
            # STRATEGY PARAMETERS
            self.week_lookback = None

        self.initialise_strategy()

    def initialise_strategy(self):
        self.initialise_parameters(self.strategy_params)

        if self.generate_signals:
            self.initialise_trackers(self.price_data)
            self.create_strategy_data(self.price_data)
        else:
            self.read_in_pass_df(self.signal_path)

    def initialise_parameters(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)
        # print('\nInitialised Strategy Parameters...')
        return

    def initialise_trackers(self, data):
        # Tracking variables
        # nan_vector = np.empty(len(data.weekly_closes.columns))
        # nan_vector[:] = np.nan
        #
        # self.uptrnd_bool_series = pd.Series(
        #     dict(zip(data.weekly_closes.columns, np.zeros(len(data.weekly_closes.columns)).astype(bool))))
        # self.pullback_bool_series = pd.Series(
        #     dict(zip(data.weekly_closes.columns, np.zeros(len(data.weekly_closes.columns)).astype(bool))))
        return

    def create_strategy_data(self, data):

        # Creation Of Strategy




        logging.info('Strategy preparation complete.')
        return

    def update_strat(self, data, trade_date):
        d = trade_date
        # Updation Of Strategy Each Day.
        entry_signals = []
        exit_signals = []




        return entry_signals, exit_signals





#  Ignore This whole Part
if __name__ == '__main__':
    print('Strategy')
