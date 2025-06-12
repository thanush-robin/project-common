import pandas as pd
import numpy as np
import os
import datetime as dt


from PriceData_ import PriceData
from plotly.offline import plot
import plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jproperties import Properties

class BOA:  # Backtest Output Analysis
    def __init__(self, portfolio, output_path=None, starting_capital=100000, name='strat'):
        self.name = name
        self.starting_capital = starting_capital
        if isinstance(portfolio, str):
            # this assumes we are running using csv outputs located in the str passed
            self.input_path = portfolio
            self.output_path = portfolio
            for fname in os.listdir(self.input_path):
                if '.csv' in fname:
                    setattr(self, f'{fname.split("_")[0]}_{fname.split("_")[1]}',
                            pd.read_csv(self.input_path + '\\' + fname, index_col=0, parse_dates=True, dayfirst=True))

            self.trade_list['open_date'] = pd.to_datetime(self.trade_list['open_date'])
            self.trade_list['close_date'] = pd.to_datetime(self.trade_list['close_date'])
        else:
            # the other option is Portfolio object passed directly after backtest has been run
            self.output_path = output_path
            self.equity_curve = (pd.DataFrame.from_dict(portfolio.wealth_track, orient='index') - self.starting_capital)
            self.equity_curve.columns = ['equity']
            self.trade_list = pd.DataFrame.from_dict(portfolio.trade_log, orient='index')
            self.utility_s = pd.Series(portfolio.utility_track, name='utility')
            self.portfolio = portfolio
            self.equity_df = pd.DataFrame()

            self.dd_series = self.get_drawdown_stats()
            # self.db = DatabaseConnection()

            # self.Spy_returns_dict = self.db.get_Spy_Results()

    def get_rolling_strategy_beta(self, data_obj, period=220, benchmark='SPY'):
        beta_df = pd.DataFrame(self.equity_curve.pct_change())
        beta_df.columns = ['equity']
        data_obj.get_benchmark_prices(benchmark)
        beta_df = beta_df.join(data_obj.benchmark.pct_change())
        # beta_df = pd.concat([beta_df, data_obj.benchmark.pct_change()], axis=1)
        cov_df = beta_df.rolling(period).cov().unstack()['equity'][benchmark]
        var_market_df = beta_df[benchmark].to_frame().rolling(period).var()

        beta_df['rolling_beta'] = (cov_df / (var_market_df.T)).T

        return beta_df['rolling_beta']

    def get_rolling_strat_correlation(self, data_obj, period=220, benchmark='SPY'):
        corr_df = pd.DataFrame(self.equity_curve.pct_change())
        corr_df.columns = ['equity']
        data_obj.get_benchmark_prices(benchmark)
        corr_df = corr_df.join(data_obj.benchmark.pct_change())
        # corr_df = pd.concat([corr_df, data_obj.benchmark.pct_change()], axis=1)
        corr_df['rolling_correlation'] = corr_df.rolling(period).corr().unstack()['equity'][benchmark]

        return corr_df['rolling_correlation']

    def get_drawdown_stats(self):
        drawdown = (self.equity_curve - self.equity_curve.cummax())['equity']
        max_dd = drawdown.min()
        max_dd_percent = 100 * max_dd / self.starting_capital
        max_dd_date = drawdown.idxmin()
        max_dd_start = drawdown.where(drawdown == 0, np.nan).loc[:max_dd_date].last_valid_index()
        max_dd_end = drawdown.where(drawdown == 0, np.nan).loc[max_dd_date:].first_valid_index()
        max_dd_length = len(drawdown[max_dd_start:max_dd_end])
        # avg_dd =
        stats = {'Max Drawdown': max_dd,
                 'Max Drawdown %': max_dd_percent,
                 'Length of Max Drawdown': max_dd_length,
                 'Average Drawdown': drawdown.mean()}  # I didn't do drawdown[drawdown<0].mean() because I wanted to include the 0's
        drawdown.name = 'drawdown'
        setattr(self, 'dd_series', drawdown)
        setattr(self, 'dd_stats', stats)

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
                drawdown_dict['max_dd'].append(temp_series.min())   #.round(2))
                drawdown_dict['avg_dd'].append(temp_series.mean())  #.round(2))

        dd_df = pd.DataFrame(drawdown_dict)
        setattr(self, 'dd_df', dd_df)

        return drawdown

    def get_equity_df(self, data_obj, benchmark='SPY'):
        equity_df = self.equity_curve.join(self.dd_series)
        equity_df = equity_df.join(self.utility_s)

        # data_obj.get_benchmark_prices(benchmark)
        # equity_df['benchmark'] = data_obj.benchmark
        # equity_df['corr'] = self.get_rolling_strat_correlation(data_obj, period=220, benchmark=benchmark)
        # equity_df['beta'] = self.get_rolling_strategy_beta(data_obj, period=220, benchmark=benchmark)
        self.equity_df = equity_df
        return equity_df

    def get_summary(self, strat_params=None):
        trade_list = self.trade_list
        equity = self.equity_curve.copy()
        equity.replace(0, np.nan, inplace=True)
        winner_mask = trade_list['profit'] > 0
        loser_mask = trade_list['profit'] < 0
        profit_factor = (trade_list[winner_mask]['profit'].sum() / abs(trade_list[loser_mask]['profit'].sum())).round(2)
        exp_val = (1 / len(trade_list)) * ((len(trade_list[winner_mask]) * trade_list[winner_mask]['profit'].mean()) -
                                           (len(trade_list[loser_mask]) * abs(trade_list[loser_mask]['profit'].mean())))
        expectation = exp_val / abs(trade_list[loser_mask]['profit'].mean())
        cum_ret = self.equity_curve.iloc[-1]['equity'] / self.starting_capital
        ann_ret = (cum_ret ** (365 / (equity.index[-1] - equity.index[0]).days)) - 1
        ann_vol = equity['equity'].pct_change().std()

        max_dd_index = self.dd_df['max_dd'].idxmin()
        ten_worst_dd = self.dd_df.sort_values(["max_dd"]).iloc[:10].reset_index(drop=True)

        strat_str = ''
        if strat_params is not None:
            strat_str = "\n".join(f"{k: <20} {v}" for k, v in strat_params.items())

        summary_str = (
            f'------------------------------------------------------------|'
            f'\n              {self.name.upper()} Summary Performance          '
            f'\n------------------------------------------------------------|'
            f'\n{strat_str}'
            f'\n{"starting capital": <20} {self.starting_capital}\n'
            f''
            f'\n{"Total Profit ($):":<20}  {round(self.equity_curve.iloc[-1]["equity"] / 1000, 2):>10}k ({round(cum_ret * 100, 2)}%)'
            f'\n{"CAGR:":<20}  {round(ann_ret * 100, 2):>10}%'
            f'\n{"Volatility:":<20}  {round(ann_vol * 100, 2):>10}%\n'
            f'\n{"Total Trades:":<20}  {len(trade_list):>10}'
            f'\n{"Win Rate:":<20}  {round((len(trade_list[winner_mask]) / len(trade_list)) * 100, 2):>10}% ({len(trade_list[winner_mask])})\n'
            f'\n{"Avg Trade Profit ($):":<20}  {round(trade_list["profit"].mean(), 2):>10} ({round(trade_list["profit%"].mean() * 100, 2)}%)'
            f'\n{"Avg Trade Profit RSI sinals ($):":<20}  {round(trade_list[trade_list["open_reason"] == "Buy Signal"]["profit"].mean(), 2):>10} ({round(trade_list[trade_list["open_reason"] == "Buy Signal"]["profit%"].mean() * 100, 2)}%)'
            f'\n{"Avg Trade strategy Profit ($):":<20}  {round(trade_list[trade_list["open_reason"] != "Buy Signal"]["profit"].mean(), 2):>10} ({round(trade_list[trade_list["open_reason"] != "Buy Signal"]["profit%"].mean() * 100, 2)}%)'
            f'\n{"Profit Factor:":<20}  {profit_factor:>10}'
            f'\n{"Expected Value ($):":<20}  {round(exp_val, 2):>10} \n'
            f'\n{"Max Drawdown ($):":<20}  {round(self.dd_df["max_dd"].min() / 1000, 2):>10}k ({round(self.dd_df["max_dd"].min() * 100 / self.starting_capital, 2)}%)'
            f'\n{"Len of Max Drawdown:":<20}  {self.dd_df.iloc[max_dd_index]["length"]:>10} days'
            f'\n{"Average trade length:":<20}  {round(trade_list["time_in_trade"].sum()/len(trade_list),2):>10} '
            f'\n{"Average utility length:":<20}  {round(self.equity_df["utility"].mean(),2):>10} '
            # f'\n{"Avg Drawdown ($):":<20}  {round(self.dd_df["max_dd"].mean() / 1000, 2):>10}k'
            f'\n------------------------------------------------------------|'
            f'\n                    Trade Analysis                          |'
            f'\n------------------------------------------------------------|'
            f'\n{"Total Trades:": <15} {len(trade_list): >6}           '
            f'\n{"Total Winners:": <15} {len(trade_list[winner_mask]): >6} '
            f' ({round((len(trade_list[winner_mask]) / len(trade_list)) * 100, 2)}%) '
            f'\n{"Total Losers:": <15} {len(trade_list[loser_mask]): >6} '
            f' ({round((len(trade_list[loser_mask]) / len(trade_list)) * 100, 2)}%) '
            f'\n------------------------------------------------------------|'
            f'\n                  Profit % Distrubtion \n'
            f'\n  All Trades    |      Winners      |       Losers      |'
            f'\nMin  =  {round(np.quantile(trade_list["profit%"], 0) * 100, 2):>5}%  |   Min  =  {round(np.quantile(trade_list[winner_mask]["profit%"] * 100, 0), 2):>5}%  |   Min  =  {round(np.quantile(trade_list[loser_mask]["profit%"], 0) * 100, 2):>5}%  |'
            f'\n1Q   =  {round(np.quantile(trade_list["profit%"], 0.25) * 100, 2):>5}%  |   1Q   =  {round(np.quantile(trade_list[winner_mask]["profit%"] * 100, 0.25), 2):>5}%  |   1Q   =  {round(np.quantile(trade_list[loser_mask]["profit%"], 0.25) * 100, 2):>5}%  |'
            f'\nMed  =  {round(np.quantile(trade_list["profit%"], 0.5) * 100, 2):>5}%  |   Med  =  {round(np.quantile(trade_list[winner_mask]["profit%"] * 100, 0.5), 2):>5}%  |   Med  =  {round(np.quantile(trade_list[loser_mask]["profit%"], 0.5), 2) * 100:>5}%  |'
            f'\n3Q   =  {round(np.quantile(trade_list["profit%"], 0.75) * 100, 2):>5}%  |   3Q   =  {round(np.quantile(trade_list[winner_mask]["profit%"] * 100, 0.75), 2):>5}%  |   3Q   =  {round(np.quantile(trade_list[loser_mask]["profit%"], 0.75) * 100, 2):>5}%  |'
            f'\nMax  =  {round(np.quantile(trade_list["profit%"], 1), 2) * 100:>5}%  |   Max  =  {round(np.quantile(trade_list[winner_mask]["profit%"] * 100, 1), 2):>5}%  |   Max  =  {round(np.quantile(trade_list[loser_mask]["profit%"], 1) * 100, 2):>5}%  |\n'
            f'\nAvg  =  {round(np.mean(trade_list["profit%"]) * 100, 2):>5}%  |   Avg  =  {round(np.mean(trade_list[winner_mask]["profit%"]) * 100, 2):>5}%  |   Avg  =  {round(np.mean(trade_list[loser_mask]["profit%"]) * 100, 2):>5}%  |'
            f'\n-------------------------------------------------------------|'
            f'\n                BizDays in Trade Distribution \n'
            f'\n  All Trades   |      Winners       |       Losers      |'
            f'\nMin  =  {round(np.quantile(trade_list["time_in_trade"], 0)):>5}  |   Min  =  {round(np.quantile(trade_list[winner_mask]["time_in_trade"], 0)):>5}    |   Min  =  {round(np.quantile(trade_list[loser_mask]["time_in_trade"], 0)):>5}   |'
            f'\n1Q   =  {round(np.quantile(trade_list["time_in_trade"], 0.25)):>5}  |   1Q   =  {round(np.quantile(trade_list[winner_mask]["time_in_trade"], 0.25)):>5}    |   1Q   =  {round(np.quantile(trade_list[loser_mask]["time_in_trade"], 0.25)):>5}   |'
            f'\nMed  =  {round(np.quantile(trade_list["time_in_trade"], 0.5)):>5}  |   Med  =  {round(np.quantile(trade_list[winner_mask]["time_in_trade"], 0.5)):>5}    |   Med  =  {round(np.quantile(trade_list[loser_mask]["time_in_trade"], 0.5)):>5}   |'
            f'\n3Q   =  {round(np.quantile(trade_list["time_in_trade"], 0.75)):>5}  |   3Q   =  {round(np.quantile(trade_list[winner_mask]["time_in_trade"], 0.75)):>5}    |   3Q   =  {round(np.quantile(trade_list[loser_mask]["time_in_trade"], 0.75)):>5}   |'
            f'\nMax  =  {round(np.quantile(trade_list["time_in_trade"], 1)):>5}  |   Max  =  {round(np.quantile(trade_list[winner_mask]["time_in_trade"], 1)):>5}    |   Max  =  {round(np.quantile(trade_list[loser_mask]["time_in_trade"], 1)):>5}   |\n'
            f'\nAvg  =  {round(np.mean(trade_list["time_in_trade"])):>5}  |   Avg  =  {round(np.mean(trade_list[winner_mask]["time_in_trade"])):>5}    |   Avg  =  {round(np.mean(trade_list[loser_mask]["time_in_trade"])):>5}   |'
            f'\n------------------------------------------------------------|'
            f'\n            Drawdown Analysis - 10 worst cases              |'
            f'\n------------------------------------------------------------|'
            f'\n{"Worst Drawdown:":<30}  {round(self.dd_df["max_dd"].min() / 1000, 2):>10}k'
            f'\n{"Length of Worst Drawdown:":<30}  {self.dd_df.iloc[max_dd_index]["length"]:>10}'
            f'\n{"Average of 10 Worst:":<30}  {round(ten_worst_dd["max_dd"].mean() / 1000, 2):>10}k'
            f'\n{"Average Length of 10:":<30}  {round(ten_worst_dd["length"].mean()):>10}'
            f'\n-------------------------------------------------------------|'
            f'\n                   Top 10 Worst Drawdowns \n'
            f'\n{ten_worst_dd}'
        )

        return summary_str

    def get_performance_plots(self, starting_capital=100000,enable_auto_open = False):
        equity_df = self.equity_df.copy()

        equity_df['equity'] = equity_df['equity'] + self.starting_capital
        equity_df['year'] = equity_df.index.year
        equity_df['month'] = equity_df.index.month
        equity_df['day'] = equity_df.index.day

        # We do not look at compounding returns as we dont reinvest profits.
        # annual_ret = ((equity_df.groupby(['year'])[['equity', 'benchmark']].last() /
        #                equity_df.groupby(['year'])[['equity', 'benchmark']].first() - 1) * 100).round(2)
        equity_ann_ret = (((equity_df.groupby(['year'])[['equity']].last() - equity_df.groupby(['year'])[
            ['equity']].first()) /
                           starting_capital) * 100).round(2)
        # benchmark_ann_ret = ((equity_df.groupby(['year'])[['benchmark']].last() /
        #                       equity_df.groupby(['year'])[['benchmark']].first() - 1) * 100).round(2)

        annual_ret = pd.concat([equity_ann_ret], axis=1)
        annual_ret.replace(0, np.nan, inplace=True)
        # annual_ret['diff'] = [ annual_ret.benchmark[i] + annual_ret.equity[i] if annual_ret.equity[i] < 0 or annual_ret.benchmark[i] < 0 else annual_ret.equity[i]-annual_ret.benchmark[i] for i in annual_ret.equity.index]
        utility_monthly_return = round(equity_df.groupby(['year'])[['utility']].mean(), 2)
        annual_ret['utility_year_Avg'] = utility_monthly_return


        # monthly_ret = ((equity_df.groupby(['year', 'month'])[['equity', 'benchmark']].last() /
        #                 equity_df.groupby(['year', 'month'])[['equity', 'benchmark']].first() - 1) * 100).round(2)
        monthly_equity_ret = (((equity_df.groupby(['year', 'month'])[['equity']].last() -
                                equity_df.groupby(['year', 'month'])[['equity']].first()) /
                               starting_capital) * 100).round(2)
        # monthly_bench_ret = ((equity_df.groupby(['year', 'month'])[['benchmark']].last() /
        #                 equity_df.groupby(['year', 'month'])[['benchmark']].first() - 1) * 100).round(2)



        monthly_ret = pd.concat([monthly_equity_ret], axis=1)
        # monthly_ret['diff'] = monthly_ret.equity - monthly_ret.benchmark
        final_df = pd.DataFrame()
        # final_df = monthly_ret['diff'].unstack()
        final_df['utility_year_Avg'] = utility_monthly_return
        annual_ret['score'] = float(0)
        weaker_years = ['2001', '2002', '2008', '2011', '2015', '2018']
        bull_years = ['2003', '2006', '2009', '2010', '2012', '2013', '2014', '2016', '2017', '2019', '2020', '2021']
        other_years = ['2000', '2004', '2005', '2007']
        annual_ret.fillna(0,inplace=True)
        # for index, row in annual_ret.iterrows():
        #     print(row.name)
        #     for w_y in weaker_years:
        #         if int(row.name) == int(w_y) and row['equity'] > -5:
        #             annual_ret['score'][index] = 2
        #
        #     for o_y in other_years:
        #         if int(row.name) == int(o_y) and row['equity'] < -4:
        #             annual_ret['score'][index] = -1
        #
        #     for b_y in bull_years:
        #         ten_percent = self.Spy_returns_dict[int(b_y)] / 10
        #         eighty_percent = ten_percent * 7
        #
        #         if int(row.name) == int(b_y) and row['equity'] > eighty_percent:
        #             annual_ret['score'][index] = 1.5




        cols = [str(x) for x in final_df.columns]
        cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        indx = [str(x) for x in final_df.index]

        monthly_returns_CSV = final_df.copy()

        hovertext = []
        for i in final_df.index:
            hovertext.append(list())
            for j in final_df.columns:
                try:
                    # extract monthly returns from here
                    monthly_returns_CSV.loc[i][j] = monthly_ret.loc[(i, j)]["equity"]

                    hovertext[-1].append(f'{cols[j - 1]} - {i}<br />'
                                         f'Strat: {monthly_ret.loc[(i, j)]["equity"]}<br />'
                                         f'Difference:{round(monthly_ret.loc[(i, j)]["diff"], 2)}<br />'
                                         f'Utility_yearly_Avg:{round(monthly_ret.loc[(i, j)]["utility_year_Avg"], 2)}<br />')
                except:
                    pass

        '''
        Monthly returns can be written to CSV 
        '''
        # monthly_returns_CSV.to_csv(r'C:\Tharun\Source\Hv & 7 slots\Kill Switch types\type  1 kill switch\phantom csv\Dmonthly_Retruns.csv')

        # fig = make_subplots(rows=1, cols=2,
        #                     shared_yaxes=True,
        #                     horizontal_spacing=0.05,
        #                     column_width=[12 / 13, 1 / 13])

        fig = go.Figure(data=go.Heatmap(z=final_df,
                                        x=cols,
                                        y=indx,
                                        coloraxis='coloraxis',
                                        hoverongaps=False,
                                        hoverinfo='text',
                                        hovertext=hovertext))
        # row=1, col=1)

        # hovertext2 = []
        # for i in annual_ret['diff'].index:
        #     hovertext2.append(f'{i}<br />'
        #                        f'Benchmark: {annual_ret.loc[i]["benchmark"]}<br />'
        #                        f'Strat: {annual_ret.loc[i]["equity"]}<br />'
        #                        f'Difference:{round(annual_ret.loc[i]["diff"], 2)}')
        # fig.add_trace(go.Heatmap(z=annual_ret['diff'],
        #                          x=['Annual'],
        #                          y=annual_ret.index,
        #                          coloraxis='coloraxis',
        #                          hoverongaps=False,
        #                          hoverinfo='text',
        #                          hovertext=hovertext2),
        #               row=1, col=2)

        fig.update_xaxes(side="top")
        # fig.update_layout(coloraxis={'colorscale': 'RdBu'})
        fig.update_layout(title='Monthly Performance against Benchmark',
                          template='plotly_dark')
        if enable_auto_open:
            plot(fig, auto_open=True)
        annual_ret.fillna(0, inplace=True)
        return fig, annual_ret

    def plot_equity_curve(self, data_obj, with_benchmark=False, benchmark='SPY'):
        self.equity_df.fillna(0,inplace=True)



        # Create SubPlot
        fig = make_subplots(rows=3, cols=1,
                            specs=[[{'secondary_y': True}],
                                   [{'secondary_y': False}],
                                   [{'secondary_y': True}]],
                            row_width=[0.15, 0.15, 0.7],
                            shared_xaxes=True,
                            subplot_titles=['Equity', 'Drawdown', 'Utility'],
                            vertical_spacing=0.05)

        # EQUITY CURVE
        fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['equity'].values,
                                 name='Equity', line=dict(color='#086e53')),
                      row=1, col=1, secondary_y=False)
        # DRAWDOWN
        fig.add_trace(go.Scatter(x=self.equity_df.index,
                                 y=self.equity_df['drawdown'].values,
                                 fill='tozeroy',
                                 name='Drawdown'),
                      row=2, col=1)
        try:

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom'].values,
                                         name='Phantom', line=dict(color='khaki')),
                              row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_phantom'].values,
                                     fill='tozeroy',
                                     name='Drawdown_Phantom'),
                          row=2, col=1)
        except:
            print('Phantom Exception')
        try:

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_legacy'].values,
                                     name='phantom_legacy', line=dict(color='blue')),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_phantom_legacy'].values,
                                     fill='tozeroy',
                                     name='Drawdown_Phantom'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_ks'].values,
                                     name='Phantom_ks', line=dict(color='red')),
                          row=1, col=1, secondary_y=False)


            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_phantom_ks'].values,
                                     fill='tozeroy',
                                     name='Drawdown_Phantom_ks'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['utility_ks'].values, name='Utility_ks',
                                     fill='tozeroy'),
                          row=3, col=1, secondary_y=False)



        except:
            print('phantom_legacy Exception')





        # UTILITY
        fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['utility'].values, name='Utility',
                                 fill='tozeroy'),
                      row=3, col=1, secondary_y=False)

        if with_benchmark:
            #  BENCHMARK DATA
            index_closes = data_obj.daily_closes[benchmark]
            index_closes.name = 'close'
            index_df = pd.DataFrame(index_closes)
            index_df['open'] = data_obj.daily_opens[benchmark]
            index_df['high'] = data_obj.daily_highs[benchmark]
            index_df['low'] = data_obj.daily_lows[benchmark]
            index_df['volume'] = data_obj.daily_volumes[benchmark]

            # OHLC candlstick plot
            fig.add_trace(go.Scatter(x=index_df.index, y=index_df['close'], name=benchmark),
                          row=1, col=1, secondary_y=True)
            # fig.add_trace(go.Candlestick(x=index_df.index,
            #                              open=index_df['open'],
            #                              high=index_df['high'],
            #                              low=index_df['low'],
            #                              close=index_df['close'],
            #                              name=benchmark),
            #               row=1, col=1, secondary_y=False)
            fig.update_layout(yaxis2=dict(scaleanchor='y', scaleratio=1000))

            # # BETA & CORRELATION
            # beta_series = self.get_rolling_strategy_beta(data_obj, benchmark=benchmark)
            # fig.add_trace(go.Scatter(x=beta_series.index,
            #                          y=beta_series.values,
            #                          line=dict(color='darkturquoise'),
            #                          name='Beta'),
            #               row=3, col=1)
            #
            # corr_series = self.get_rolling_strat_correlation(data_obj, benchmark=benchmark)
            # fig.add_trace(go.Scatter(x=corr_series.index,
            #                          y=corr_series.values,
            #                          line=dict(color='orange'),
            #                          name='Correlation'),
            #               row=3, col=1)

        try:
            if self.portfolio.in_sample_test:
                for k in data_obj.oos_bounds.keys():
                    # print(k, data_obj.oos_bounds[k][0], data_obj.oos_bounds[k][1])
                    fig.add_vrect(x0=data_obj.oos_bounds[k][0], x1=data_obj.oos_bounds[k][1],
                                  fillcolor="red", opacity=0.25, line_width=0)
        except:
            pass

        fig.update_layout(xaxis_rangeslider_visible=False,
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              orientation='h'),
                          template='plotly_dark')

        plot(fig, auto_open=True)

        return fig


    def plot_equity_curve_Qas(self, data_obj, with_benchmark=False, benchmark='SPY'):
        self.equity_df.fillna(0,inplace=True)



        # Create SubPlot
        fig = make_subplots(rows=3, cols=1,
                            specs=[[{'secondary_y': True}],
                                   [{'secondary_y': False}],
                                   [{'secondary_y': True}]],
                            row_width=[0.15, 0.15, 0.7],
                            shared_xaxes=True,
                            subplot_titles=['Equity', 'Drawdown', 'Utility'],
                            vertical_spacing=0.05)

        # EQUITY CURVE
        fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_killswitch'].values,
                                 name='phantom_kill_switch', line=dict(color='#086e53')),
                      row=1, col=1, secondary_y=False)
        # DRAWDOWN
        fig.add_trace(go.Scatter(x=self.equity_df.index,
                                 y=self.equity_df['drawdown_ks'].values,
                                 fill='tozeroy',
                                 name='Drawdown_ks'),
                      row=2, col=1)
        try:

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['equity'].values,
                                         name='phantom_Equity', line=dict(color='khaki')),
                              row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown'].values,
                                     fill='tozeroy',
                                     name='drawdown_Equity'),
                          row=2, col=1)
        except:
            print('Phantom Exception')
        try:

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_vix'].values,
                                     name='phantom_vix', line=dict(color='blue')),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_vix'].values,
                                     fill='tozeroy',
                                     name='drawdown_vix'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_200'].values,
                                     name='phantom_200', line=dict(color='brown')),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_200'].values,
                                     fill='tozeroy',
                                     name='drawdown_phantom_200'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_50'].values,
                                     name='phantom_50', line=dict(color='red')),
                          row=1, col=1, secondary_y=False)


            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_50'].values,
                                     fill='tozeroy',
                                     name='drawdown_50'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['utility_ks'].values, name='Utility_ks',
                                     fill='tozeroy'),
                          row=3, col=1, secondary_y=False)



        except:
            print('phantom_legacy Exception')





        # UTILITY
        fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['utility'].values, name='Utility',
                                 fill='tozeroy'),
                      row=3, col=1, secondary_y=False)

        if with_benchmark:
            #  BENCHMARK DATA
            index_closes = data_obj.daily_closes[benchmark]
            index_closes.name = 'close'
            index_df = pd.DataFrame(index_closes)
            index_df['open'] = data_obj.daily_opens[benchmark]
            index_df['high'] = data_obj.daily_highs[benchmark]
            index_df['low'] = data_obj.daily_lows[benchmark]
            index_df['volume'] = data_obj.daily_volumes[benchmark]

            # OHLC candlstick plot
            fig.add_trace(go.Scatter(x=index_df.index, y=index_df['close'], name=benchmark),
                          row=1, col=1, secondary_y=True)
            # fig.add_trace(go.Candlestick(x=index_df.index,
            #                              open=index_df['open'],
            #                              high=index_df['high'],
            #                              low=index_df['low'],
            #                              close=index_df['close'],
            #                              name=benchmark),
            #               row=1, col=1, secondary_y=False)
            fig.update_layout(yaxis2=dict(scaleanchor='y', scaleratio=1000))

            # # BETA & CORRELATION
            # beta_series = self.get_rolling_strategy_beta(data_obj, benchmark=benchmark)
            # fig.add_trace(go.Scatter(x=beta_series.index,
            #                          y=beta_series.values,
            #                          line=dict(color='darkturquoise'),
            #                          name='Beta'),
            #               row=3, col=1)
            #
            # corr_series = self.get_rolling_strat_correlation(data_obj, benchmark=benchmark)
            # fig.add_trace(go.Scatter(x=corr_series.index,
            #                          y=corr_series.values,
            #                          line=dict(color='orange'),
            #                          name='Correlation'),
            #               row=3, col=1)

        try:
            if self.portfolio.in_sample_test:
                for k in data_obj.oos_bounds.keys():
                    # print(k, data_obj.oos_bounds[k][0], data_obj.oos_bounds[k][1])
                    fig.add_vrect(x0=data_obj.oos_bounds[k][0], x1=data_obj.oos_bounds[k][1],
                                  fillcolor="red", opacity=0.25, line_width=0)
        except:
            pass

        fig.update_layout(xaxis_rangeslider_visible=False,
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              orientation='h'),
                          template='plotly_dark')

        plot(fig, auto_open=True)

        return fig

    def plot_equity_curve_Qas_1(self, data_obj, with_benchmark=False, benchmark='SPY'):
        self.equity_df.fillna(0,inplace=True)



        # Create SubPlot
        fig = make_subplots(rows=3, cols=1,
                            specs=[[{'secondary_y': True}],
                                   [{'secondary_y': False}],
                                   [{'secondary_y': True}]],
                            row_width=[0.15, 0.15, 0.7],
                            shared_xaxes=True,
                            subplot_titles=['Equity', 'Drawdown', 'Utility'],
                            vertical_spacing=0.05)


        try:
            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['equity'].values,
                                         name='phantom_legacy', line=dict(color='khaki')),
                              row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown'].values,
                                     fill='tozeroy',
                                     name='drawdown_Equity'),
                          row=2, col=1)
            # EQUITY CURVE
            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_killswitch'].values,
                                     name='phantom_kill_switch', line=dict(color='#086e53')),
                          row=1, col=1, secondary_y=False)
            # DRAWDOWN
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_ks'].values,
                                     fill='tozeroy',
                                     name='Drawdown_ks'),
                          row=2, col=1)


        except:
            print('Phantom Exception')
        try:

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_vix'].values,
                                     name='phantom_vix', line=dict(color='blue')),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_vix'].values,
                                     fill='tozeroy',
                                     name='drawdown_vix'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_200'].values,
                                     name='phantom_200', line=dict(color='brown')),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_200'].values,
                                     fill='tozeroy',
                                     name='drawdown_phantom_200'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom_50'].values,
                                     name='phantom_50', line=dict(color='red')),
                          row=1, col=1, secondary_y=False)


            fig.add_trace(go.Scatter(x=self.equity_df.index,
                                     y=self.equity_df['drawdown_50'].values,
                                     fill='tozeroy',
                                     name='drawdown_50'),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['utility_ks'].values, name='Utility_ks',
                                     fill='tozeroy'),
                          row=3, col=1, secondary_y=False)



        except:
            print('phantom_legacy Exception')





        # UTILITY
        fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['utility'].values, name='Utility',
                                 fill='tozeroy'),
                      row=3, col=1, secondary_y=False)

        if with_benchmark:
            #  BENCHMARK DATA
            index_closes = data_obj.daily_closes[benchmark]
            index_closes.name = 'close'
            index_df = pd.DataFrame(index_closes)
            index_df['open'] = data_obj.daily_opens[benchmark]
            index_df['high'] = data_obj.daily_highs[benchmark]
            index_df['low'] = data_obj.daily_lows[benchmark]
            index_df['volume'] = data_obj.daily_volumes[benchmark]

            # OHLC candlstick plot
            fig.add_trace(go.Scatter(x=index_df.index, y=index_df['close'], name=benchmark),
                          row=1, col=1, secondary_y=True)
            # fig.add_trace(go.Candlestick(x=index_df.index,
            #                              open=index_df['open'],
            #                              high=index_df['high'],
            #                              low=index_df['low'],
            #                              close=index_df['close'],
            #                              name=benchmark),
            #               row=1, col=1, secondary_y=False)
            fig.update_layout(yaxis2=dict(scaleanchor='y', scaleratio=1000))

            # # BETA & CORRELATION
            # beta_series = self.get_rolling_strategy_beta(data_obj, benchmark=benchmark)
            # fig.add_trace(go.Scatter(x=beta_series.index,
            #                          y=beta_series.values,
            #                          line=dict(color='darkturquoise'),
            #                          name='Beta'),
            #               row=3, col=1)
            #
            # corr_series = self.get_rolling_strat_correlation(data_obj, benchmark=benchmark)
            # fig.add_trace(go.Scatter(x=corr_series.index,
            #                          y=corr_series.values,
            #                          line=dict(color='orange'),
            #                          name='Correlation'),
            #               row=3, col=1)

        try:
            if self.portfolio.in_sample_test:
                for k in data_obj.oos_bounds.keys():
                    # print(k, data_obj.oos_bounds[k][0], data_obj.oos_bounds[k][1])
                    fig.add_vrect(x0=data_obj.oos_bounds[k][0], x1=data_obj.oos_bounds[k][1],
                                  fillcolor="red", opacity=0.25, line_width=0)
        except:
            pass

        fig.update_layout(xaxis_rangeslider_visible=False,
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              orientation='h'),
                          template='plotly_dark')

        plot(fig, auto_open=True)

        return fig

    def plot_vol_break(self):
        fig = make_subplots(rows=4, cols=1,
                            specs=[[{'secondary_y': True}],
                                    [{'secondary_y': False}],
                                    [{'secondary_y': False}],
                                   [{'secondary_y': False}]],
                            row_width=[0.12, 0.12,0.12,0.12],
                            shared_xaxes=True,
                            subplot_titles=[ 'Vix','Equities','Spy','Spy_volatility'],
                            vertical_spacing=0.05)
        fig.add_trace(
            go.Scatter(x=self.portfolio.price_data.closes_vix.index, y=self.portfolio.price_data.closes_vix.values,
                       name='Vix close price', line=dict(color='red')),
            row=1, col=1, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=self.portfolio.price_data.moving_avg_vix.index,
                       y=self.portfolio.price_data.moving_avg_vix.values,
                       name='Vix moving average', line=dict(color='blue')),
            row=1, col=1, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=self.portfolio.price_data.moving_avg_vix_v2.index,
                       y=self.portfolio.price_data.moving_avg_vix_v2.values,
                       name='Vix moving average_2', line=dict(color='#086e53')),
            row=1, col=1, secondary_y=False)

        fig.add_trace(
            go.Scatter(x=self.portfolio.vol_triggeres_indexes,
                       y=self.portfolio.price_data.closes_vix.loc[self.portfolio.vol_triggeres_indexes].values,
                       name='Trigger points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
            row=1, col=1, secondary_y=False)

        fig.add_trace(
            go.Scatter(x=self.portfolio.vol_back_live_indexes,
                       y=self.portfolio.price_data.closes_vix.loc[self.portfolio.vol_back_live_indexes].values,
                       name='back live points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
            row=1, col=1, secondary_y=False)

        # fig.add_trace(
        #     go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom'].values,
        #                name='Phantom', line=dict(color='blue')),
        #     row=1, col=1, secondary_y=False)


        # fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom'].values,
        #                          name='Phantom', line=dict(color='khaki')),
        #               row=1, col=1, secondary_y=False)

        fig.add_trace(
            go.Scatter(x=self.equity_df.index, y=self.equity_df['phantom'].values,
                       name='Phantom', line=dict(color='blue')),
            row=2, col=1, secondary_y=False)
        # EQUITY CURVE
        fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['equity'].values,
                                 name='Equity', line=dict(color='#086e53')),
                      row=2, col=1, secondary_y=False)

        # Spy CURVE
        fig.add_trace(go.Scatter(x=self.portfolio.price_data.spy_prices.index, y=self.portfolio.price_data.spy_prices['Close'].values,
                                 name='Spy', line=dict(color='Green')),
                      row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=self.portfolio.price_data.moving_avg_spy.index, y=self.portfolio.price_data.moving_avg_spy.values,
                                 name='Spy', line=dict(color='red')),
                      row=3, col=1, secondary_y=False)

        # fig.add_trace(
        #     go.Scatter(x=self.portfolio.vol_back_live_indexes,
        #                y=self.portfolio.price_data.spy_prices['Close'].loc[self.portfolio.vol_back_live_indexes].values,
        #                name='back live points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
        #     row=3, col=1, secondary_y=False)
        # fig.add_trace(
        #     go.Scatter(x=self.portfolio.vol_triggeres_indexes,
        #                y=self.portfolio.price_data.spy_prices['Close'].loc[self.portfolio.vol_triggeres_indexes].values,
        #                name='Trigger points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
        #     row=3, col=1, secondary_y=False)

        # Spy-volatitlity CURVE
        fig.add_trace(go.Scatter(x=self.portfolio.test.index, y=self.portfolio.test.values,
                                 name='Spy_vol', line=dict(color='Green')),
                      row=4, col=1, secondary_y=False)
        # fig.add_trace(
        #     go.Scatter(x=self.portfolio.vol_triggeres_indexes,
        #                y=self.portfolio.test.loc[self.portfolio.vol_triggeres_indexes].values,
        #                name='Trigger points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
        #     row=4, col=1, secondary_y=False)
        #
        # fig.add_trace(
        #     go.Scatter(x=self.portfolio.vol_back_live_indexes,
        #                y=self.portfolio.test.loc[self.portfolio.vol_back_live_indexes].values,
        #                name='back live points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
        #     row=4, col=1, secondary_y=False)



        # fig.add_trace(
        #     go.Scatter(x=self.portfolio.price_data.RSI_12.index, y=self.portfolio.price_data.RSI_12.values,
        #                name='RSI Spy ', line=dict(color='DarkSlateGrey')),
        #     row=3, col=1, secondary_y=False)
        #
        # fig.add_trace(
        #     go.Scatter(x=self.portfolio.vol_back_live_indexes,
        #                y=self.portfolio.price_data.RSI_12.loc[self.portfolio.vol_back_live_indexes].values,
        #                name='back live points on RSI', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
        #     row=3, col=1, secondary_y=False)
        return fig

    def plot_vol_break_nif(self):
        fig = make_subplots(rows=4, cols=1,
                            specs=[[{'secondary_y': True}],
                                    [{'secondary_y': False}],
                                    [{'secondary_y': False}],
                                   [{'secondary_y': False}]],
                            row_width=[0.12, 0.12,0.12,0.12],
                            shared_xaxes=True,
                            subplot_titles=[ 'Vix','Equities','Spy','Spy_volatility'],
                            vertical_spacing=0.05)
        try:
            fig.add_trace(
                go.Scatter(x=self.portfolio.price_data.nifty_prices['Close'].index,
                           y=self.portfolio.price_data.nifty_prices['Close'].values,
                           name='Nifty_price', line=dict(color='green')),
                row=1, col=1, secondary_y=False)
            fig.add_trace(
                go.Scatter(x=self.portfolio.price_data.mov_avg_200_nif.index,
                           y=self.portfolio.price_data.mov_avg_200_nif.values,
                           name='Nifty moving average', line=dict(color='blue')),
                row=1, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['equity'].values,
                                     name='Equity', line=dict(color='#086e53')),
                          row=2, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(x=self.portfolio.price_data.nif_macd['long_ema'].index,
                                     y=self.portfolio.price_data.nif_macd['long_ema'].values,
                                     name='nift_vol', line=dict(color='red')),
                          row=3, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=self.portfolio.price_data.nif_macd['short_ema'].index,
                                     y=self.portfolio.price_data.nif_macd['short_ema'].values,
                                     name='nif_ma', line=dict(color='green')),
                          row=3, col=1, secondary_y=False)
        except Exception as e:
            print('Exe')



        # fig.add_trace(
        #     go.Scatter(x=self.portfolio.price_data.mov_avg_volity.index,
        #                y=self.portfolio.price_data.mov_avg_volity.values,
        #                name='Vix moving average', line=dict(color='red')),
        #     row=2, col=1, secondary_y=False)

        return fig

    # def plot_vix_curve(self,func_type='bull'):
    #     # bf = Bull_Market_Rally_Fall(bull_path=r'C:\Users\tharu\Desktop\Bull and Bear Rally Fall\BULL_MARKET_SEPERATED.csv'
    #     #                             , bear_path=r'C:\Users\tharu\Desktop\Bull and Bear Rally Fall\BEAR_MARKET_SEPERATED.csv')
    #
    #     if func_type == '':
    #         print(False)
    #     else:
    #         fig = make_subplots(rows=2, cols=3)
    #
    #         fig.add_trace(
    #             go.Scatter(x=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][1] : bf.bear_market_df['DATE_END'][1]].index,
    #                        y=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][1] : bf.bear_market_df['DATE_END'][1]].values,
    #                        name='equity'),
    #             row=1, col=1
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(x=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][3] : bf.bear_market_df['DATE_END'][3]].index,
    #                        y=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][3] : bf.bear_market_df['DATE_END'][3]].values,
    #                        name='equity'),
    #             row=1, col=2
    #         )
    #
    #
    #
    #
    #         fig.add_trace(
    #             go.Scatter(x=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][7] : bf.bear_market_df['DATE_END'][7]].index,
    #                        y=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][7] : bf.bear_market_df['DATE_END'][7]].values,
    #                        name='equity'),
    #             row=1, col=3
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(x=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][9] : bf.bear_market_df['DATE_END'][9]].index,
    #                        y=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][9] : bf.bear_market_df['DATE_END'][9]].values,
    #                        name='equity'),
    #             row=2, col=1
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][11]: bf.bear_market_df['DATE_END'][11]].index,
    #                 y=self.portfolio.price_data.closes_vix.loc[
    #                   bf.bear_market_df['DATE_START'][11]: bf.bear_market_df['DATE_END'][11]].values,
    #                 name='equity'),
    #             row=2, col=2
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.portfolio.price_data.closes_vix.loc[bf.bear_market_df['DATE_START'][15]: bf.bear_market_df['DATE_END'][15]].index,
    #                 y=self.portfolio.price_data.closes_vix.loc[
    #                   bf.bear_market_df['DATE_START'][15]: bf.bear_market_df['DATE_END'][15]].values,
    #                 name='equity'),
    #             row=2, col=3
    #         )
    #
    #         fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    #         return fig

    # def plot_weak_subplots(self, func_type='bull'):
    #     # bf = Bull_Market_Rally_Fall(
    #     #     bull_path=r'C:\Users\tharu\Desktop\Bull and Bear Rally Fall\BULL_MARKET_SEPERATED.csv'
    #     #     , bear_path=r'C:\Users\tharu\Desktop\Bull and Bear Rally Fall\BEAR_MARKET_SEPERATED.csv')
    #
    #     if func_type == '':
    #         print(False)
    #     else:
    #         fig = make_subplots(rows=2, cols=3)
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.equity_df.loc[bf.bear_market_df['DATE_START'][1]: bf.bear_market_df['DATE_END'][1]].index,
    #                 y=self.equity_df['equity'].loc[
    #                   bf.bear_market_df['DATE_START'][1]: bf.bear_market_df['DATE_END'][1]].values,
    #                 name='equity'),
    #             row=1, col=1
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.equity_df.loc[bf.bear_market_df['DATE_START'][3]: bf.bear_market_df['DATE_END'][3]].index,
    #                 y=self.equity_df['equity'].loc[
    #                   bf.bear_market_df['DATE_START'][3]: bf.bear_market_df['DATE_END'][3]].values,
    #                 name='equity'),
    #             row=1, col=2
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.equity_df.loc[bf.bear_market_df['DATE_START'][7]: bf.bear_market_df['DATE_END'][7]].index,
    #                 y=self.equity_df['equity'].loc[
    #                   bf.bear_market_df['DATE_START'][7]: bf.bear_market_df['DATE_END'][7]].values,
    #                 name='equity'),
    #             row=1, col=3
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.equity_df.loc[bf.bear_market_df['DATE_START'][9]: bf.bear_market_df['DATE_END'][9]].index,
    #                 y=self.equity_df['equity'].loc[
    #                   bf.bear_market_df['DATE_START'][9]: bf.bear_market_df['DATE_END'][9]].values,
    #                 name='equity'),
    #             row=2, col=1
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.equity_df.loc[bf.bear_market_df['DATE_START'][11]: bf.bear_market_df['DATE_END'][11]].index,
    #                 y=self.equity_df['equity'].loc[
    #                   bf.bear_market_df['DATE_START'][11]: bf.bear_market_df['DATE_END'][11]].values,
    #                 name='equity'),
    #             row=2, col=2
    #         )
    #
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.equity_df.loc[bf.bear_market_df['DATE_START'][15]: bf.bear_market_df['DATE_END'][15]].index,
    #                 y=self.equity_df['equity'].loc[
    #                   bf.bear_market_df['DATE_START'][15]: bf.bear_market_df['DATE_END'][15]].values,
    #                 name='equity'),
    #             row=2, col=3
    #         )
    #
    #         fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    #         return fig

    # def plot_vix_curve_using_close(self):
    #     fig = make_subplots(rows=4, cols=1,
    #                         specs=[[{'secondary_y': True}],
    #                                 [{'secondary_y': False}],
    #                                 [{'secondary_y': False}],
    #                                [{'secondary_y': False}]],
    #                         row_width=[0.12, 0.12,0.12,0.12],
    #                         shared_xaxes=True,
    #                         subplot_titles=[ 'Vix','Equities','Spy','Spy_volatility'],
    #                         vertical_spacing=0.05)
    #     fig.add_trace(
    #         go.Scatter(x=self.portfolio.price_data.closes_vix.index, y=self.portfolio.price_data.closes_vix.values,
    #                    name='Vix close price', line=dict(color='red')),
    #         row=1, col=1, secondary_y=False)
    #     fig.add_trace(
    #         go.Scatter(x=self.portfolio.price_data.moving_avg_vix.index,
    #                    y=self.portfolio.price_data.moving_avg_vix.values,
    #                    name='Vix moving average', line=dict(color='blue')),
    #         row=1, col=1, secondary_y=False)
    #     fig.add_trace(
    #         go.Scatter(x=self.portfolio.price_data.moving_avg_vix_ks.index,
    #                    y=self.portfolio.price_data.moving_avg_vix_ks.values,
    #                    name='Vix moving_avg_vix_ks average_2', line=dict(color='#086e53')),
    #         row=1, col=1, secondary_y=False)
    #
    #     fig.add_trace(
    #         go.Scatter(x=self.portfolio.vol_triggeres_indexes,
    #                    y=self.portfolio.price_data.closes_vix.loc[self.portfolio.vol_triggeres_indexes].values,
    #                    name='Trigger points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
    #         row=1, col=1, secondary_y=False)
    #
    #     fig.add_trace(
    #         go.Scatter(x=self.portfolio.vol_back_live_indexes,
    #                    y=self.portfolio.price_data.closes_vix.loc[self.portfolio.vol_back_live_indexes].values,
    #                    name='back live points', mode="markers", marker_size=10, fillcolor='DarkSlateGrey'),
    #         row=1, col=1, secondary_y=False)
    #
    #     fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['equity'].values,
    #                              name='Equity', line=dict(color='#086e53')),
    #                   row=2, col=1, secondary_y=False)
    #
    #     fig.add_trace(go.Scatter(x=self.portfolio.price_data.spy_prices.index, y=self.portfolio.price_data.spy_prices['Close'].values,
    #                              name='Spy', line=dict(color='Green')),
    #                   row=3, col=1, secondary_y=False)
    #     fig.add_trace(go.Scatter(x=self.portfolio.price_data.moving_avg_spy_200.index,
    #                              y=self.portfolio.price_data.moving_avg_spy_200.values,
    #                              name='Spy_mv', line=dict(color='blue')),
    #                   row=3, col=1, secondary_y=False)
    #
    #     fig.add_trace(go.Scatter(x=self.portfolio.price_data.spy_rsi.index,
    #                              y=self.portfolio.price_data.spy_rsi.values,
    #                              name='Spy_rsi', line=dict(color='blue')),
    #                   row=4, col=1, secondary_y=False)
    #
    #     fig.add_trace(go.Scatter(x=self.portfolio.price_data.rsi_vix.index,
    #                              y=self.portfolio.price_data.rsi_vix.values,
    #                              name='Vix_rsi', line=dict(color='green')),
    #                   row=4, col=1, secondary_y=False)
    #
    #     return fig

    def get_break_down_areas(self):
        fig = make_subplots(rows=2, cols=2)

        # fig.add_trace(
        #     go.Scatter(x=self.equity_df.iloc[192:234].index, y=self.equity_df['phantom'].iloc[192:234].values,
        #                name='Phantom', line=dict(color='blue')),
        #     row=2, col=1, secondary_y=False)

        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[192:234].index, y=self.equity_df['phantom'].iloc[192:234].values,name='phantom'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[192:234].index, y=self.equity_df['equity'].iloc[192:234].values,name='equity'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[341:572].index, y=self.equity_df['phantom'].iloc[341:572].values,name='phantom'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[341:572].index, y=self.equity_df['equity'].iloc[341:572].values,name='equity'),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[2791:3000].index, y=self.equity_df['phantom'].iloc[2791:3000].values,name='phantom'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[2791:3000].index, y=self.equity_df['equity'].iloc[2791:3000].values,name='equity'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[5394:5494].index, y=self.equity_df['phantom'].iloc[5394:5494].values,name='phantom'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.equity_df.iloc[5394:5494].index, y=self.equity_df['equity'].iloc[5394:5494].values,name='equity'),
            row=2, col=2
        )

        fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
        return fig


    def plot_ta_chart(self, data_obj, ticker_key='SPY', timeframe='daily', ma=False, ma_type='simp', equity=False):
        if timeframe == 'weekly':
            data_obj.get_weekly_data(data_obj)
            index_closes = data_obj.weekly_closes[ticker_key]
        elif timeframe == 'daily':
            index_closes = data_obj.daily_closes[ticker_key]
            index_closes.name = 'close'
            index_df = pd.DataFrame(index_closes)
            index_df['open'] = data_obj.daily_opens[ticker_key]
            index_df['high'] = data_obj.daily_highs[ticker_key]
            index_df['low'] = data_obj.daily_lows[ticker_key]
            index_df['volume'] = data_obj.daily_volumes[ticker_key]

        # Create SubPlot
        fig = make_subplots(rows=2, cols=1,
                            specs=[[{'secondary_y': True}],
                                   [{'secondary_y': False}]],
                            row_width=[0.3, 0.7],
                            shared_xaxes=True,
                            subplot_titles=('Equity', 'Drawdown'))
        # OHLC candlstick plot
        fig.add_trace(go.Candlestick(x=index_df.index,
                                     open=index_df['open'],
                                     high=index_df['high'],
                                     low=index_df['low'],
                                     close=index_df['close'],
                                     name=ticker_key),
                      row=1, col=1, secondary_y=False)

        fig.update_layout(title=f'Strategy performance plotted against {ticker_key}',
                          xaxis_rangeslider_visible=False,
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              orientation='h'))
        # template='plotly_dark')

        # # HISTORICAL VOLATILITY
        # vol_lookback = 52
        # index_df[f'hist_vol_{vol_lookback}'] = HistoricVolatility(index_df['close'], n=vol_lookback)
        # add rolling historic vol to subplot
        # fig.add_trace(go.Scatter(x=index_df.index, y=index_df['hist_vol_52'],
        #                          name=f'rolling_hist_vol_{vol_lookback}'),
        #               row=2, col=1)

        # # add VOLUME
        # fig.add_trace(go.Bar(x=index_df.index, y=index_df['volume'],
        #                      name='volume',
        #                      marker=dict(color='white')),
        #               row=3, col=1)

        # plot our EQUTIY CURVE over our Stock data
        if equity:
            fig.add_trace(go.Scatter(x=self.equity_curve.index, y=self.equity_curve.values,
                                     name=f'{ticker_key} Equity Curve', line=dict(color='fuchsia')),
                          row=1, col=1, secondary_y=True)
            fig.update_layout(yaxis2=dict(scaleanchor='y', scaleratio=0.001))

        # Add in MOVING AVERAGES to the stock data
        # ma_lookbacks = [10, 20, 50, 100, 200]
        if ma:
            # Add in moving averages
            ma_lookbacks = [10, 20, 50, 100, 200]
            ma_colors = ['yellow', 'lime', 'orange', 'darkturquoise', 'mediumslateblue']
            for i in ma_lookbacks:
                if ma == 'simp':
                    index_df[f'SMA_{i}'] = index_df['close'].rolling(window=i).mean()
                elif ma == 'exp':
                    pass

            for i in ma_lookbacks:
                if ma == 'simp':
                    index_df[f'SMA_{i}'] = index_df['close'].rolling(window=i).mean()
                elif ma == 'exp':
                    pass
            # add moving averages to plot
            for i in range(len(ma_lookbacks)):
                if ma == 'simp':
                    fig.add_trace(go.Scatter(x=index_df.index, y=index_df[f'SMA_{ma_lookbacks[i]}'],
                                             name=f'SMA_{ma_lookbacks[i]}',
                                             line=dict(color=ma_colors[i],
                                                       dash='dot')),
                                  row=1, col=1)
                elif ma == 'exp':
                    fig.add_trace(go.Scatter(x=index_df.index, y=index_df[f'EMA_{ma_lookbacks[i]}'],
                                             name=f'EMA_{ma_lookbacks[i]}',
                                             line=dict(color=ma_colors[i],
                                                       dash='dot')),
                                  row=1, col=1)

            # # add in MACD to subplot
            # macd_df = MACD(index_df['close'])
            # # plot our MACD line = long_ema - short_ema and then plot the signal line as well histogram of the difference
            # fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df['MACD_line'],
            #                          name='MACD line',
            #                          line=dict(color='blue')),
            #               row=2, col=1)
            # fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df['signal_line'],
            #                          name='signal line',
            #                          line=dict(color='orange')),
            #               row=2, col=1)
            # colours = {'True': 'green',
            #            'False': 'red'}
            # fig.add_trace(go.Bar(x=macd_df.index, y=macd_df['MACD_diff'], marker={'color': 'green'}, name='MACD diff'),
            #               row=2, col=1)

        # fig.show()

        return fig

    def get_trade_val_dict(self):
        val_track = self.value_track.copy(True)
        val_track.drop('Total', axis=1, inplace=True)
        # drop the columns of nans
        val_track.dropna(axis=1, how='all', inplace=True)
        val_track.replace(0, np.nan, inplace=True)

        # breakout to get each trade
        trade_val_dict = {}
        for col in val_track.columns:
            stock_series = val_track[col]
            sparse_ts = val_track[col].astype(pd.SparseDtype('float'))
            block_locs = zip(sparse_ts.values.sp_index.to_block_index().blocs,
                             sparse_ts.values.sp_index.to_block_index().blengths)
            trade_num = 1
            for start, length in block_locs:
                trade_val_dict[f'{col}{trade_num}'] = stock_series.iloc[start:(start + length)]
                trade_num += 1

        return trade_val_dict

    def plot_excursion_graphs(self):
        trade_equity_dict = self.get_trade_val_dict()
        trade_list = self.trade_list.copy(True)
        # mask = trade_list['symbol'].duplicated(keep=False)
        trade_list['symbol'] = np.where(trade_list['symbol'].duplicated(keep=False),
                                        trade_list['symbol'] + trade_list.groupby('symbol').cumcount().add(1).astype(
                                            str),
                                        trade_list['symbol'] + '1')
        # trade_list.loc[mask, 'symbol'] += trade_list.groupby('symbol').cumcount().add(1).astype(str)

        trade_mapping = dict(zip(trade_list['symbol'], trade_list.index))
        color_map = {'winner': 'lime',
                     'loser': 'red'}
        symbol_map = {'winner': 'triangle-up',
                      'loser': 'triangle-down'}
        # RUN UP PNL PLOT
        trade_pnl_fig = make_subplots(rows=1, cols=1)
        # runup_pnl_fig = make_subplots(rows=1, cols=1)
        # dd_pnl_fig = make_subplots(rows=1, cols=1)
        mfe_fig = make_subplots(rows=1, cols=1)
        mae_fig = make_subplots(rows=1, cols=1)
        for k, v in trade_equity_dict.items():
            x_ = trade_mapping[k]
            # y_ = (v - v[0]).round(2)   # profit in absolute terms
            y_ = ((v / v[0] - 1) * 100).round(2)  # profit in relative terms
            if y_[-1] < 0:
                state = 'loser'
            else:
                state = 'winner'
            max_val = (y_.cummax()).max()
            min_val = (y_.cummin()).min()

            trade_pnl_fig.add_trace(go.Scatter(x=[x_, x_, x_], y=[max_val, y_[-1], min_val],
                                               line=dict(color=color_map[state]),
                                               name=k,
                                               hovertemplate=f'{k}\n |PnL:{y_[-1]}|\n|MaxValue:{max_val}|\n|MinValue={min_val}|'),
                                    row=1, col=1)
            mfe_fig.add_trace(go.Scatter(mode='markers',
                                         x=[min_val], y=[y_[-1]],
                                         marker_symbol=symbol_map[state],
                                         marker_color=color_map[state],
                                         hovertemplate=f'|PnL:{y_[-1]}|\n|MinValue={min_val}|',
                                         name=k),
                              row=1, col=1)
            mae_fig.add_trace(go.Scatter(mode='markers',
                                         x=[max_val], y=[y_[-1]],
                                         marker_symbol=symbol_map[state],
                                         marker_color=color_map[state],
                                         hovertemplate=f'|PnL:{y_[-1]}|\n|MaxValue={max_val}|',
                                         name=k),
                              row=1, col=1)

            # runup_pnl_fig.add_trace(go.Scatter(x=[x_, x_], y=[max_val, y_[-1]],
            #                                    line=dict(color=color_map[state]),
            #                                    name=k),
            #                         row=1, col=1)
            # dd_pnl_fig.add_trace(go.Scatter(x=[x_, x_], y=[min_val, y_[-1]],
            #                                 line=dict(color=color_map[state]),
            #                                 name=k),
            #                      row=1, col=1)
        # runup_pnl_fig.update_layout(title='Run-up P&L Plot',
        #                             xaxis_rangeslider_visible=False,
        #                             template='plotly_dark')
        # dd_pnl_fig.update_layout(title='Drawdown P&L Plot',
        #                          xaxis_rangeslider_visible=False,
        #                          template='plotly_dark')
        trade_pnl_fig.update_layout(title=f'{self.name.upper()} Drawdown P&L Plot',
                                    xaxis_rangeslider_visible=False,
                                    template='plotly_dark')
        mfe_fig.update_layout(title=f'{self.name.upper()} Max Favourable Excursion Plot',
                              xaxis_rangeslider_visible=False,
                              template='plotly_dark')
        mae_fig.update_layout(title=f'{self.name.upper()} Max Adverse Excursion Plot',
                              xaxis_rangeslider_visible=False,
                              template='plotly_dark')
        # runup_pnl_fig.show()
        # dd_pnl_fig.show()
        trade_pnl_fig.show()
        mfe_fig.show()
        mae_fig.show()
        return

    def plot_excursion_graphs_v2(self, trade_dict=None):
        if not trade_dict:
            return print('Please pass trade equity dictonary!')

        trade_equity_dict = trade_dict

        color_map = {'winner': 'lime',
                     'loser': 'red'}
        symbol_map = {'winner': 'triangle-up',
                      'loser': 'triangle-down'}
        # RUN UP PNL PLOT
        trade_pnl_fig = make_subplots(rows=1, cols=1)
        # runup_pnl_fig = make_subplots(rows=1, cols=1)
        # dd_pnl_fig = make_subplots(rows=1, cols=1)
        mfe_fig = make_subplots(rows=1, cols=1)
        mae_fig = make_subplots(rows=1, cols=1)
        count = 1
        for k, v in trade_equity_dict.items():
            x_ = count
            open_value = self.portfolio.trade_log[k]['open_value']
            # y_ = (v - v[0]).round(2)   # profit in absolute terms
            y_ = ((list(v.values()) / open_value - 1) * 100).round(2)  # profit in relative terms
            if self.portfolio.trade_log[k]['profit'] < 0:
                state = 'loser'
            else:
                state = 'winner'
            max_val = y_.max()
            min_val = y_.min()
            final_prof = self.portfolio.trade_log[k]['profit%'] * 100
            trade_pnl_fig.add_trace(go.Scatter(x=[x_, x_, x_], y=[max_val, final_prof, min_val],
                                               line=dict(color=color_map[state]),
                                               name=k,
                                               hovertemplate=f'{k}\n |PnL:{final_prof}|\n|MaxValue:{max_val}|\n|MinValue={min_val}|'),
                                    row=1, col=1)
            mfe_fig.add_trace(go.Scatter(mode='markers',
                                         x=[min_val], y=[final_prof],
                                         marker_symbol=symbol_map[state],
                                         marker_color=color_map[state],
                                         hovertemplate=f'|PnL:{final_prof}|\n|MinValue={min_val}|',
                                         name=k),
                              row=1, col=1)
            mae_fig.add_trace(go.Scatter(mode='markers',
                                         x=[max_val], y=[final_prof],
                                         marker_symbol=symbol_map[state],
                                         marker_color=color_map[state],
                                         hovertemplate=f'|PnL:{final_prof}|\n|MaxValue={max_val}|',
                                         name=k),
                              row=1, col=1)
            count += 1

            # runup_pnl_fig.add_trace(go.Scatter(x=[x_, x_], y=[max_val, y_[-1]],
            #                                    line=dict(color=color_map[state]),
            #                                    name=k),
            #                         row=1, col=1)
            # dd_pnl_fig.add_trace(go.Scatter(x=[x_, x_], y=[min_val, y_[-1]],
            #                                 line=dict(color=color_map[state]),
            #                                 name=k),
            #                      row=1, col=1)
        # runup_pnl_fig.update_layout(title='Run-up P&L Plot',
        #                             xaxis_rangeslider_visible=False,
        #                             template='plotly_dark')
        # dd_pnl_fig.update_layout(title='Drawdown P&L Plot',
        #                          xaxis_rangeslider_visible=False,
        #                          template='plotly_dark')
        trade_pnl_fig.update_layout(title=f'{self.name.upper()} Drawdown P&L Plot',
                                    xaxis_rangeslider_visible=False,
                                    template='plotly_dark')
        mfe_fig.update_layout(title=f'{self.name.upper()} Max Favourable Excursion Plot',
                              xaxis_rangeslider_visible=False,
                              template='plotly_dark')
        mae_fig.update_layout(title=f'{self.name.upper()} Max Adverse Excursion Plot',
                              xaxis_rangeslider_visible=False,
                              template='plotly_dark')
        # runup_pnl_fig.show()
        # dd_pnl_fig.show()
        plot(trade_pnl_fig)
        plot(mfe_fig)
        plot(mae_fig)
        return trade_pnl_fig, mfe_fig, mae_fig

    def deeper_trade_analysis(self, plot=True, pct_list=[], year_split=False):
        """

        Parameters
        ----------
        plot = Boolean : you can plot how trades evolve over time
        pct_list = List : list containing percent levels that trades hit/exceeded.
                        If postive i will be greater than these levels, if negative its less that.

        Returns It doesnt return but prints out a summary in the console
        -------
        Logic is fuzzy around zero so need to update but can hack for now. Hardcode it in whether you want to see if
        the trades went above below zero using <>=
        """
        trade_equity_dict_all = self.get_trade_val_dict()
        trade_list = self.trade_list.copy(True)
        trade_list['symbol'] = np.where(trade_list['symbol'].duplicated(keep=False),
                                        trade_list['symbol'] + trade_list.groupby('symbol').cumcount().add(1).astype(
                                            str),
                                        trade_list['symbol'] + '1')

        trade_year_mapping = dict(zip(trade_list['symbol'], trade_list['open_date'].str[:4]))
        if year_split:
            trade_years = trade_list['open_date'].str[:4].unique()
        else:
            trade_years = ['all']

        for x_pct in pct_list:
            for j in trade_years:
                if j is 'all':  # empty dicts for all trades that meet the requirements
                    trade_equity_dict = trade_equity_dict_all
                else:
                    trade_equity_dict = {k: v for k, v in trade_equity_dict_all.items() if trade_year_mapping[k] == j}
                yes_path = {}
                yes_time_to_hit = {}
                yes_time_after_hit = {}
                # loop through all our trades and filter out those that hit a certain level
                for k, v in trade_equity_dict.items():
                    trade_path = (v / v[0] - 1) * 100  # get value evolution in percent
                    if x_pct <= 0:
                        if (trade_path < x_pct).any():  # if the trade went below the level
                            yes_path[k] = trade_path
                            # Save down the time it took to breach that level and the time the trade stay alive after that level
                            yes_time_to_hit[k] = len(trade_path[:trade_path[trade_path < x_pct].index[0]])
                            yes_time_after_hit[k] = len(trade_path[trade_path[trade_path < x_pct].index[0]:])
                    else:
                        if (trade_path > x_pct).any():  # if the trade went below the level
                            yes_path[k] = trade_path
                            # Save down the time it took to breach that level and the time the trade stay alive after that level
                            yes_time_to_hit[k] = len(trade_path[:trade_path[trade_path < x_pct].index[0]])
                            yes_time_after_hit[k] = len(trade_path[trade_path[trade_path < x_pct].index[0]:])

                # Anaylsis on yes group
                # Get values for all of the trades
                yes_all_profit = [i[-1] for i in yes_path.values()]
                # time to hit
                yes_all_tth = [i for i in yes_time_to_hit.values()]
                # time after hit
                yes_all_tah = [i for i in yes_time_after_hit.values()]

                # Split them out by winners and losers
                yes_winners_key = [k for k in yes_path.keys() if yes_path[k][-1] > 0]
                yes_losers_key = [k for k in yes_path.keys() if yes_path[k][-1] < 0]

                yes_winners_profit = [yes_path[k][-1] for k in yes_winners_key]
                yes_losers_profit = [yes_path[k][-1] for k in yes_losers_key]

                yes_winners_tth = [yes_time_to_hit[k] for k in yes_winners_key]
                yes_losers_tth = [yes_time_to_hit[k] for k in yes_losers_key]

                yes_winners_tah = [yes_time_after_hit[k] for k in yes_winners_key]
                yes_losers_tah = [yes_time_after_hit[k] for k in yes_losers_key]

                yes_num_winners = len(yes_winners_profit)
                yes_num_losers = len(yes_losers_profit)

                if not yes_winners_key:
                    yes_winners_tah = [np.nan]
                    yes_winners_tth = [np.nan]
                    yes_winners_profit = [np.nan]
                if not yes_losers_key:
                    yes_losers_tah = [np.nan]
                    yes_losers_tth = [np.nan]
                    yes_losers_profit = [np.nan]

                # yes_winners_profit1 = [i for i in yes_all_profit if i > 0]
                # yes_winners_profit = [i[-1] for i in list(filter(lambda x: x[-1] > 0, yes_path.values()))]
                # yes_losers_profit = [i[-1] for i in list(filter(lambda x: x[-1] < 0, yes_path.values()))]
                # yes_winners_tth = [yes_time_to_hit[i.name] for i in list(filter(lambda x: x[-1] > 0, yes_path.values()))]

                # Print Total trades with yes and no groups
                if year_split:
                    print(f'\n\n------------------------------------|'
                          f'\n                                    |'
                          f'\n               {j.upper()}                 |'
                          f'\n                                    |'
                          f'\n------------------------------------|\n')

                print(f'\n\n------------------------------------|'
                      f'\nTotal Trades        = {len(trade_equity_dict)}           |'
                      f'\nTrades hit {x_pct}%      = {len(yes_path)} ({round((len(yes_path) / len(trade_equity_dict)) * 100, 2)}%)  |'
                      f'\nTrades not hit {x_pct}%  = {len(trade_equity_dict) - len(yes_path)} '
                      f'({round(((len(trade_equity_dict) - len(yes_path)) / len(trade_equity_dict)) * 100, 2)}%)  |'
                      f'\n------------------------------------|\n')

                print(f'------------------------------------|'
                      f'\nStatistics for Trades *THAT* hit {x_pct}%|'
                      f'\n------------------------------------|'
                      f'\nTotal Trades        = {len(yes_path)}           |'
                      f'\nTotal Winners       = {yes_num_winners} '
                      f' ({round(yes_num_winners / len(yes_path) * 100, 2)}%)  |'
                      f'\nTotal Losers        = {yes_num_losers} '
                      f'({round(yes_num_losers / len(yes_path) * 100, 2)}%)  |'
                      f'\n------------------------------------|'
                      f'\nProfit % Distrubtion \n'
                      f'\n  All Trades      |      Winners       |       Losers        |'
                      f'\nMin  =  {round(np.quantile(yes_all_profit, 0), 2)}%   |   Min  =  {round(np.quantile(yes_winners_profit, 0), 2)}%    |   Min  =  {round(np.quantile(yes_losers_profit, 0), 2)}%   |'
                      f'\n1Q   =  {round(np.quantile(yes_all_profit, 0.25), 2)}%    |   1Q   =  {round(np.quantile(yes_winners_profit, 0.25), 2)}%    |   1Q   =  {round(np.quantile(yes_losers_profit, 0.25), 2)}%    |'
                      f'\nMed  =  {round(np.quantile(yes_all_profit, 0.5), 2)}%    |   Med  =  {round(np.quantile(yes_winners_profit, 0.5), 2)}%   |   Med  =  {round(np.quantile(yes_losers_profit, 0.5), 2)}%    |'
                      f'\n3Q   =  {round(np.quantile(yes_all_profit, 0.75), 2)}%    |   3Q   =  {round(np.quantile(yes_winners_profit, 0.75), 2)}%   |   3Q   =  {round(np.quantile(yes_losers_profit, 0.75), 2)}%    |'
                      f'\nMax  =  {round(np.quantile(yes_all_profit, 1), 2)}%    |   Max  =  {round(np.quantile(yes_winners_profit, 1), 2)}%   |   Max  =  {round(np.quantile(yes_losers_profit, 1), 2)}%    |\n'
                      f'\nAvg  =  {round(np.mean(yes_all_profit), 2)}%    |   Avg  =  {round(np.mean(yes_winners_profit), 2)}%   |    Avg  =  {round(np.mean(yes_losers_profit), 2)}%    |'
                      f'\n-------------------------------------------------------------|'
                      f'\n Distribution of time taken to hit {x_pct}% (in biz days) \n'
                      f'\n  All Trades      |      Winners       |       Losers        |'
                      f'\nMin  =  {round(np.quantile(yes_all_tth, 0), 1)}         |   Min  =  {round(np.quantile(yes_winners_tth, 0), 1)}        |   Min  =  {round(np.quantile(yes_losers_tth, 0), 1)}         |'
                      f'\n1Q   =  {round(np.quantile(yes_all_tth, 0.25), 1)}       |   1Q   =  {round(np.quantile(yes_winners_tth, 0.25), 1)}      |   1Q   =  {round(np.quantile(yes_losers_tth, 0.25), 1)}       |'
                      f'\nMed  =  {round(np.quantile(yes_all_tth, 0.5), 1)}      |   Med  =  {round(np.quantile(yes_winners_tth, 0.5), 1)}     |   Med  =  {round(np.quantile(yes_losers_tth, 0.5), 1)}      |'
                      f'\n3Q   =  {round(np.quantile(yes_all_tth, 0.75), 1)}      |   3Q   =  {round(np.quantile(yes_winners_tth, 0.75), 1)}     |   3Q   =  {round(np.quantile(yes_losers_tth, 0.75), 1)}      |'
                      f'\nMax  =  {round(np.quantile(yes_all_tth, 1))}       |   Max  =  {round(np.quantile(yes_winners_tth, 1), 1)}       |   Max  =  {round(np.quantile(yes_losers_tth, 1), 1)}       |\n'
                      f'\nAvg  =  {round(np.mean(yes_all_tth), 1)}      |   Avg  =  {round(np.mean(yes_winners_tth), 1)}     |   Avg  =  {round(np.mean(yes_losers_tth), 1)}      |'
                      f'\n-------------------------------------------------------------|'
                      f'\n Distribution of time taken after hitting {x_pct}% (in biz days) \n'
                      f'\n  All Trades      |      Winners       |       Losers        |'
                      f'\nMin  =  {round(np.quantile(yes_all_tah, 0), 1)}         |   Min  =  {round(np.quantile(yes_winners_tah, 0), 1)}       |   Min  =  {round(np.quantile(yes_losers_tah, 0), 1)}         |'
                      f'\n1Q   =  {round(np.quantile(yes_all_tah, 0.25), 1)}       |   1Q   =  {round(np.quantile(yes_winners_tah, 0.25), 1)}     |   1Q   =  {round(np.quantile(yes_losers_tah, 0.25), 1)}       |'
                      f'\nMed  =  {round(np.quantile(yes_all_tah, 0.5), 1)}      |   Med  =  {round(np.quantile(yes_winners_tah, 0.5), 1)}     |   Med  =  {round(np.quantile(yes_losers_tah, 0.5), 1)}       |'
                      f'\n3Q   =  {round(np.quantile(yes_all_tah, 0.75), 1)}      |   3Q   =  {round(np.quantile(yes_winners_tah, 0.75), 1)}    |   3Q   =  {round(np.quantile(yes_losers_tah, 0.75), 1)}      |'
                      f'\nMax  =  {round(np.quantile(yes_all_tah, 1))}       |   Max  =  {round(np.quantile(yes_winners_tah, 1), 1)}      |   Max  =  {round(np.quantile(yes_losers_tah, 1), 1)}       |\n'
                      f'\nAvg  =  {round(np.mean(yes_all_tah), 1)}      |   Avg  =  {round(np.mean(yes_winners_tah), 1)}     |   Avg  =  {round(np.mean(yes_losers_tah), 1)}      |\n\n'
                      )

                if plot:
                    # plotting yes group
                    self.plot_trade_life_equity(yes_path, trade_type='winners',
                                                plot_title=f'{j.upper()} Winning trades that hit {x_pct}%')
                    self.plot_trade_life_equity(yes_path, trade_type='losers',
                                                plot_title=f'{j.upper()} Losing trades that hit {x_pct}%')

        return

    def plot_trade_life_equity(self, trade_type='all', plot_title='Trade P&L evolution'):
        trade_equity_dict = self.get_trade_val_dict()
        color_map = {'winners': 'lime',
                     'losers': 'red'}
        fig = make_subplots(rows=1, cols=1)
        end_x = []
        end_y = []
        for k, v in trade_equity_dict.items():
            x_ = list(range(len(v.index)))  # integer number of dates
            # x_ = v.index  # datetime index
            y_ = (v / v[0] - 1) * 100  # if results are not in realtive value
            # y_ = v
            if y_[-1] < 0:
                state = 'losers'
            else:
                state = 'winners'

            if trade_type == 'all':
                fig.add_trace(go.Scatter(x=x_, y=y_, line=dict(color=color_map[state]), name=k), row=1, col=1)
                end_x.append(x_[-1])
                end_y.append(y_[-1])
            elif trade_type == 'losers':
                if state == 'losers':
                    fig.add_trace(go.Scatter(x=x_, y=y_, line=dict(color=color_map[state]), name=k), row=1, col=1)
                    end_x.append(x_[-1])
                    end_y.append(y_[-1])
            elif trade_type == 'winners':
                if state == 'winners':
                    fig.add_trace(go.Scatter(x=x_, y=y_.values, line=dict(color=color_map[state]), name=k), row=1,
                                  col=1)
                    end_x.append(x_[-1])
                    end_y.append(y_[-1])
        # DOESNT WORK NEED TO FIX SO WE CAN SEE THE END POINT OF TRADES
        fig.add_trace(go.Scatter(mode='markers', x=end_x, y=end_y,
                                 marker_line_color='white', marker_symbol='x-thin',
                                 marker_size=8, marker_line_width=2), row=1, col=1)

        fig.update_layout(title=f'{trade_type} {plot_title}'.upper(),
                          xaxis_rangeslider_visible=False,
                          template='plotly_dark')

        fig.show()
        return fig

    def generate_outputs(self, strat_param, export=True):
        summary_str = self.get_summary(strat_param)
        if export:
            with open(f'{self.output_path}/summary.txt', 'w') as f:
                f.write(summary_str)

        return

    def generate_extra_plots(self, export=True):

        return

    def generate_trade_plot(self, data_obj, ticker=None, time_frame='daily', ma=False,
                            ma_type=None, ma_lookbacks=None):
        fig = data_obj.get_stock_chart(stock=ticker, timeframe=time_frame, ma=ma, ma_type=ma_type,
                                       ma_lookbacks=ma_lookbacks, display_plot=False)

        trades_df = self.trade_list[self.trade_list['symbol'] == ticker].reset_index(drop=True)

        for i in range(len(trades_df)):
            ann_txt = f'OpenDate: {trades_df.iloc[i]["open_date"].strftime("%d-%m-%Y")}' \
                      f'\nCloseDate: {trades_df.iloc[i]["close_date"].strftime("%d-%m-%Y")} ' \
                      f' ({trades_df.iloc[i]["time_in_trade"]} days)' \
                      f'\nProfit: {trades_df.iloc[i]["profit%"] * 100}%'
            if trades_df.iloc[i]["profit%"] > 0:
                col = 'green'
            else:
                col = 'red'
            fig.add_vrect(x0=trades_df.iloc[i]['open_date'],
                          x1=trades_df.iloc[i]['close_date'],
                          annotation_text=f'\nProfit: {round(trades_df.iloc[i]["profit%"] * 100, 2)}%',
                          # annotation_position='top left',
                          fillcolor=col,
                          opacity=0.2, line_width=0)

        plot(fig)

        return

    def plot_CI_chart(self, data_obj, ticker_key='SPY', timeframe='daily', ci_name='40_3', equity=False):
        bt_output_path=''
        if timeframe == 'weekly':
            data_obj.get_weekly_data(data_obj)
            index_closes = data_obj.weekly_closes[ticker_key]
        elif timeframe == 'daily':
            index_closes = data_obj.daily_closes[ticker_key]
            index_closes.name = 'close'
            index_df = pd.DataFrame(index_closes)
            index_df['open'] = data_obj.daily_opens[ticker_key]
            index_df['high'] = data_obj.daily_highs[ticker_key]
            index_df['low'] = data_obj.daily_lows[ticker_key]
            index_df['volume'] = data_obj.daily_volumes[ticker_key]

        # Create SubPlot
        fig = make_subplots(rows=3, cols=1,
                            specs=[[{'secondary_y': True}],
                                   [{'secondary_y': False}],
                                   [{'secondary_y': False}]],
                            row_width=[0.15, 0.15, 0.7],
                            shared_xaxes=True,
                            subplot_titles=('Equity', 'Congestion Index', 'CI ROC'), vertical_spacing=0.05)
        # OHLC candlstick plot
        fig.add_trace(go.Candlestick(x=index_df.index,
                                     open=index_df['open'],
                                     high=index_df['high'],
                                     low=index_df['low'],
                                     close=index_df['close'],
                                     name=ticker_key),
                      row=1, col=1, secondary_y=False)

        fig.update_layout(title=f'Strategy performance plotted against {ticker_key}',
                          xaxis_rangeslider_visible=False,
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              orientation='h'),
                          template='plotly_dark')
        if equity:
            fig.add_trace(go.Scatter(x=self.equity_df.index, y=self.equity_df['equity'],
                                     name=f'{ticker_key} Equity Curve', line=dict(color='white')),
                          row=1, col=1, secondary_y=True)
            fig.update_layout(yaxis2=dict(scaleanchor='y', scaleratio=0.001))

        ci_series = eval(f'data_obj.CI_{ci_name}[ticker_key]')
        ci_df = pd.DataFrame(ci_series)
        ci_df.columns = [f'{col}_CI' for col in ci_df.columns]

        ci_roc_smoothing = 3
        ci_df['CI_diff'] = (ci_series - ci_series.shift(1))
        ci_df['CI_diff_Smoothed'] = ci_df['CI_diff'].ewm(span=ci_roc_smoothing).mean()

        final_df = pd.concat([self.equity_df, ci_df], axis=1)
        fig.add_trace(go.Scatter(x=ci_series.index, y=ci_series.values,
                                 name=f'{ticker_key} Congestion Index', line=dict(color='green'), fill='tozeroy'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=ci_series.index, y=[20] * len(ci_series.index),
                                 name=f'CI bound', line=dict(color='red')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=ci_series.index, y=[-20] * len(ci_series.index),
                                 name=f'CI bound', line=dict(color='red')),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=ci_df['CI_diff'].index, y=ci_df['CI_diff'].values,
                                 name=f'{ticker_key} CI diff', line=dict(color='blue')),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=ci_df['CI_diff_Smoothed'].index, y=ci_df['CI_diff_Smoothed'].values,
                                 name=f'{ticker_key} CI diff Smoothed', line=dict(color='yellow')),
                      row=3, col=1)
        final_df.to_csv(f'{bt_output_path}/congestion index.csv')
        plot(fig)
        return fig


#
# ------------------------------------------------------------------------------------------------------------------
#       MULTI STRAT COMPARISION (outside class)


def compare_multiple_strategies(strat_path_dict, out_path=None):
    strat_dict = {}
    for k, v in strat_path_dict.items():
        # exec(f'{k} = BacktestOutputAnalysis(r"{v}")')
        strat_dict[k] = BOA(v, name=k)

    get_strat_comparison_plot(strat_dict)
    comparision_summary_stats(strat_dict)
    return 1


def compare_strats_from_path(path):
    equity_df_dict = {}
    for fold in os.listdir(path):
        strat_name = fold   #fold.split('-')[1]
        for fname in os.listdir(f'{path}/{fold}'):
            if 'equity_df' in fname and '.csv' in fname:
                equity_df_dict[strat_name] = pd.read_csv(f'{path}/{fold}/{fname}', index_col=0, parse_dates=True,
                                                         dayfirst=True)

    fig = get_strat_comparison_plot(equity_df_dict)
    fig.write_html(f'{path}/Strat_comparison.html')
    return


def get_strat_comparison_plot(equity_df_dict):
    # # Create SubPlot - CAN ADD UTILITY to this plot
    fig = make_subplots(rows=3, cols=1,
                        row_width=[0.15, 0.15, 0.7],
                        shared_xaxes=True,
                        subplot_titles=['Equity', 'Drawdown', 'Utility'],
                        vertical_spacing=0.05)
    # fig = plot_ta_chart(data)

    for k, v in equity_df_dict.items():
        fig.add_trace(go.Scatter(x=v.index,
                                 y=v['equity'],
                                 name=f'{k} equity'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=v.index,
                                 y=v['drawdown'],
                                 fill='tozeroy',
                                 name=f'{k} drawdown'),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=v.index, y=v['utility'], name=f'{k} utility',
                                 fill='tozeroy'),
                      row=3, col=1, secondary_y=False)
    fig.update_layout(title='Strategy performance comparison',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')
    plot(fig)

    return fig


def comparision_summary_stats(strategy_dict):
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 7)
    summary_df = None
    trade_stat_fig = make_subplots(rows=1, cols=4, subplot_titles=['Biz Days in Trades',
                                                                   'All Profit %',
                                                                   'Winner Profit %',
                                                                   'Loser Profit %'])
    i = 0
    colours = ['darkviolet', 'deeppink', 'deepskyblue', 'blue', 'green', 'orange', 'aquamarine', 'yellow']
    for k, v in strategy_dict.items():
        # Summary stats
        # if summary_df is None:
        #     summary_df = v.summary.T.rename(columns={0: k}).round(2)
        # else:
        #     summary_df = pd.concat([summary_df, v.summary.T.rename(columns={0: k}).round(2)], axis=1)

        v.get_summary()
        trade_prof = v.trade_list['profit%'].round(2)
        strat_key = [k] * len(trade_prof)
        winners = trade_prof[trade_prof > 0]
        losers = trade_prof[trade_prof < 0]
        # trade stat plot
        trade_stat_fig.add_trace(go.Box(x=strat_key,
                                        y=v.trade_list['days_in_trade'].values, notched=False, name=k,
                                        line=dict(color=colours[i])),
                                 col=1, row=1)
        trade_stat_fig.add_trace(go.Box(x=strat_key, y=trade_prof.values, notched=False, name=k,
                                        line=dict(color=colours[i])),
                                 col=2, row=1)
        trade_stat_fig.add_trace(go.Box(x=strat_key, y=winners.values, notched=False, name=k,
                                        line=dict(color=colours[i])),
                                 col=3, row=1)
        trade_stat_fig.add_trace(go.Box(x=strat_key, y=losers.values, notched=False, name=k,
                                        line=dict(color=colours[i])),
                                 col=4, row=1)
        i += 1

    trade_stat_fig.update_layout(title=f'Trade Analysis', showlegend=False)
    trade_stat_fig.show()

    return


def get_optimisation_boxplots(basepath, filename):
    opt_df = pd.read_csv(f'{basepath}/{filename}.csv')

    user_params = [x for x in opt_df.columns if 'user.' in x]

    user_params = ['week_lookback',
                   'upthrust_weeks',
                   'end_uptrend',
                   # 'stock_hist_percent',
                   'index_hist_percent',
                   'exit_weeks'
                   ]

    colours = ['darkviolet', 'deeppink', 'deepskyblue']
    perform_metric = ['total_profit',
                      '%_winners',
                      'max_drawdown',
                      '#trades',
                      'avg_profit',
                      'length_max_drawdown']

    # fig = make_subplots(rows=3, cols=3, subplot_titles=user_params)
    # rw = 0
    # cl = 0
    # for v in range(len(user_params)):
    #     for t in px.box(opt_df, x=user_params[v], y='total_profit', color=user_params[v]).values:
    #         fig.add_trace(t, row=(v//3)+1, col=(v % 3)+1)
    #
    # fig.update_layout(boxmode='group', )

    for u in user_params:
        fig = make_subplots(rows=2, cols=3, subplot_titles=perform_metric)
        i = 0
        for p in perform_metric:
            fig.add_trace(go.Box(x=opt_df[u], y=opt_df[p], notched=False, name=p),
                          col=(i % 3) + 1, row=(i // 3) + 1)
            i += 1
        fig.update_layout(title=f'Optimisation of {u}')
        fig.show()

    return


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    #       STATIC DATA
    stock_data = r'C:\Users\Nick_Elmer\Documents\BarnesD\Dev\stock_data\Data_1996_2022Jan'  # folder path
    # spy_key = '147418'
    # spy_key = 'SPY'
    # stock_data = {'SPY'} # for norgate
    # # FOR NORGATE
    # data_fields_needed = ['Open', 'High', 'Low', 'Close']  # The fields needed. If `check_stop_loss` is used, need OHLC
    # data_adjustment = 'TotalReturn'  # The type of adjustment of the data

    # Strat static
    rebalance = 'weekly'  # 'daily', 'weekly', 'month-end', 'month-start'
    starting_cash = 100000  # 2 dp. float
    data_source = 'local_csv'  # Either 'Norgate' or 'local_csv'
    start_date = dt.date(1998, 11, 18)  # The date trading will start
    end_date = dt.date(2021, 9, 20)  # The date trading will end - left off last two years so we dont over fit
    offset = 1
    max_lookback = 0

    print_outpath = r'C:\Users\Nick_Elmer\Documents\BarnesD\spy_breakout\Results'
    # ------------------------------------------------------------------------------------------------------------------
    #       RETREIVE PRICE DATA
    pricedata = PriceData(start_dt=start_date,
                          end_dt=end_date,
                          rebalance=rebalance,
                          offset=offset,
                          max_lkback=max_lookback,
                          data_source=data_source,
                          data_path=stock_data,
                          )
    ci_period = 40
    ci_smoothing = 3
    congestion_name = f'{ci_period}_{ci_smoothing}'
    pricedata.get_congestion_index_df(ci_period, ci_smoothing)

    # ------------------------------------------------------------------------------------------------------------------
    #       SINGLE STRAT COMPARISION
    #
    # bt_output_path = r'C:\Users\Nick_Elmer\Documents\BarnesD\spy_breakout\Results\12.corrected\7-7'
    # bt_output_path = r'C:\Users\Nick_Elmer\Documents\BarnesD\Dev\q_lib\sp_breakout_strat\Outputs\Backtests\dailybars Feb22\20220211_1517'
    # strat = BOA(portfolio=bt_output_path)
    # strat.plot_excursion_graphs()
    # strat.plot_CI_chart(pricedata, ci_name=congestion_name, equity=True)
    # #
    # # # strat.generate_trade_plot(pricedata, ticker='BA', time_frame='weekly', ma=True, ma_type='simp',
    # # #                           ma_lookbacks=[20, 50, 200])
    # # # strat.generate_trade_plot(pricedata, ticker='BA', time_frame='daily', ma=True, ma_type='simp',
    # # #                           ma_lookbacks=[20, 50, 200])
    bla = 1
    # # strat.get_summary()
    # # strat.plot_trade_life_equity(trade_type='all')
    # # strat.plot_trade_life_equity(trade_type='winners')
    # # strat.plot_trade_life_equity(trade_type='losers')
    # #
    # # strat.deeper_trade_analysis(plot=False, pct_list=[0, -1, -2, -3, -4, -5, -6], year_split=False)
    # strat.plot_equity_curve(pricedata, with_benchmark=True, benchmark='SPY')

    # ------------------------------------------------------------------------------------------------------------------
    #       MULTI STRAT COMPARISION
    #
    # strats = {
    #     'strat_v2': r'C:\Users\Nick_Elmer\Documents\BarnesD\spy_breakout\Results\10.insample_stratv2\strat_v2_fix',
    #     'strat_v2.03': r'C:\Users\Nick_Elmer\Documents\BarnesD\spy_breakout\Results\10.insample_stratv2\strat_v203_fix',
    #     'strat_v2.03_dym': r'C:\Users\Nick_Elmer\Documents\BarnesD\spy_breakout\Results\10.insample_stratv2\strat_v203_dyn'
    # }
    # sum_df = compare_multiple_strategies(strats)

    # MULTI EQUITY CURVE
    strat_path = r'C:\Users\Nick_Elmer\Documents\BarnesD\Dev\q_lib\sp_breakout_strat\Outputs\Backtests\weekly_breakevn'
    compare_strats_from_path(strat_path)

    # # BOXPLOTS
    # strat_path = r'C:\Users\Nick_Elmer\Documents\BarnesD\Dev\q_lib\sp_breakout_strat\Outputs\Optimisation\20220215_1411'
    # get_optimisation_boxplots(strat_path, 'OptimisationReport_20220215_1411')
