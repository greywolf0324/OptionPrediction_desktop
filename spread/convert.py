import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import time
from multiprocessing import Process, Manager

def vertical_combination_score(stock_price, option1, option2):

    """ calculate probability_of_profit """
    if option1.type == 'call':
        long_option = option1 if option1.strike < option2.strike else option2
        short_option = option2 if option1.strike >= option2.strike else option2
        
        net_premium = short_option.bid - long_option.ask
        breakeven = long_option.strike + net_premium
    else:  # for put
        long_option = option1 if option1.strike > option2.strike else option2
        short_option = option2 if option1.strike <= option2.strike else option2
        
        net_premium = short_option.bid - long_option.ask
        breakeven = long_option.strike - net_premium
    
    # Calculate days to expiration
    expiry_date = datetime.strptime(long_option.expiration, '%Y-%m-%d')
    quote_date = datetime.strptime(long_option.quote_date, '%Y-%m-%d')
    days_to_expiry = (expiry_date - quote_date).days
    
    # Estimate standard deviation at expiration
    avg_volatility = (long_option.implied_volatility + short_option.implied_volatility) / 2
    std_dev = avg_volatility * (days_to_expiry ** 0.5) * stock_price
    
    # Calculate PoP using the standard normal distribution
    if long_option.type == 'call':
        probability_of_profit = 1 - norm.cdf((breakeven - stock_price) / std_dev)
    else:  # for put
        probability_of_profit = norm.cdf((breakeven - stock_price) / std_dev)
    
    if probability_of_profit[0] < 0.98:
        return []
    """ calculate max_profit_percentage """
    net_premium = short_option.bid - long_option.ask

    if net_premium > 0:  # Credit Spread
        max_profit = net_premium
        capital_required = abs(option2.strike - option1.strike) - net_premium
    else:  # Debit Spread
        max_profit = abs(option2.strike - option1.strike) + net_premium
        capital_required = -net_premium  # Since net_premium is negative for a debit spread
    if capital_required == 0:
        max_profit_percentage = 0
    else:
        max_profit_percentage = (max_profit / capital_required) * 100
    if max_profit_percentage <= 0:
        return []
    """ calculate front and back volume """
    volume1 = option1.volume
    volume2 = option2.volume

    # Calculate front and back volume percentages
    total_volume = volume1 + volume2
    if total_volume == 0:
        front_volume_percentage = back_volume_percentage = 0
    else:
        front_volume_percentage = (volume1 / total_volume) * 100
        back_volume_percentage = 100 - front_volume_percentage
    
    """ calculate the mark and delta"""
    mark1 = (option1.bid + option1.ask) / 2
    mark2 = (option2.bid + option2.ask) / 2
    delta1 = option1.delta
    delta2 = option2.delta

    # Determine the spread type (debit or credit)
    # Note: This assumes both options are of the same type (either both calls or both puts)
    if option1.strike < option2.strike:
        mark =  mark2 - mark1  # Debit spread
        delta = delta2 - delta1
    else:
        mark = mark1 - mark2  # Credit spread
        delta = delta1 - delta2
    
    return probability_of_profit[0] * 100, max_profit_percentage, front_volume_percentage, back_volume_percentage, mark, delta

def butterfly_combination_score(stock_price, lower_option, middle_option, upper_option):
    """ calculate probability_of_profit """
    if lower_option.type == 'call':
        net_premium = lower_option.ask - 2 * middle_option.bid + upper_option.ask
    else:  # for put
        net_premium = upper_option.ask - 2 * middle_option.bid + lower_option.ask
    
    # Calculate breakeven points
    lower_breakeven = lower_option.strike + net_premium
    upper_breakeven = upper_option.strike - net_premium
    
    # Calculate days to expiration
    expiry_date = datetime.strptime(middle_option.expiration, '%Y-%m-%d')
    quote_date = datetime.strptime(middle_option.quote_date, '%Y-%m-%d')

    # Using implied volatility of the middle strike to estimate standard deviation at expiration
    days_to_expiry = (expiry_date - quote_date).days
    std_dev = middle_option.implied_volatility * (days_to_expiry ** 0.5)
    
    # Using properties of the log-normal distribution to estimate PoP
    pop_lower = norm.cdf((lower_breakeven - stock_price) / std_dev)
    pop_upper = 1 - norm.cdf((upper_breakeven - stock_price) / std_dev)
    
    probability_of_profit = pop_upper + pop_lower
    
    if probability_of_profit < 0.98 or probability_of_profit >= 0.9999:
        return []
    
    """ calculate max_profit_percentage """
    if (lower_option.ask + lower_option.bid + upper_option.ask + upper_option.bid) == (middle_option.ask + middle_option.bid) * 2:
        return []
    lower_buy_price = (lower_option.ask + lower_option.bid) / 2
    middle_sell_price = (middle_option.ask + middle_option.bid) / 2
    high_buy_price = (upper_option.ask + upper_option.bid) / 2
    # Buy 1 lower strike price option cost
    lower_cost = lower_buy_price
    
    # Sell 2 middle strike price option credit
    middle_credit = 2 * middle_sell_price
    
    # Buy 1 higher strike price option cost
    higher_cost = high_buy_price

    # Calculate average implied volatility and open interest
    avg_implied_volatility = (lower_option.implied_volatility + middle_option.implied_volatility + upper_option.implied_volatility) / 3
    avg_open_interest = (lower_option.open_interest + middle_option.open_interest + upper_option.open_interest) / 3

    # Calculate net premium received
    net_debit = lower_cost + higher_cost - middle_credit
    net_debit *= (1 - avg_implied_volatility)
    net_debit *= (1 - avg_open_interest / 10000)

    if net_debit == 0:
        return []
    # Calculate the maximum profit 
    max_profit = (middle_option.strike - lower_option.strike) - net_debit
    
    # Calculate the maximum profit percentage
    max_profit_percentage = (net_debit / max_profit) * 100

    if max_profit_percentage <= 20:
        return []
    """ calculate front and back volume """
    lower_volume = lower_option.volume
    middle_volume = middle_option.volume
    higher_volume = upper_option.volume
    
    # Calculate Front and Back volumes
    front_volume = middle_volume * 2
    back_volume = lower_volume + higher_volume
    total_volume = front_volume + back_volume

    front_volume_percentage = (front_volume / total_volume) * 100
    back_volume_percentage = (back_volume / total_volume) * 100

    """ calculate the mark"""
    marks = [(option.bid + option.ask) / 2 for option in [lower_option, middle_option, upper_option]]
    spread_mark = marks[0] - 2 * marks[1] + marks[2]

    """ calculate the delta"""
    spread_delta = lower_option['delta'] - 2 * middle_option['delta'] + upper_option['delta']

    return probability_of_profit[0] * 100, max_profit_percentage, front_volume_percentage, back_volume_percentage, spread_mark, spread_delta

def get_vertical_spread(combination_list, option_data, stock_data, start, ed):
    count = 0
    cur = -1
    for symbol, symbol_group in option_data.groupby('underlying'):
        cur += 1
        if cur < start:
            continue
        if cur >= ed:
            break
        stock_price = stock_data[stock_data['symbol'] == symbol].open
        
        for expire, expire_group in symbol_group.groupby('expiration'):
            for option_type, type_group in expire_group.groupby('type'):
                type_group = type_group.reset_index(drop=True)
                length = len(type_group)
                count += length * (length - 1) / 2
                for i in range(length):
                    for j in range(i + 1, length):
                        t = vertical_combination_score(stock_price, type_group.iloc[i], type_group.iloc[j])
                        if len(t) > 0:
                            combination_first = ['VERTICAL', 'SELL', -1, symbol, type_group.iloc[i].expiration, type_group.iloc[i].strike, type_group.iloc[i].type, t[4], f'{t[0]}%', f'{t[1]}%', t[1], f'{t[2]}%', f'{t[3]}%', f'{abs(t[3] - t[2])}%', t[5]]
                            combination_second = ['', 'BUY', 1, symbol, type_group.iloc[j].expiration, type_group.iloc[j].strike, type_group.iloc[j].type, '', '', '', '', '', '', '', '']
                            combination_list.append([combination_first, combination_second])
        print (f'VERTICAL {symbol} is done')

def get_butterfly_spread(combination_list, option_data, stock_data, start, ed):
    count = 0
    cur = -1
    for symbol, symbol_group in option_data.groupby('underlying'):
        cur += 1
        if cur < start:
            continue
        if cur >= ed:
            break
        stock_price = stock_data[stock_data['symbol'] == symbol].open
        
        for expire, expire_group in symbol_group.groupby('expiration'):
            for option_type, type_group in expire_group.groupby('type'):
                type_group = type_group.reset_index(drop=True)
                length = len(type_group)
                count += length * (length - 1) / 2
                for i in range(length):
                    for j in range(i + 1, length):
                        middle_strike = (type_group.iloc[i].strike + type_group.iloc[j].strike) / 2
                        middle_option = type_group[type_group['strike'] == middle_strike]
                        if not middle_option.empty:
                            t = butterfly_combination_score(stock_price, type_group.iloc[i], middle_option.iloc[0], type_group.iloc[j])
                            if len(t) > 0:
                                combination_first = ['BUTTERFLY', 'SELL', -1, symbol, type_group.iloc[i].expiration, type_group.iloc[i].strike, type_group.iloc[i].type, t[4], f'{t[0]}%', f'{t[1]}%', t[1], f'{t[2]}%', f'{t[3]}%', f'{abs(t[3] - t[2])}%', t[5]]
                                combination_second = ['', 'BUY', 2, symbol, middle_option.iloc[0].expiration, middle_option.iloc[0].strike, middle_option.iloc[0].type, '', '', '', '', '', '', '', '']
                                combination_third = ['', 'SELL', -1, symbol, type_group.iloc[j].expiration, type_group.iloc[j].strike, type_group.iloc[j].type, '', '', '', '', '', '', '', '']
                                combination_list.append([combination_first, combination_second, combination_third])
        print (f'SPREAD {symbol} is done')

def vertical_divide(option_data, divide_count):
    count = 0
    for symbol, symbol_group in option_data.groupby('underlying'):
        for expire, expire_group in symbol_group.groupby('expiration'):
            for option_type, type_group in expire_group.groupby('type'):
                type_group = type_group.reset_index(drop=True)
                length = len(type_group)
                count += length * (length - 1) / 2
    
    cur = 0
    divide_part = [0]
    symbol_index = 0
    for symbol, symbol_group in option_data.groupby('underlying'):
        for expire, expire_group in symbol_group.groupby('expiration'):
            for option_type, type_group in expire_group.groupby('type'):
                type_group = type_group.reset_index(drop=True)
                length = len(type_group)
                cur += length * (length - 1) / 2
        if cur >= count / divide_count:
            divide_part.append(symbol_index)
            cur -= count / divide_count
        symbol_index += 1
    return divide_part

def butterfly_divide(option_data, divide_count):
    count = 0
    for symbol, symbol_group in option_data.groupby('underlying'):
        for expire, expire_group in symbol_group.groupby('expiration'):
            for option_type, type_group in expire_group.groupby('type'):
                type_group = type_group.reset_index(drop=True)
                length = len(type_group)
                count += (length * (length + 1) * (2 * length + 1) / 12 + length * (length + 1) / 4)
    
    cur = 0
    divide_part = [0]
    symbol_index = 0
    for symbol, symbol_group in option_data.groupby('underlying'):
        for expire, expire_group in symbol_group.groupby('expiration'):
            for option_type, type_group in expire_group.groupby('type'):
                type_group = type_group.reset_index(drop=True)
                length = len(type_group)
                cur += (length * (length + 1) * (2 * length + 1) / 12 + length * (length + 1) / 4)
        if cur >= count / divide_count:
            divide_part.append(symbol_index)
            cur -= count / divide_count
        symbol_index += 1
    return divide_part

def predict_spread(spread_type, db_selection):
    option_data = pd.read_csv(f'dataset/options/{db_selection}options.csv')
    stock_data = pd.read_csv(f'dataset/stocks/{db_selection}stocks.csv')

    option_data = option_data[option_data['volume'] != 0]

    num_threads = 8
    if spread_type == 'VERTICAL':
        dividen_part = vertical_divide(option_data, num_threads)
    if spread_type == 'BUTTERFLY':
        dividen_part = butterfly_divide(option_data, num_threads)

    manager = Manager()
    combination_list = manager.list()

    threads = []
    for i in range(num_threads):
        if spread_type == 'VERTICAL':
            thread = Process(target=get_vertical_spread, args=(combination_list, option_data, stock_data, dividen_part[i], dividen_part[i + 1]))
        if spread_type == 'BUTTERFLY':
            thread = Process(target=get_butterfly_spread, args=(combination_list, option_data, stock_data, dividen_part[i], dividen_part[i + 1]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
    combination_list = list(combination_list)
    combination_list.sort(key=lambda x: x[0][10], reverse=True)

    new_list = []
    for combination in combination_list:
        new_list.extend(combination)

    column_names = ['Spread', 'Side', 'Qty', 'Symbol', 'Exp', 'Strike', 'Type', 'Mark', 'Prob Of Profit', 'Max Profit', 'PL/Margin', 'Front Vol', 'Back Vol', 'Vol Diff', 'Delta']
    combination_data = pd.DataFrame(new_list, columns=column_names)
    
    if spread_type == 'VERTICAL':
        combination_data.to_csv('spread/spread_vertical.csv', index=False)
        return combination_data
    elif spread_type == 'BUTTERFLY':
        combination_data.to_csv('spread/spread_butterfly.csv', index=False)
        return combination_data

if __name__ == '__main__':
    a = time.time()
    option_data = pd.read_csv('dataset/options/2013-01-02options.csv')
    stock_data = pd.read_csv('dataset/stocks/2013-01-02stocks.csv')

    print("original length: ", len(option_data))
    option_data = option_data[option_data['volume'] != 0]
    print("reduced length: ", len(option_data))
    
    num_threads = 8
    dividen_part = butterfly_divide(option_data, num_threads)
    # dividen_part = [0, 1000]
    print(dividen_part)

    manager = Manager()
    combination_list = manager.list()

    threads = []
    for i in range(num_threads):
        thread = Process(target=get_butterfly_spread, args=(combination_list, option_data, stock_data, dividen_part[i], dividen_part[i + 1]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    print(123124124, combination_list)
    combination_list = list(combination_list)
    combination_list.sort(key=lambda x: x[0][10], reverse=True)

    new_list = []
    for combination in combination_list:
        new_list.extend(combination)

    column_names = ['Spread', 'Side', 'Qty', 'Symbol', 'Exp', 'Strike', 'Type', 'Mark', 'Prob Of Profit', 'Max Profit', 'PL/Margin', 'Front Vol', 'Back Vol', 'Vol Diff', 'Delta']
    combination_data = pd.DataFrame(new_list, columns=column_names)
    combination_data.to_csv('spread/spread_butterfly.csv', index=False)