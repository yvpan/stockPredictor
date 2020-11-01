#coding=utf-8

import requests
import pandas
import time
import json

def get_sort_company(idx):
    url = 'https://www.nasdaq.com/api/v1/screener?page={}&pageSize=20'.format(idx)
    headers = {
        'referer': 'https://www.nasdaq.com/market-activity/stocks/screener?exchange=NASDAQ',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    }
    try:
        ret = requests.get(url,headers = headers,timeout =10)
        ret.raise_for_status()
        pagedata = ret.text
    except:
        time.sleep(3)
        try:
            ret = requests.get(url, headers=headers, timeout=10)
            ret.raise_for_status()
            pagedata = ret.text
        except:
            pagedata = ''
    ret_list = []
    try:
        data_list = json.loads(pagedata)['data']
    except:
        return ret_list
    for data in data_list:
        reqstr = 'symbol={}%7cstocks'.format(data['ticker'].lower())
        ret_list.append(reqstr)
    return ret_list

def get_content(req_list):
    reqstr = '&'.join(req_list)
    url = 'https://api.nasdaq.com/api/quote/watchlist?' + reqstr
    headers = {
        'Origin': 'https://www.nasdaq.com',
        'Referer': 'https://www.nasdaq.com/market-activity/stocks/screener?exchange=NASDAQ',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    }
    try:
        ret = requests.get(url,headers = headers,timeout =10)
        ret.raise_for_status()
        pagedata = ret.text
    except:
        time.sleep(3)
        try:
            ret = requests.get(url, headers=headers, timeout=10)
            ret.raise_for_status()
            pagedata = ret.text
        except:
            pagedata = ''
    ret_list = []
    try:
        data_list = json.loads(pagedata)['data']
    except:
        return  ret_list
    for data in data_list:
        ret_list.append([data['symbol'],data['companyName']])
    return ret_list


def control_spider():
    Result_List = []
    Result_List.append(['SYMBOL','COMPANY'])
    for idx in range(1,315):
        time.sleep(3)
        req_list = get_sort_company(idx)
        time.sleep(3)
        ret_list = get_content(req_list)
        for ret in ret_list:
            Result_List.append(ret)
        print('已经爬取第%d页数据'%idx)
    pd = pandas.DataFrame(Result_List)
    pd.to_excel('result.xls',index=None,header=None)

if __name__ == '__main__':
    control_spider()


