'''
Our methodology to preprocess our data. 
'''

import datetime
import functools

from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
import pandas as pd
import gc

def aggregate_features(df: pd.DataFrame):
    # # Features which we want to aggregate something special from (per cid)
    features_avg = [c for c in df.columns if c not in ['customer_ID', 'S_2']] #['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']
    features_min = [c for c in df.columns if c not in ['customer_ID', 'S_2']] #['B_2', 'B_4', 'B_5', 'B_9', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_20', 'B_28', 'B_29', 'B_33', 'B_36', 'B_42', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_71', 'D_74', 'D_75', 'D_78', 'D_83', 'D_102', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 'D_140', 'D_141', 'D_144', 'D_145', 'P_2', 'P_3', 'R_1', 'R_27', 'S_3', 'S_5', 'S_7', 'S_9', 'S_11', 'S_12', 'S_23', 'S_25']
    features_max = [c for c in df.columns if c not in ['customer_ID', 'S_2']] #['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_21', 'B_23', 'B_24', 'B_25', 'B_29', 'B_30', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_52', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_63', 'D_64', 'D_65', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_91', 'D_102', 'D_105', 'D_107', 'D_110', 'D_111', 'D_112', 'D_115', 'D_116', 'D_117', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_138', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_3', 'R_5', 'R_6', 'R_7', 'R_8', 'R_10', 'R_11', 'R_14', 'R_17', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
    features_last = [c for c in df.columns if c not in ['customer_ID', 'S_2']] #['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 'D_80', 'D_81', 'D_82', 'D_83', 'D_86', 'D_91', 'D_96', 'D_105', 'D_106', 'D_112', 'D_114', 'D_119', 'D_120', 'D_121', 'D_122', 'D_124', 'D_125', 'D_126', 'D_127', 'D_130', 'D_131', 'D_132', 'D_133', 'D_134', 'D_138', 'D_140', 'D_141', 'D_142', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_12', 'R_13', 'R_14', 'R_15', 'R_19', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 'S_13', 'S_16', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
    
    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer
    if 'target' in df.columns:
        df.drop(columns=['target'], inplace=True)
    gc.collect()
    print('Read')
    df_avg = (df
              .groupby(cid)
              .mean()[features_avg]
              .rename(columns={f: f"{f}_avg" for f in features_avg})
             )
    gc.collect()
    print('Computed avg.')
    df_min = (df
              .groupby(cid)
              .min()[features_min]
              .rename(columns={f: f"{f}_min" for f in features_min})
             )
    gc.collect()
    print('Computed min.')
    df_max = (df
              .groupby(cid)
              .max()[features_max]
              .rename(columns={f: f"{f}_max" for f in features_max})
             )
    gc.collect()
    print('Computed max.')
    df = (df.loc[last, features_last]
          .rename(columns={f: f"{f}_last" for f in features_last})
          .set_index(np.asarray(cid[last]))
         )
    gc.collect()
    print('Computed last')
    df = pd.concat([df, df_min, df_max, df_avg], axis=1)
    del df_min, df_max, df_avg
    gc.collect()
    return df
    
def chart_diff(before, after, markers):
    n_cols = 4
    n_rows = 40
    for i, f in enumerate(before.columns):
        if i % (n_rows * n_cols) == 0 or i == before.columns.size - 1:
            if i > 0 or i == before.columns.size - 1: 
                plt.savefig(f"../output/feature-avg-vs-day/{i}.png")
                plt.close()
            plt.figure(figsize=(n_cols * 9, n_rows * 7))
            if i == 0: plt.suptitle('Avg. of Feature for each Day')
        plt.subplot(n_rows, n_cols, i % (n_rows * n_cols) + 1)
        
        def plot(df, label):
            df_temp = pd.concat([df[f], markers['d_end']], axis=1, keys=[f, 'd_end'])
            df_temp.set_index('d_end', inplace=True)
            df_temp.sort_index(inplace=True)
            df_temp[f] = df_temp.groupby("d_end")[f].mean()
            
            x = np.array(df_temp.index.values)
            y = np.array(df_temp[f])
            
            # Choose scatter or polyfit
            plt.scatter(x, y, alpha=0.5, label=label, marker='x')
            # plt.plot(x, np.poly1d(np.polyfit(x, y, 8))(x), alpha=0.5, label=label)
        
        plot(before, f"Before. Unique={before[f].unique().size}")
        plot(after, f"After. Unique={after[f].unique().size}")
        
        plt.xlabel('Date')
        plt.ylabel('Feature Avg.')
        plt.title(f'Feature: {f}')
        plt.legend(loc="upper right")
        
def plot_non_null(temp):
    temp.set_index('customer_ID', inplace=True)
    temp["S_2"] = temp["S_2"].map(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d"))
    temp['last_month'] = temp.groupby('customer_ID').S_2.max().dt.month
        
    n_cols = 4
    n_rows = 40
    for i, f in enumerate(temp.columns[2:]):
        if i % (n_rows * n_cols) == 0 or i == temp.columns.size - 1:
            if i > 0 or i == temp.columns.size - 1: 
                plt.savefig(f"../output/non-null/{i}.png")
                plt.close()
            plt.figure(figsize=(n_cols * 16, n_rows * 7))
            if i == 0: plt.suptitle('Non-null values for each Day')
        plt.subplot(n_rows, n_cols, i % (n_rows * n_cols) + 1)
        
        temp['has_f'] = temp[f].map(lambda k: k != np.nan and k != -1)
        plt.hist([temp.S_2[temp.has_f & (temp.last_month == 3)],   # ending 03/18 -> training
              temp.S_2[temp.has_f & (temp.last_month == 4)],   # ending 04/19 -> public lb
              temp.S_2[temp.has_f & (temp.last_month == 10)]], # ending 10/19 -> private lb
             bins=pd.date_range("2017-03-01", "2019-11-01", freq="MS"),
             label=['Training', 'Public leaderboard', 'Private leaderboard'],
             stacked=True)
        plt.xticks(pd.date_range("2017-03-01", "2019-11-01", freq="QS"))
        plt.xlabel('Statement date')
        plt.ylabel(f'Count of {f} non-null values')
        plt.title(f'{f} non-null values over time')
        plt.legend()

def plot_categorical(cats, days, cpy):
    cpy.set_index('customer_ID', inplace=True)
    cpy["S_2"] = cpy["S_2"].map(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d"))
    cpy['last_month'] = cpy.groupby('customer_ID').S_2.max().dt.month
    cpy.reset_index(inplace=True)
    cats["last_month"] = cpy['last_month']
    
    n_cols = 4
    n_rows = 40
    k = 1
    
    plt.figure(figsize=(n_cols * 16, n_rows * 7))
    for f in cats.columns[:-1]:
        for v in cats[f].unique():
            plt.subplot(n_rows, n_cols, k)
            plt.hist(
                [
                    days[(cats[f] == v) & (cats.last_month == 3)],   # ending 03/18 -> training
                    days[(cats[f] == v) & (cats.last_month == 4)],   # ending 04/19 -> public lb
                    days[(cats[f] == v) & (cats.last_month == 10)]   # ending 10/19 -> private lb
                ],
                label=['Training', 'Public leaderboard', 'Private leaderboard'],
                bins=pd.date_range("2017-03-01", "2019-11-01", freq="MS"),
                stacked=True
            )
            plt.xticks(pd.date_range("2017-03-01", "2019-11-01", freq="QS"))
            plt.xlabel('Statement date')
            plt.ylabel(f'Count')
            
            plt.title(f'Count of {f} valued {v} over time')
            
            k += 1
    plt.savefig(f"../output/categorical_dst.png")
    plt.close()


rmv_definite = [
    # these were useless, from categorical analysis
    'R_18', 'D_87', 
] 
rmb_maybe = [
    # from null-distribution analysis
    'B_29', 'D_42', 'S_9',  
    
    # from categorical analysis
    'D_116',
]

def preprocess(train_in: str, train_out: str, test_in: str, test_out: str):
    # train = pd.read_parquet(train_in)
    # test = pd.read_parquet(test_in)
    # print("Read inputs")
    
    # test_start_cid = test['customer_ID'][0]
    # temp = pd.concat([train, test], axis=0, ignore_index=True)
    # del train, test
    
    # # Remove outlier statements from categorical analysis
    # temp = temp[(temp['D_64'] != 1) & (temp['D_66'] != 0) & (temp['D_68'] != 0)]
    # temp.reset_index(inplace=True)
    # print("Removed outliers")
    
    # # Drop features from our graphical analysis
    # temp.drop(rmv_definite, axis=1, inplace=True)
    # temp.drop(rmb_maybe, axis=1, inplace=True)
    # print("Dropped features")
    
    # # Remove cids 
    # cids = temp.pop('customer_ID')
    
    # # Remove day lavels
    # days = temp.pop("S_2").map(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d"))
    
    # # Remove all categorical and categorical-like features
    # categorical = pd.DataFrame()
    # # if f is one of the AMEX given caterogicals
    # for f in ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']:
    #     try:
    #         categorical[f] = temp.pop(f)
    #     except:
    #         # f has been dropped beforehand
    #         pass
    # # if f only consists of [-1, 0, 1]
    # for f in temp.columns:
    #     if np.isin(temp[f].unique(), [-1, 0, 1]).all():
    #         categorical[f] = temp.pop(f)
    # print(f"Calculated categorical: {categorical.columns}")
    # # plot_categorical(categorical, days, cpy)
    
    # # Create Markers to mark relative time of each statement
    # markers = pd.DataFrame()
    # markers['d_end'] = days.map(lambda d: (datetime.datetime(2019, 12, 1) - d).days)
    # markers['d_year'] = days.map(lambda d: (d - datetime.datetime(d.year, 1, 1)).days)
    # markers['d_biannual'] = markers['d_year'] % int(365 / 2)
    # markers['d_month'] = days.map(lambda d: d.day)
    # markers['d_week'] = days.map(lambda d: d.weekday())
    # print("Made markers")
    
    # # Fit time-static bias to each marker, and shift the continious data
    # for f in temp.columns:
    #     for marker in markers.columns:
    #         fit = pd.concat([temp[f], markers[marker]], axis=1, keys=[f, marker])
    #         fit.dropna(inplace=True)
    #         fit.set_index(marker, inplace=True)
    #         fit[f] = fit.groupby(marker)[f].mean()
            
    #         p = np.poly1d(np.polyfit(fit.index.values, fit[f], 1))
    #         bias = p(markers[marker])
    #         temp[f] -= bias
    # # chart_diff(before, temp, markers)
    # print("Finished Scaling")
    
    # # now combine and separate back into test and train 
    # df = pd.concat([cids, days, categorical, temp], axis=1)
    # df.to_parquet('../input/processed/scaled_all.parquet')
    # return
    
    test = pd.read_parquet(test_in)
    test_start_cid = test['customer_ID'][0]
    del test
    
    df = pd.read_parquet('../input/processed/scaled_all.parquet')
    print('Now aggregating')
    df = aggregate_features(df)
    print('Done aggregating')
    
    s = df.index.get_loc(test_start_cid)
    train = df.iloc[0:s]
    test = df.iloc[s:]
    
    train.to_parquet(train_out)
    test.to_parquet(test_out)
    print("All done")

if __name__ == "__main__":
    # dir_train_in = '../input/subsampled'
    # dir_test_in = '../input/subsampled'
    dir_train_in = '../input'
    dir_test_in = '../input'
    dir_out = '../input/processed' 

    preprocess(
        f'{dir_train_in}/train.parquet', f'{dir_out}/train.parquet', 
        f'{dir_test_in}/test.parquet', f'{dir_out}/test.parquet'
    )