df_avg = (df
#             .groupby(cid)
#             .mean()[features_avg]
#             .rename(columns={f: f"{f}_avg" for f in features_avg})
#             )
# gc.collect()
# print(df_avg.shape)
