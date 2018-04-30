def vc_to_df(s):
    s = s.to_frame().reset_index()
    s.columns = ['col_name','value_counts']
    s.set_index('col_name', inplace=True)
    return s
 
def threshold(df,col,minimum):
    s = df[col].value_counts()
    lst = list(s[s >= minimum].index)
    return  df[df[col].isin(lst)]


__name__ == '__main__'