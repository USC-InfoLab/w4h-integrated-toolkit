import pandas as pd

# 创建一个空的 DataFrame
df = pd.DataFrame(columns=['path_name', 'color', 'width'])

# 添加一行数据，其中 "path_name" 列包含路径数据
data = {
    'path_name': [ [1, 2], [3, 4], [5, 6] ],
    'color': 'red',
    'width': 2
}

# 将数据添加到 DataFrame
df = df.append(data, ignore_index=True)



# 打印结果 DataFrame
print(df)