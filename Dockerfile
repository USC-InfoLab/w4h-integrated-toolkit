# 使用官方的 Python 3.8.10 镜像作为基础镜像
FROM python:3.8.10

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到容器的工作目录中
COPY . .

# 安装依赖（如果有的话）
RUN pip install -r requirements.txt

# 运行应用程序
CMD ["python", "stream_sim.py"]
CMD ["streamlit", "run", "viz.py"]
