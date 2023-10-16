FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN chmod +x /app/inituser_and_start.sh

# 在容器启动时运行 entrypoint.sh 脚本
CMD ["./inituser_and_start.sh"]

