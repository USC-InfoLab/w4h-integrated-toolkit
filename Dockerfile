FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN ls

RUN chmod +x /app/inituser_and_start.sh

CMD ["./app/inituser_and_start.sh"]

