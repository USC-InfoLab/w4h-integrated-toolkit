FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

<<<<<<< HEAD
RUN ls

RUN chmod +x /app/inituser_and_start.sh

CMD ["./app/inituser_and_start.sh"]
=======
RUN chmod +x /app/inituser_and_start.sh

CMD ["./inituser_and_start.sh"]
>>>>>>> 5336d8cbb43d02384e2ba0e67982f8b5efe19783

