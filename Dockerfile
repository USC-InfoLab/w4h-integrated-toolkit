FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python","init_user.py"]
CMD ["python", "stream_sim.py"]
CMD ["streamlit", "run", "viz.py"]
