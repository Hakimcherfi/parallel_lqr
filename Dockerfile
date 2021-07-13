FROM python:3.9.5-slim

RUN pip install numpy==1.20.3 matplotlib==3.4.2

WORKDIR /ddp

COPY ./parallel_lqr.py .

ENTRYPOINT ["python3","parallel_lqr.py"]
