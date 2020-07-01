FROM python:3-slim as build

WORKDIR /usr/src/app

COPY main.py note_processor.py requirements.txt CovidParser_pb2_grpc.py CovidParser_pb2.py /usr/src/app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

FROM gcr.io/distroless/python3
COPY --from=build /usr/src/app /app
WORKDIR /app

CMD [ "main.py" ]