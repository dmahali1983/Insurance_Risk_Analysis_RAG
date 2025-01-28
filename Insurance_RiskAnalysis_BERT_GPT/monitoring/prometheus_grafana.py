from prometheus_client import start_http_server, Summary, Counter, Gauge
import time

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total request count')
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the model')

def process_request():
    REQUEST_COUNT.inc()
    with REQUEST_TIME.time():
        time.sleep(1)  # Simulate processing delay

if __name__ == "__main__":
    start_http_server(8000)
    while True:
        process_request()
        MODEL_ACCURACY.set(0.85)  # Example metric update