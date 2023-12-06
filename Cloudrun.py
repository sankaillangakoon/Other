import requests
import os

def pubsub_trigger(event, context):
    """Triggered by a Pub/Sub message."""
    cloud_run_url = os.getenv('https://cloudrun-jbeso3reka-nw.a.run.app')  # Set this as an environment variable

    response = requests.post(f"{cloud_run_url}/run-model")  # Add any required payload
    return response.text
