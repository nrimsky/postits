"""
Use Google Cloud Vision API to detect handwritten text in image
"""

from google.cloud import vision
import io


def detect_text(path):
    # key.json has service account credentials for service account with access to cloud vision API
    client = vision.ImageAnnotatorClient.from_service_account_json("key.json")
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    return response.full_text_annotation.text
