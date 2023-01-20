"""
Use Google Cloud Vision API to detect handwritten text in image
"""

from google.cloud import vision
import io

def detect_text(path):
    # key.json has service account credentials for service account with access to cloud vision API
    client = client = vision.ImageAnnotatorClient.from_service_account_json("key.json")

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            b = []
            for paragraph in block.paragraphs:
                p = []
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    p.append(word_text)
                b.append(" ".join(p))
            print(" ".join(p))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))