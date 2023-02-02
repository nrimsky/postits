from segment import segment_image
import os
import datetime
from detecttext import detect_text

def get_post_its_text(model_dir, image_path):
    dir_name = f'postits_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    print("Outputting images to", dir_name)
    os.mkdir(dir_name)
    segment_image(model_dir, image_path, save_output=True, path=dir_name)
    print("Finished segmenting image")
    items = []
    for filename in os.listdir(dir_name):
        full_path = os.path.abspath(os.path.join(dir_name, filename))
        text = detect_text(full_path)
        items.append(text)
    return items

if __name__ == "__main__":
    texts = get_post_its_text("finetuned.pt", "example.png")
    print("_____")
    for t in texts:
        print(t)
        print("_____")
    print("_____")
