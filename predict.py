import argparse

import numpy as np

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

import json




def argparser():
    
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--imagepath', type=str)
    
    args = parser.parse_args()
    
    return args

IMAGE_SIZE = 224

def process_image(image):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image /= 255
    image = image.numpy()
    return image 

def map_data(labels_path):
    with open(labels_path, 'r') as f:
        class_names = json.load(f)
    return class_names
    
def predict(image_path, model, top_k):
    if 0 > top_k or top_k > 112:
        top_k = num_classes
    
    image = np.asarray(Image.open(image_path))
    image_arr = process_image(image)
    
    image_arr = np.expand_dims(image_arr, axis=0)
    ps = model.predict(image_arr).squeeze()
    
    predictions, labels = tf.nn.top_k(ps, k=top_k)

    predicted_classes = labels.numpy()
    predicted_classes += 1
    predicted_classes = [str(int) for int in predicted_classes] 

    return predictions, predicted_classes


def main():
    args = argparser()

    labels_data = args.category_names
    
    
    
    loaded_model = args.model
    
    model = tf.keras.models.load_model(loaded_model, custom_objects={"KerasLayer": hub.KerasLayer})

    img_path = args.imagepath
    probs, classes = predict(img_path, model, args.top_k)
    
    names = []

    if (labels_data != None):
        class_names = map_data(labels_data)
        for flower_number in classes:
            names.append(class_names.get(flower_number))
    else:
        for flower_number in classes:
            names.append(flower_number)   


    for i in range(len(classes)):
        print("The image is classified a {} with a probability of {}".format( names[i], probs[i]))

if __name__ == "__main__":
    main()