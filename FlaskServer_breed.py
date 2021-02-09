import flask
import werkzeug
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import time
import keras

breedModel = load_model("breedModel_150.h5py")

label_mapping = {1: 'Affenpinscher',
 2: 'Afghan_hound',
 3: 'Airedale_terrier',
 4: 'Akita',
 5: 'Alaskan_malamute',
 6: 'American_eskimo_dog',
 7: 'American_foxhound',
 8: 'American_staffordshire_terrier',
 9: 'American_water_spaniel',
 10: 'Anatolian_shepherd_dog',
 11: 'Australian_cattle_dog',
 12: 'Australian_shepherd',
 13: 'Australian_terrier',
 14: 'Basenji',
 15: 'Basset_hound',
 16: 'Beagle',
 17: 'Bearded_collie',
 18: 'Beauceron',
 19: 'Bedlington_terrier',
 20: 'Belgian_malinois',
 21: 'Belgian_sheepdog',
 22: 'Belgian_tervuren',
 23: 'Bernese_mountain_dog',
 24: 'Bichon_frise',
 25: 'Black_and_tan_coonhound',
 26: 'Black_russian_terrier',
 27: 'Bloodhound',
 28: 'Bluetick_coonhound',
 29: 'Border_collie',
 30: 'Border_terrier',
 31: 'Borzoi',
 32: 'Boston_terrier',
 33: 'Bouvier_des_flandres',
 34: 'Boxer',
 35: 'Boykin_spaniel',
 36: 'Briard',
 37: 'Brittany',
 38: 'Brussels_griffon',
 39: 'Bull_terrier',
 40: 'Bulldog',
 41: 'Bullmastiff',
 42: 'Cairn_terrier',
 43: 'Canaan_dog',
 44: 'Cane_corso',
 45: 'Cardigan_welsh_corgi',
 46: 'Cavalier_king_charles_spaniel',
 47: 'Chesapeake_bay_retriever',
 48: 'Chihuahua',
 49: 'Chinese_crested',
 50: 'Chinese_shar-pei',
 51: 'Chow_chow',
 52: 'Clumber_spaniel',
 53: 'Cocker_spaniel',
 54: 'Collie',
 55: 'Curly-coated_retriever',
 56: 'Dachshund',
 57: 'Dalmatian',
 58: 'Dandie_dinmont_terrier',
 59: 'Doberman_pinscher',
 60: 'Dogue_de_bordeaux',
 61: 'English_cocker_spaniel',
 62: 'English_setter',
 63: 'English_springer_spaniel',
 64: 'English_toy_spaniel',
 65: 'Entlebucher_mountain_dog',
 66: 'Field_spaniel',
 67: 'Finnish_spitz',
 68: 'Flat-coated_retriever',
 69: 'French_bulldog',
 70: 'German_pinscher',
 71: 'German_shepherd_dog',
 72: 'German_shorthaired_pointer',
 73: 'German_wirehaired_pointer',
 74: 'Giant_schnauzer',
 75: 'Glen_of_imaal_terrier',
 76: 'Golden_retriever',
 77: 'Gordon_setter',
 78: 'Great_dane',
 79: 'Great_pyrenees',
 80: 'Greater_swiss_mountain_dog',
 81: 'Greyhound',
 82: 'Havanese',
 83: 'Ibizan_hound',
 84: 'Icelandic_sheepdog',
 85: 'Irish_red_and_white_setter',
 86: 'Irish_setter',
 87: 'Irish_terrier',
 88: 'Irish_water_spaniel',
 89: 'Irish_wolfhound',
 90: 'Italian_greyhound',
 91: 'Japanese_chin',
 92: 'Keeshond',
 93: 'Kerry_blue_terrier',
 94: 'Komondor',
 95: 'Kuvasz',
 96: 'Labrador_retriever',
 97: 'Lakeland_terrier',
 98: 'Leonberger',
 99: 'Lhasa_apso',
 100: 'Lowchen',
 101: 'Maltese',
 102: 'Manchester_terrier',
 103: 'Mastiff',
 104: 'Miniature_schnauzer',
 105: 'Neapolitan_mastiff',
 106: 'Newfoundland',
 107: 'Norfolk_terrier',
 108: 'Norwegian_buhund',
 109: 'Norwegian_elkhound',
 110: 'Norwegian_lundehund',
 111: 'Norwich_terrier',
 112: 'Nova_scotia_duck_tolling_retriever',
 113: 'Old_english_sheepdog',
 114: 'Otterhound',
 115: 'Papillon',
 116: 'Parson_russell_terrier',
 117: 'Pekingese',
 118: 'Pembroke_welsh_corgi',
 119: 'Petit_basset_griffon_vendeen',
 120: 'Pharaoh_hound',
 121: 'Plott',
 122: 'Pointer',
 123: 'Pomeranian',
 124: 'Poodle',
 125: 'Portuguese_water_dog',
 126: 'Saint_bernard',
 127: 'Silky_terrier',
 128: 'Smooth_fox_terrier',
 129: 'Tibetan_mastiff',
 130: 'Welsh_springer_spaniel',
 131: 'Wirehaired_pointing_griffon',
 132: 'Xoloitzcuintli',
 133: 'Yorkshire_terrier'}
 
app = flask.Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    # to extract file name
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name: " + imagefile.filename)
    imagefile.save(filename)
        

    imageML = load_img('androidFlask.jpg', color_mode='rgb', target_size= (224,224))
    
    imageArray = img_to_array(imageML)
    imageArray = np.expand_dims(imageArray, axis = 0)
    imageArray = imageArray/255.0
    
    
    prediction = breedModel.predict(imageArray)
    predicted_class = np.argmax(np.round(prediction), axis = 1)
    

    result = str(label_mapping[predicted_class[0]])    
    result = result.replace("_", " ")

    return result

@app.route('/predict/', methods=['GET', 'POST'])
def handle_request_web():
    imagefile = flask.request.files['image']
    # to extract file name
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name: " + imagefile.filename)
    imagefile.save(filename)
    
    im = keras.preprocessing.image.load_img('androidFlask.jpg', color_mode='rgb', target_size=(224,224)) # -> PIL image
    doc = keras.preprocessing.image.img_to_array(im) # -> numpy array
    doc = np.expand_dims(doc, axis=0)
    doc = doc/255.0
        #display(keras.preprocessing.image.array_to_img(doc[0]))

    # make a prediction of dog_breed based on image
    prediction = breedModel.predict(doc)[0]
    dog_breed_indexes = prediction.argsort()[-5:][::-1]
    probabilities = sorted(prediction, reverse=True)[:5]
        
    output = ""

    for i in range(5):
        output += "This dog looks like a {} with probability {:.5f}.".format(label_mapping[dog_breed_indexes[i]+1], probabilities[i])
        output += "&"
    
    #time.sleep(30)
    return output

if __name__ == '__main__':
    app.run(debug=True)
#app.run(host="0.0.0.0", port=5000, debug=True)
# set host to 0.0.0.0 to use the current IPv4 address
# True -> server restarts itself whenever source code changes