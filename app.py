from flask import Flask, render_template, request, url_for
import os
import cv2 as cv
import numpy as np
import base64
import string
from gtts import gTTS
from tensorflow.keras.models import load_model
from io import BytesIO
import re

app = Flask(__name__)

devnagarik_word = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'क', 'ब', 'भ', 'म', 'य',
                   'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ',
                   '०', '१', '२', '३', '४', '५', '६', '७', '८', '९']

trained_model = load_model('Handwritten_OCR.h5')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def up_load():
    return render_template('upload.html')

@app.route('/check')
def check_home():
    return render_template('check.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/photo', methods=['GET', 'POST'])
def upload_file():
    path = os.path.join(APP_ROOT, 'static/outimg')
    if not os.path.isdir(path):
        os.makedirs(path)

    imgToRemove = os.listdir(path)
    for i in imgToRemove:
        os.remove(os.path.join(path, i))

    target = os.path.join(APP_ROOT, 'static/images')
    if not os.path.isdir(target):
        os.makedirs(target)

    file = request.files['file']
    filename = file.filename
    destination = os.path.join(target, filename)
    file.save(destination)
    newDes = os.path.join('static/images', filename)
    readingimg = cv.imread(newDes)

    name = list(string.ascii_letters)
    word = preprocessing(readingimg)
    char = ""

    for i in range(len(word)):
        out_path = os.path.join("static/outimg", f"image-{name[i]}.jpg")
        cv.imwrite(out_path, word[i])

    total_count = len(word)
    probab = 0
    for count, i in enumerate(range(total_count), start=1):
        resize = cv.resize(word[i], (32, 32)) / 255.0
        reshaped = np.reshape(resize, (1, 32, 32, 1))

        prediction = trained_model.predict(reshaped)
        score_prediction = prediction > 0.5
        probab = str(np.amax(prediction))
        max_idx = score_prediction.argmax()
        predict_character = devnagarik_word[max_idx]
        char += predict_character

    final_char = ' '.join(re.findall(r'.{1,2}', char))
    tts = gTTS(final_char, lang='ne')
    speech_file = BytesIO()
    tts.write_to_fp(speech_file)
    speech_file.seek(0)
    speech_b64 = base64.b64encode(speech_file.read()).decode()

    return render_template('index.html', photos=newDes, result=final_char, probability=probab,
                           processedImg=url_for('static', filename='outimg/image-a.jpg'),
                           title='NepaliOCR - Predict', speech_b64=speech_b64)

def ROI(img):
    row, col = img.shape
    np_gray = np.array(img, np.uint8)
    one_row = np.zeros((1, col), np.uint8)

    images_location = []
    line_seg_img = np.array([])

    for r in range(row - 1):
        if np.equal(img[r:(r + 1)], one_row).all():
            if line_seg_img.size != 0:
                images_location.append(line_seg_img)
                line_seg_img = np.array([])
        else:
            if line_seg_img.size == 0:
                line_seg_img = np_gray[r:r + 1]
            else:
                line_seg_img = np.vstack((line_seg_img, np_gray[r + 1:r + 2]))

    if line_seg_img.size != 0:
        images_location.append(line_seg_img)

    return images_location

def preprocessing(img):
    img = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    image_area = img.shape[0] * img.shape[1]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(img_gray, (3, 3), 0)
    _, thresh_img = cv.threshold(gaussian, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    contours, _ = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if cv.contourArea(contour) < image_area * 0.0001:
            thresh_img[y:(y + h), x:(x + w)] = 0

    line_segmentation = ROI(thresh_img)
    each_word_segmentation = []

    for line in np.asarray(line_segmentation):
        word_segmentation = ROI(line.T)
        for i in word_segmentation:
            i = ROI(i.T)
            for words in np.asarray(i):
                each_word_segmentation.append(words)

    return each_word_segmentation

def dikka_remove(output):
    resultafterdikka = []
    each_character = []
    for i in range(len(output)):
        each = []
        main = output[i]
        r, inv3 = cv.threshold(main, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        ig = output[i]
        row, col = ig.shape

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
        detect_horizontal = cv.morphologyEx(ig, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts, _ = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)
            X, Y, w, h = cv.boundingRect(c)
            ig[0:Y + h + 2, 0:X + w].fill(0)

            r, inv1 = cv.threshold(ig, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            cnts1, _ = cv.findContours(inv1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for co in reversed(cnts1):
            if cv.contourArea(co) > 100:
                X, Y, w, h = cv.boundingRect(co)
                cv.rectangle(inv3, (X, 0), (X + w, Y + h), 255, 1)
                each.append((inv3[0:Y + h, X:X + w]))
        each_character.append(each)
        resultafterdikka.append(inv3)

    return resultafterdikka, each_character

@app.route('/send_pic', methods=['POST'])
def button_pressed():
    data_url = request.values['imgBase64']
    encoded_data = data_url.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    name = list(string.ascii_letters)
    word = preprocessing(img)
    char = ""

    for i in range(len(word)):
        out_path = os.path.join("static/outimg", f"image-{name[i]}.jpg")
        cv.imwrite(out_path, word[i])

    total_count = len(word)
    for count, i in enumerate(range(total_count), start=1):
        resize = cv.resize(word[i], (32, 32)) / 255.0
        reshaped = np.reshape(resize, (1, 32, 32, 1))

        prediction = trained_model.predict(reshaped)
        score_prediction = prediction > 0.5
        max_idx = score_prediction.argmax()
        predict_character = devnagarik_word[max_idx]
        char += predict_character

    final_char = ' '.join(re.findall(r'.{1,2}', char))
    return final_char

@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    path = os.path.join(APP_ROOT, "static/outimg")
    if not os.path.isdir(path):
        os.makedirs(path)

    willremoveimage = os.listdir(path)
    for i in willremoveimage:
        os.remove(os.path.join(path, i))

    target = os.path.join(APP_ROOT, 'static/images/')
    if not os.path.isdir(target):
        os.makedirs(target)

    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        destination = os.path.join(target, filename)
        file.save(destination)
        newDes = os.path.join('static/images/', filename)
        readingimg = cv.imread(newDes)

        def prediction(each_character):
            final_all_word = ""
            prob = 0
            ran = 0

            for word in each_character:
                final_word = ""
                for character in word:
                    resize = cv.resize(character, (32, 32)) / 255.0
                    resize = np.reshape(resize, (1, 32, 32, 1))
                    result = trained_model.predict(resize)
                    score_prediction = result > 0.5
                    ran = ran + 1
                    prob = prob + np.amax(result)
                    max_idx = score_prediction.argmax()
                    predict_character = devnagarik_word[max_idx]
                    final_word = final_word + predict_character
                final_all_word = final_all_word + ' ' + final_word  # Adding a space between words

            average_prob = str(prob / ran)
            return final_all_word.strip(), average_prob  # Stripping leading/trailing spaces

        word = preprocessing(readingimg)
        dikka_removed, each_character = dikka_remove(word)
        for i in range(len(dikka_removed)):
            out_path = os.path.join("static/outimg", f"image-{i}.jpg")
            cv.imwrite(out_path, dikka_removed[i])

        final_char, probab = prediction(each_character)

        tts = gTTS(final_char, lang='ne')
        speech_file = BytesIO()
        tts.write_to_fp(speech_file)
        speech_file.seek(0)
        speech_b64 = base64.b64encode(speech_file.read()).decode()

        return render_template('upload.html', photos=newDes, result=final_char, probability=probab,
                               processedImg=url_for('static', filename='outimg/image-a.jpg'),
                               title='NepaliOCR - Predict', speech_b64=speech_b64)

    return None

if __name__ == "__main__":
    app.run(debug=True)
