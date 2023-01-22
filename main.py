import PySimpleGUI as sg
import cv2
from PIL import Image
from scripts.pandalizator import Pandalizator
from scripts.proceed_dataset import proceed_image
from scripts.evaluate_model import evaluate_model, fit_model
import numpy as np

pandalizator = Pandalizator()
pandalizator.read_model_from_file()
# pandalizator = fit_model(pandalizator)
# evaluate_model(pandalizator)



layout = [[sg.Text('wybierz swoją pandę:')],
          [sg.Input(key='filename', enable_events=True), sg.FileBrowse(file_types=(("Pliki zdjęć", "*.jpg;*.jpeg;*.png;*.bmp"),))],
          [sg.Button('Wyświetl wybrane zdjecie',key="Wyświetl"), sg.Button('Wyjdź'), sg.Button('Sprawdz')],
          [sg.Image(key='image', visible=False)],
          [sg.Output(size=(80,20))]]

window = sg.Window('Pandalizator', layout)

while True:
    event, values = window.read()

    if event == 'Wyświetl':
        if values['filename']:
            try:
                img = Image.open(values['filename'])
                img.show()
            except:
                sg.popup_error("Nieprawidłowy format pliku lub plik uszkodzony")
    if event == 'Sprawdz':
        img = np.array(Image.open(values['filename']))
        response = 'Panda :)' if pandalizator.predict(proceed_image(img)) > 0 else 'Not panda :('
        print(response)
    if event == 'Wyjdź':
        break

    if event is not None:
        continue

    if event in ('Exit'):
        break

window.close()
