import PySimpleGUI as sg
import cv2
from PIL import Image

layout = [[sg.Text('wybierz swoją pandę:')],
          [sg.Input(key='filename', enable_events=True), sg.FileBrowse(file_types=(("Pliki zdjęć", "*.jpg;*.jpeg;*.png;*.bmp"),))],
          [sg.Button('Wyświetl wybrane zdjecie',key="Wyświetl"), sg.Button('Wyjdź')],
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
    if event == 'Wyjdź':
        break

    if event is not None:
        continue

    if event in ('Exit'):
        break

window.close()
