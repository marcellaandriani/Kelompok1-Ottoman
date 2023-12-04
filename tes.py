from ultralytics import YOLO
from PIL import Image

model = YOLO('model/best.pt')

path_gambar = 'tembokretak1.jpg'

prediksi_gambar = model(path_gambar, conf=0.8)

for r in prediksi_gambar:
    im_arr = r.plot()
    im = Image.fromarray(im_arr[..., ::-1])
    im.show()