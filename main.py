from fastapi import FastAPI, Request, Form, File, UploadFile # type: ignore

from fastapi.responses import HTMLResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore

from tensorflow.keras.models import load_model
import os

from PIL import Image
import numpy as np

# from app_main import Prediction

app = FastAPI()

# pdp = Prediction()


templates = Jinja2Templates(directory="User-Interface_And_Backend/templates")


@app.get("/", response_class=HTMLResponse)
async def upload_screen(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post("/upload/")
async def create_upload_files(files: list[UploadFile]):
    classes = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    for file in files:
        updated_img = preprocess_image(file.file)

        model_name = "potato_disease_model.h5"
        if (os.path.exists(model_name)):
            model = load_model("potato_disease_model.h5")
        
            predictions = model.predict(updated_img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            print(predicted_class)
            print("Predictions============= ", classes[predicted_class])

    
    return {"PRedicted class- ": classes[predicted_class]}
    
def preprocess_image(file_like):
    img = Image.open(file_like)

    img = img.resize((256, 256))
    img_array = np.array(img)
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array