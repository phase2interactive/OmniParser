from modal import Image, App, web_endpoint, Volume
import modal
import os
from typing import Tuple
from logging import getLogger
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class ParseResponse(BaseModel):
    image: str
    parsed_content: str
    coordinates: str

web_app = FastAPI(title="OmniParser API", description="Vision-based GUI element parser")

LOGGER = getLogger(__name__)
# Define volume paths
VOLUME_ROOT = "/vol/models"
WEIGHTS_DIR = os.path.join(VOLUME_ROOT, "weights")
YOLO_DIR = os.path.join(WEIGHTS_DIR, "icon_detect/model.pt")
FLORENCE_DIR = os.path.join(WEIGHTS_DIR, "icon_caption_florence")

# Create persistent volume
volume = Volume.from_name("omniparser-vol",create_if_missing=True)

def init_ocr():
    from util.utils import get_caption_model_processor
    

# Create Modal image with dependencies
image = (
    Image.debian_slim()
        .apt_install("git", 
                    "wget", 
                    "libgl1-mesa-glx",  
                    "libglib2.0-0",  
                    "libsm6",  
                    "libxext6",  
                    "libxrender-dev")
        .pip_install_from_requirements("requirements.txt")
        .pip_install("fastapi[standard]", "openai", "huggingface_hub")
        .add_local_dir("util", "/root/util", copy=True)
        .run_function(init_ocr)
)
# Clone repo and download models
#image = image.run_commands(
 #   "git clone https://github.com/microsoft/OmniParser /root/OmniParser",
  #  "cd /root/OmniParser && mkdir -p weights/icon_detect weights/icon_caption_florence",
   # "cd /root/OmniParser && huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.pt --local-dir weights",
    #"cd /root/OmniParser && huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.yaml --local-dir weights",
    #"cd /root/OmniParser && huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/train_args.yaml --local-dir weights",
    #"cd /root/OmniParser && huggingface-cli download microsoft/Florence-2-base config.json --local-dir weights/icon_caption",
    #"cd /root/OmniParser && huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/generation_config.json --local-dir weights",
    #"cd /root/OmniParser && huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.safetensors --local-dir weights",
    #"cd /root/OmniParser && mv weights/icon_caption weights/icon_caption_florence",
#)

#image = image.env({"EASYOCR_MODULE_PATH": f"{WEIGHTS_DIR}/easyocr"})
app = App("omniparser", image=image)
@app.cls(
    image=image,
    volumes={VOLUME_ROOT: volume},  # Mount volume in the container
    gpu="T4",
    scaledown_window=60,
    allow_concurrent_inputs=10,
    #mounts=[modal.Mount.from_local_dir("util", remote_path="/root/util")]
)
class OmniparserService:
    def __init__(self):
        volume.reload()  # Ensure volume is up to date
    @modal.enter()
    def load_models(self):
        
        import subprocess
        output = subprocess.check_output(["nvidia-smi"], text=True)
        print(output)
        # print the contents of WEIGHTS_DIR
        LOGGER.info(os.listdir(WEIGHTS_DIR))
        #from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
        #import easyocr
        #reader = easyocr.Reader(['en'])
        #LOGGER.info(reader)
        #self.yolo_model = get_yolo_model(model_path=YOLO_DIR)
        #self.caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path=FLORENCE_DIR)
    @modal.method()
    def process_image(
        self,
        image_bytes: bytes,
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1
    ) -> Tuple[str, str, str]:
        import subprocess
        output = subprocess.check_output(["nvidia-smi"], text=True)
        print(output)
        assert "Driver Version" in output
        assert "CUDA Version" in output
        return None, None, None
    
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    parser_service = OmniparserService()
    @web_app.post("/parse", response_model=ParseResponse)
    async def parse_image(
        file: UploadFile = File(...),
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1
    ):
        LOGGER.info("Parsing image")
        """Parse an image to detect and caption GUI elements"""
        try:
            contents = await file.read()
            labeled_image, parsed_content, coordinates = await parser_service.process_image.remote.aio(
                contents,
                box_threshold,
                iou_threshold
            )
            return JSONResponse({
                "image": labeled_image,
                "parsed_content": parsed_content,
                "coordinates": coordinates
            })
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing image: {str(e)}"}
            )
    return web_app
import base64
if __name__ == "__main__":
    #init_omniparser()
    with open("imgs/demo_image.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    # post request to modal
    import requests
  #  response = requests.post(
   #     "https://phase2interactive-omniparse--omniparser-parse-image-dev.modal.run",
   #     json={"image_base64": image_base64},
   # )

    from util.utils import get_caption_model_processor
    
    params = {
        'box_threshold': 0.05,
        'iou_threshold': 0.1
    }
    response = requests.post(
        "https://phase2interactive-omniparse--omniparser-fastapi-app-dev.modal.run/parse",
        files={"file": ("demo_image.jpg", open("imgs/demo_image.jpg", "rb"), "image/jpeg")},
        params=params,
    )
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.text)