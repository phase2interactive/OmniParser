from modal import App, web_endpoint, Volume
import modal
import os
from typing import Tuple
from logging import getLogger
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
from tempfile import NamedTemporaryFile
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
    modal.Image.debian_slim()
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

app = App("omniparser", image=image)
@app.cls(
    image=image,
    volumes={VOLUME_ROOT: volume},  # Mount volume in the container
    gpu="A100-40GB",
    scaledown_window=60,
    allow_concurrent_inputs=10,
    #mounts=[modal.Mount.from_local_dir("util", remote_path="/root/util")]
)
class OmniparserService:
    def __init__(self):
        volume.reload()  # Ensure volume is up to date
    @modal.enter()
    def load_models(self):
        from util.utils import get_yolo_model, get_caption_model_processor
        self.yolo_model = get_yolo_model(model_path=YOLO_DIR)
        self.caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path=FLORENCE_DIR)
    
    @modal.method()
    def process_image(
        self,
        image_bytes: bytes,
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        use_paddleocr: bool = True,
        imgsz = 640,
    ) -> Tuple[str, str, str]:
        print('start processing')
        from util.utils import check_ocr_box, get_som_labeled_img
        
        image = Image.open(io.BytesIO(image_bytes))
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        # import pdb; pdb.set_trace()

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
        text, ocr_bbox = ocr_bbox_rslt
        # print('prompt:', prompt)
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        print('finish processing')
        parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    
        return dino_labled_img, parsed_content_list, str(label_coordinates)
    
    @modal.method()
    def process_image2(
        self,
        image_bytes: bytes,
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        use_paddleocr: bool = True,
        imgsz = 640,
    ) -> Tuple[str, str]:
        print('start processing')
        from util.utils import check_ocr_box, get_som_labeled_img

        image = Image.open(io.BytesIO(image_bytes))

        # Get OCR results
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image, display_img=False, output_bb_format='xyxy',
                                                        goal_filtering=None,
                                                        easyocr_args={'paragraph': False, 'text_threshold':0.9},
                                                        use_paddleocr=use_paddleocr)
        text, ocr_bbox = ocr_bbox_rslt

        # Skip drawing bounding boxes and just get coordinates
        _, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, self.yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=None,  # Set to None to skip drawing
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )

        parsed_content_list = '\n'.join([f'icon {i}: ' + str(v["content"]) for i,v in enumerate(parsed_content_list)])

        print('finish processing')
        return parsed_content_list, str(label_coordinates)
    


    @modal.method()
    def fast_process_image(
        self,
        image_bytes: bytes,
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        use_paddleocr: bool = False,
        imgsz = 640,
    ) -> str:
        print('start fast processing')
        from util.utils import check_ocr_box, predict_yolo
        from PIL import Image
        import io
        import torch
        import json

        # Open image
        print('reading image')
        image = Image.open(io.BytesIO(image_bytes))
        w, h = image.size

        # Get OCR results
        print('getting OCR results')
        ocr_bbox_rslt, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold':0.9},
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt

        # Get YOLO detections (boxes only)
        print('getting YOLO detections')
        boxes, conf, _ = predict_yolo(
            model=self.yolo_model,
            image=image,
            box_threshold=box_threshold,
            imgsz=imgsz,
            scale_img=False,
            iou_threshold=iou_threshold
        )

        # Convert to normalized coordinates (0-1 range)
        normalized_boxes = boxes / torch.Tensor([w, h, w, h]).to(boxes.device)

        # Format OCR boxes
        ocr_elements = [
            {
                'type': 'text',
                'bbox': box,
                'content': txt
            } for box, txt in zip(ocr_bbox, text)
        ]

        # Format YOLO boxes
        yolo_elements = [
            {
                'type': 'icon',
                'bbox': box.tolist(),
                'confidence': float(conf[i])
            } for i, box in enumerate(normalized_boxes)
        ]

        # Combine results
        result = {
            'ocr_elements': ocr_elements,
            'icon_elements': yolo_elements
        }

        print('finish fast processing')
        return result

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
        
    @web_app.post("/fast_parse")
    async def fast_parse_image(
        file: UploadFile = File(...),
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1
    ) -> JSONResponse:
        LOGGER.info("Fast parsing image")
        """Parse an image to detect GUI elements without generating an annotated image"""
        try:
            contents = await file.read()
            result = await parser_service.fast_process_image.remote.aio(
                contents,
                box_threshold,
                iou_threshold
            )
            return JSONResponse(result)
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