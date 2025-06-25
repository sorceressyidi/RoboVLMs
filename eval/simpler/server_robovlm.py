import os
import torch
import numpy as np
import argparse
from simpler_env.evaluation.argparse import get_args
from eval.simpler.env_utlis import DictAction
from eval.simpler.model_wrapper import BaseModelInference

import argparse

import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat


import base64
import json
import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import json_numpy
json_numpy.patch()

import draccus
from dataclasses import dataclass
import uvicorn


# ------------------------------------------
# RoboVLM Inference Server
# ------------------------------------------
class RoboVLMServer:
    def __init__(self, model) -> None:
        logging.info("Initializing Octo model...")
        self.model = model
        self.task_description = "put the spoon on the towel"
        self.model.reset()  # Pre-warm model
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.post("/reset")(self.reset_model)
        self.app.post("/set_task")(self.set_task_model)
    
    async def set_task_model(self, request: Request):
        try:
            payload = await request.json()
            self.task_description = payload["task_description"]
            self.model.reset()  
            return JSONResponse(content={"status": "task updated"})
        except Exception as e:
            logging.error(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=500)
    async def reset_model(self):
        try:
            self.model.reset()
            return JSONResponse(content={"status": "model reset"})
        except Exception as e:
            logging.error(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=500)

    async def predict_action(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()  # <-- await here
            instruction = payload["instruction"]
            new_image = np.array(payload["image"], dtype=np.uint8)
            raw_action, action = self.model.step(new_image, instruction)
            return JSONResponse(content={
                "raw_action": {k: v.tolist() for k, v in raw_action.items()},
                "action": {k: v.tolist() for k, v in action.items()},
            })
        except Exception as e:
            logging.error(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=500)


    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        import uvicorn
        logging.info(f"🚀 Running RoboVLM server on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the model configuration')
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    config_path = args.config_path
    
    from robovlms.utils.config_utils import load_config
    eval_log_dir = os.path.dirname(ckpt_path)
    policy_setup = "widowx_bridge"
    configs = load_config(config_path)
    model = BaseModelInference(
        ckpt_path=ckpt_path,
        configs=configs,
        device=torch.device("cuda"),
        save_dir=eval_log_dir,
        policy_setup=policy_setup,
    )
    server = RoboVLMServer(model)
    server.run()