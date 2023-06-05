import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from matplotlib.patches import Rectangle

cfg = get_cfg()  # Yapılandırma Dosyasını Oluşturuyoruz
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))  # Yapılandırma Dosyasını Çeker ve Yapılandırma Dosyasına Ekler
cfg.DATASETS.TRAIN = ("my_train")  # Train Verilerimiz Yapılandırma Dosyasına Kaydeder
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2  # Çalışan Sayısı
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Ağırlıkları Çeker ve Yapılandırma Dosyasına Ekler
cfg.SOLVER.IMS_PER_BATCH = 4  # Batch Size
cfg.SOLVER.BASE_LR = 0.001  # Learning Rate (Öğrenme Oranı)
cfg.SOLVER.GAMMA = 0.05  # Learning Rate Azaltma Çarpımı
cfg.SOLVER.STEPS = [500]  # Learning Rate Azaltma Adım Sayısı
cfg.TEST.EVAL_PERIOD = 200  # Eğitim Sırasında Modeli Değerlendirmek İçin Adım Sayısı
cfg.SOLVER.MAX_ITER = 1000  # İterasyon Sayısı
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Sınıf Sayısı
