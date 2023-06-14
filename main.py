from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import json
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)


@app.post("/recognize")
def recognize(file: UploadFile = File(...)):
    classes = {}
    # Modelden gelen sonuçları eşleştirmek için kullanılan dizi
    classes['memory'] = 'Ram'
    classes['cpu'] = 'İşlemci'
    classes['disk'] = "Disk"
    classes['gbic'] = 'Gbic'
    classes['nic'] = 'Nic'
    try:
        contents = file.file.read()
        extension = file.content_type
        if extension not in ['image/jpeg', 'image/png']:
            return {
                "status": "fail",
                "code": 403,
                "result": "Bulunamadı"
            }
        with open(file.filename, 'wb') as f:
            f.write(contents)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', device='cpu')
        results = model(file.filename)
        res = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

    except Exception:
        return {"message": "Dosya yüklenirken bir hata oluştu."}
    finally:
        file.file.close()
        os.remove(file.filename)
    try:
        return {
            "status": "success",
            "code": 200,
            "result": classes[res[0]['name']]
        }
    except Exception:
        return {
            "status": "fail",
            "code": 403,
            "result": "Bulunamadı"
        }


@app.get("/run-scrape")
def scrape(item):
    data = {
        "categoryUrls": [

            {
                "url": "https://www.amazon.com.tr/s?k=" + item + "&__mk_tr_TR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&ref"
                                                                 "=nb_sb_noss_1"
            }
        ],
        "debug": False,
        "detailedInformation": False,
        "maxItems": 6,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": [
                "RESIDENTIAL"
            ],
            "apifyProxyCountry": "TR"
        }
    }
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    x = requests.post('https://api.apify.com/v2/acts/junglee~free-amazon-product-scraper/run-sync-get-dataset-items'
                      '?token=apify_api_SncXLFAyXKQ5ANmy7mzMJ0vSWqQrSL2hd4cM', data=json.dumps(data), headers=headers)
    return json.loads(x.text)


@app.get("/get-scrape")
def get_scrape():
    x = requests.get(
        'https://api.apify.com/v2/acts/junglee~free-amazon-product-scraper/runs/last/dataset/items?token=apify_api_SncXLFAyXKQ5ANmy7mzMJ0vSWqQrSL2hd4cM')
    return {
        "status": "success",
        "code": 200,
        "result": json.loads(x.text)
    }
