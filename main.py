import torch
import json
classes = {}
classes['memory'] = 'Ram'
classes['cpu'] = 'CPU'
classes['disk'] = 'SSD'
classes['gbic'] = 'Gbic'
classes['nic'] = 'Nic'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', device='cpu')
img = 'images/image.jpg'
results = model(img)
# results.save(save_dir='results')
res = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
print(classes[res[0]['name']])
