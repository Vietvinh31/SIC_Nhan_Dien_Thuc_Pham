"""Read-only smoke inference on existing datasets, not an accuracy evaluation."""
import importlib.util,json
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fv_app',ROOT/'website/app.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
out=ROOT/'reports/unified_ui';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'datasets/food_raw/source/test'
raw=a.WORK/'datasets/raw_5classes_v1/test'
def find_sample(base, target=None, multi=False):
    for label in sorted((base/'labels').glob('*.txt')):
        ids={int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
        if (multi and len(ids)>1) or (not multi and ids=={target}):
            for ext in ('.jpg','.png','.jpeg'):
                p=base/'images'/(label.stem+ext)
                if p.is_file(): return p
    return None
samples=[('fish',find_sample(raw,1)),('pork',find_sample(raw,3)),('shrimp',find_sample(raw,0)),
         ('tomato_OOD',find_sample(source,19)),('bell_pepper_OOD',find_sample(source,1)),
         ('multiple_objects',find_sample(raw,multi=True))]
results=[]
real_classifier=a.classify_crop
for name,path in samples+[('blank_synthetic',None)]:
    if not path and name!='blank_synthetic':
        results.append(dict(case=name,status='NOT RUN: no matching image'));continue
    image=Image.open(path).convert('RGB') if path else Image.new('RGB',(416,416),'white')
    calls=[]
    def traced(crop,cp):
        calls.append(str(cp));return real_classifier(crop,cp)
    a.classify_crop=traced
    try:
        annotated,boxes,rows=a.analyze_raw(image,.35)
        annotated.save(out/(name+'.jpg'))
        results.append(dict(case=name,path=str(path),boxes=boxes,rows=rows,classifier_calls=calls))
    except Exception as e:
        results.append(dict(case=name,error=str(e)))
    print(name,json.dumps(results[-1],ensure_ascii=True),flush=True)
a.classify_crop=real_classifier
food=ROOT/'modules/yolo_food/images/test image.png'
try:
    _,boxes=a.detect(Image.open(food).convert('RGB'),a.FOOD_MODEL,.35)
    results.append(dict(case='food_existing_demo',path=str(food),boxes=boxes))
except Exception as e: results.append(dict(case='food_existing_demo',error=str(e)))
(out/'inference_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
