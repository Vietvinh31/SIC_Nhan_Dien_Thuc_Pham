import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fv_app', ROOT/'website/app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (100,100))
        self.calls = []
    def classifier(self, crop, path):
        self.calls.append((crop.size, path))
        return 'Fresh', .8
    def box(self, label, confidence=.9, bbox=None):
        return dict(label=label, confidence=confidence, bbox=bbox or [10,10,70,80])
    def test_meat_scope_is_unverified(self):
        self.assertEqual(app.VERIFIED_MEAT_CLASSES, frozenset())
        rows = app.route_detections(self.image,[self.box(n) for n in app.RAW_CLASSES if n!='raw_fish'],.35,classifier=self.classifier)
        self.assertFalse(self.calls)
        self.assertTrue(all(not r['Trạng thái'] for r in rows))
    def test_fish_cropped_and_yolo_first(self):
        events=[]
        def detector(*args):
            events.append('YOLO')
            return self.image,[self.box('raw_fish')]
        def classifier(crop,path):
            events.append('classifier')
            self.assertEqual(crop.size,(60,70))
            self.assertEqual(path,app.FISH_MODEL)
            return 'Fresh',.8
        with patch.object(app,'detect',detector),patch.object(app,'classify_crop',classifier):
            app.analyze_raw(self.image,.35)
        self.assertEqual(events,['YOLO','classifier'])
    def test_tomato_ood_empty_low_confidence(self):
        for boxes in ([],[self.box('tomato')],[self.box('chair')],[self.box('raw_fish',.2)]):
            app.route_detections(self.image,boxes,.35,classifier=self.classifier)
        self.assertFalse(self.calls)
    def test_multi_object_only_fish(self):
        rows=app.route_detections(self.image,[self.box('raw_fish'),self.box('raw_pork'),self.box('raw_shrimp')],.35,classifier=self.classifier)
        self.assertEqual(len(rows),3)
        self.assertEqual(len(self.calls),1)
    def test_invalid_crop_and_detection_only(self):
        for bbox in ([30,30,10,10],[0,0,3,3],[float('nan'),0,50,50]):
            app.route_detections(self.image,[self.box('raw_fish',bbox=bbox)],.35,classifier=self.classifier)
        app.route_detections(self.image,[self.box('raw_fish')],.35,False,self.classifier)
        self.assertFalse(self.calls)
    def test_missing_classifier_does_not_abort_other_objects(self):
        def bad(*args): raise FileNotFoundError('missing')
        rows=app.route_detections(self.image,[self.box('raw_fish'),self.box('raw_shrimp')],.35,classifier=bad)
        self.assertEqual(len(rows),2)
        self.assertIn('missing',rows[0]['Ghi chú'])
        self.assertFalse(rows[0]['Trạng thái'])
    def test_nutrition_scaling(self):
        result=app.nutrition_values(dict(calories=200,protein=10,carbs=20,fat=5),150)
        self.assertEqual(result,dict(calories=300,protein=15,carbs=30,fat=7.5))

class InterfaceTests(unittest.TestCase):
    def test_sections_and_nutrition(self):
        at=AppTest.from_file(str(ROOT/'website/app.py'),default_timeout=20).run()
        self.assertFalse(at.exception)
        for section in ['Nguyên liệu & trạng thái','Giới thiệu','Tính dinh dưỡng','Món ăn']:
            at.radio(key='section').set_value(section).run()
            self.assertFalse(at.exception, str(at.exception))
        at.radio(key='section').set_value('Tính dinh dưỡng').run()
        db=app.load_nutrition_db(str(app.NUTRITION_JSON))
        food=next(iter(db))
        at.multiselect(key='meal_foods').set_value([food]).run()
        at.number_input(key='meal_grams_'+food).set_value(150).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value, f"{db[food]['calories']*1.5:.1f} kcal")

if __name__=='__main__': unittest.main(verbosity=2)
