from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

wf = Workflow()
yolov7 = wf.add_task(name="infer_yolo_v7", auto_connect=True)
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_fireman.jpg")
display(yolov7.get_image_with_graphics())