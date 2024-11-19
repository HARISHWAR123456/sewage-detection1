# sewage block detection
This project demonstrates waste detection using a YOLOv8 (You Only Look Once) object detection model items in a webcam stream.

Our datasets used to train:
https://universe.roboflow.com/ai-project-i3wje/waste-detection-vqkjo/model/3
## Project Structure

- `app.py`: Main application file containing Streamlit code.
- `helper.py`: Helper functions for waste detection using the YOLO model.
- `settings.py`: Configuration settings, including the path to the YOLO model and waste types.
- `train.py`: To train the model

## Classifying Waste Items

'cardboard_box','can','plastic_bottle_cap','plastic_bottle','reuseable_paper'
['plastic_bag','scrap_paper','stick','plastic_cup','snack_bag','plastic_box','straw','plastic_cup_lid','scrap_plastic','cardboard_bowl','plastic_cultery']
['battery','chemical_spray_can','chemical_plastic_bottle','chemical_plastic_gallon','light_bulb','paint_bucket']






- [Streamlit Documentation](https://docs.streamlit.io/)
- [YOLO Documentation](https://github.com/ultralytics/yolov5)

