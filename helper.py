# from ultralytics import YOLO
# import time
# import streamlit as st
# import cv2
# import settings
# import threading

# def sleep_and_clear_success():
#     time.sleep(3)
#     st.session_state['recyclable_placeholder'].empty()
#     st.session_state['non_recyclable_placeholder'].empty()
#     st.session_state['hazardous_placeholder'].empty()

# def load_model(model_path):
#     model = YOLO(model_path)
#     return model

# def classify_waste_type(detected_items):
#     recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
#     non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
#     hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    
#     return recyclable_items, non_recyclable_items, hazardous_items

# def remove_dash_from_class_name(class_name):
#     return class_name.replace("_", " ")

# def _display_detected_frames(model, st_frame, image):
#     image = cv2.resize(image, (640, int(640*(9/16))))
    
#     if 'unique_classes' not in st.session_state:
#         st.session_state['unique_classes'] = set()

#     if 'recyclable_placeholder' not in st.session_state:
#         st.session_state['recyclable_placeholder'] = st.sidebar.empty()
#     if 'non_recyclable_placeholder' not in st.session_state:
#         st.session_state['non_recyclable_placeholder'] = st.sidebar.empty()
#     if 'hazardous_placeholder' not in st.session_state:
#         st.session_state['hazardous_placeholder'] = st.sidebar.empty()

#     if 'last_detection_time' not in st.session_state:
#         st.session_state['last_detection_time'] = 0

#     res = model.predict(image, conf=0.6)
#     names = model.names
#     detected_items = set()

#     for result in res:
#         new_classes = set([names[int(c)] for c in result.boxes.cls])
#         if new_classes != st.session_state['unique_classes']:
#             st.session_state['unique_classes'] = new_classes
#             st.session_state['recyclable_placeholder'].markdown('')
#             st.session_state['non_recyclable_placeholder'].markdown('')
#             st.session_state['hazardous_placeholder'].markdown('')
#             detected_items.update(st.session_state['unique_classes'])

#             recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)

#             if recyclable_items:
#                 detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in recyclable_items)
#                 st.session_state['recyclable_placeholder'].markdown(
#                     f"<div class='stRecyclable'>Recyclable items:\n\n- {detected_items_str}</div>",
#                     unsafe_allow_html=True
#                 )
#             if non_recyclable_items:
#                 detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in non_recyclable_items)
#                 st.session_state['non_recyclable_placeholder'].markdown(
#                     f"<div class='stNonRecyclable'>Non-Recyclable items:\n\n- {detected_items_str}</div>",
#                     unsafe_allow_html=True
#                 )
#             if hazardous_items:
#                 detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in hazardous_items)
#                 st.session_state['hazardous_placeholder'].markdown(
#                     f"<div class='stHazardous'>Hazardous items:\n\n- {detected_items_str}</div>",
#                     unsafe_allow_html=True
#                 )

#             threading.Thread(target=sleep_and_clear_success).start()
#             st.session_state['last_detection_time'] = time.time()

#     res_plotted = res[0].plot()
#     st_frame.image(res_plotted, channels="BGR")


# def play_webcam(model):
#     source_webcam = settings.WEBCAM_PATH
#     if st.button('Detect Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(source_webcam)
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(model,st_frame,image)
#                 else:
#                     vid_cap.release()
#                     break
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))


from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading

def load_model(model_path):
    """Loads the YOLO model from the given path."""
    model = YOLO(model_path)
    return model

def remove_dash_from_class_name(class_name):
    """Replaces underscores with spaces in class names."""
    return class_name.replace("_", " ")

def _display_detected_frames(model, st_frame, image):
    """
    Detects objects in the given image frame using the YOLO model and displays
    the detected object names in the Streamlit sidebar.
    """
    # Resize the image for processingpyth
    image = cv2.resize(image, (640, int(640 * (9 / 16))))
    
    # Initialize session state for unique classes
    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()

    # Run YOLO model prediction
    res = model.predict(image, conf=0.6)
    names = model.names  # Class names defined in the YOLO model
    detected_items = set()

    for result in res:
        # Extract class names from detection results
        new_classes = set([names[int(c)] for c in result.boxes.cls])
        if new_classes != st.session_state['unique_classes']:
            st.session_state['unique_classes'] = new_classes
            
            # Update the detected items
            detected_items.update(st.session_state['unique_classes'])
            
            # Display detected items in the sidebar
            detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in detected_items)
            st.sidebar.markdown(
                f"<div class='stDetected'>Detected Items:\n\n- {detected_items_str}</div>",
                unsafe_allow_html=True
            )

    # Plot the detection results on the frame and display in the main app
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR")

def play_webcam(model):
    """
    Captures video from the webcam and processes each frame to detect objects.
    Displays the results in real-time in the Streamlit app.
    """
    source_webcam = settings.WEBCAM_PATH  # Path to the webcam source
    if st.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()  # Placeholder for video frames
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

# # Main application
# if __name__ == '__main__':
#     st.title("Real-Time Waste Object Detection")
#     st.sidebar.title("Detected Items")
    
#     # Load YOLO model
#     model_path = "path_to_your_model.pt"  # Update this with your YOLO model file path
#     model = load_model(model_path)
    
#     # Start webcam object detection
#     play_webcam(model)
