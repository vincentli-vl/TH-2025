import cv2

def find_available_cameras():
    """Tests camera indices 0 through 9 to see which are available."""
    print("Searching for available cameras...")
    for i in range(10):
        # Try to open the camera with the DirectShow backend
        cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera found at index: {i}")
            cap.release()
        else:
            # Optional: uncomment below to see all failures
            # print(f"-> No camera at index: {i}")
            pass
    print("Search complete.")

if __name__ == '__main__':
    find_available_cameras()