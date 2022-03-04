from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
from kivy.app import App

class CameraPreview(Image):
    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = cv2.VideoCapture(0)
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 30)

    #Drawing method to execute at intervals
    def update(self, dt):
        #Load frame
        ret, self.frame = self.capture.read()
        #Convert to Kivy Texture
        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture

from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import ObjectProperty

#Shoot button
class ImageButton(ButtonBehavior, Image):
    preview = ObjectProperty(None)

    #Execute when the button is pressed
    def on_press(self):
        cv2.namedWindow("CV2 Image")
        cv2.imshow("CV2 Image", self.preview.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class MyCameraApp(App):
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    MyCameraApp().run()