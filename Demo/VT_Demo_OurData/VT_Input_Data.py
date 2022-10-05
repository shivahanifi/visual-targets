import yarp

yarp.Network.init()

class VTInputData(yarp.RFModule):
    #input port for rgb image
    self.in_port_scene_image = yarp.BufferedPortImageRgb()
    self.in_port_scene_image.open()