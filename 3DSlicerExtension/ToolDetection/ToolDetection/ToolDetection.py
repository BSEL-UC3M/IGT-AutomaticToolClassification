import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import scipy.ndimage
from tensorflow.keras.models import load_model
import cv2

#
# ToolDetection
#
class ToolDetection(ScriptedLoadableModule):
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ToolDetection" 
    self.parent.categories = ["ToolDetection"]
    self.parent.dependencies = []
    self.parent.contributors = ["David Garcia Mato (Universidad Carlos III de Madrid, Spain) and Tamas Ungi (Queens University, Canada)"] 
    self.parent.helpText = """
    Scripted module to classify images in real-time using a selected model for the prediction.
    """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
    This file was originally developed by David Garcia Mato (Universidad Carlos III de Madrid, Spain) and Tamas Ungi (Queens University, Canada)
    """ 

#
# ToolDetectionWidget
#
class ToolDetectionWidget(ScriptedLoadableModuleWidget):
  
  def setup(self):
    self.logic = ToolDetectionLogic()
    
    ScriptedLoadableModuleWidget.setup(self)
    
    self.detectionOn = False
    
    self.updateTimer = qt.QTimer()
    self.updateTimer.setInterval(100)
    self.updateTimer.setSingleShot(True)
    self.updateTimer.connect('timeout()', self.onUpdateTimer)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLVectorVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    self.modelPathEdit = ctk.ctkPathLineEdit()
    parametersFormLayout.addRow("Keras model: ", self.modelPathEdit)
    
    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.05
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 1.0
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for class probability.")
    parametersFormLayout.addRow("Prediction threshold", self.imageThresholdSliderWidget)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Start detection")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)
    
    #
    # Output label
    #
    self.classLabel = qt.QLabel("-")
    classFont = self.classLabel.font
    classFont.setPointSize(32)
    self.classLabel.setFont(classFont)
    parametersFormLayout.addRow(self.classLabel)
    
    # connections
    self.imageThresholdSliderWidget.connect("valueChanged(double)", self.onImageThresholdValueChanged)
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    
    # Add vertical spacer
    self.layout.addStretch(1)


  def onUpdateTimer(self):
    if self.detectionOn:
      [classText, classProbabilityText] = self.logic.getLastClass()
      newText = classText + ' (' + classProbabilityText + ')'
      self.classLabel.setText(newText)
      self.updateTimer.start()
    else:
      self.classLabel.setText("")
    
  def onImageThresholdValueChanged(self, value):
    self.logic.predictionThreshold = self.imageThresholdSliderWidget.value
    
  def setDetection(self, currentState):
    self.detectionOn = currentState
    if self.detectionOn is True:
      self.applyButton.setText("Stop detection")
    else:
      self.applyButton.setText("Start detection")
    
  def onApplyButton(self):
    imageThreshold = self.imageThresholdSliderWidget.value
    modelFilePath = self.modelPathEdit.currentPath
    
    # Try to load Keras model
    success = self.logic.loadKerasModel(modelFilePath)
    if not success:
      logging.error("Failed to load Keras model: {}".format(modelFilePath))
      self.setDetection(False)
      return
    
    inputVolumeNode = self.inputSelector.currentNode()
    if inputVolumeNode is None:
      logging.error("Please select a valid image node!")
      self.setDetection(False)
      return
    
    success = self.logic.run(inputVolumeNode, imageThreshold)
    if not success:
      logging.error("Could not start classification!")
      self.setDetection(False)
      return
    
    if self.detectionOn is True:
      self.setDetection(False)
      return
    else:
      self.setDetection(True)
      self.updateTimer.start()
    

#
# ToolDetectionLogic
#
class ToolDetectionLogic(ScriptedLoadableModuleLogic):
  
  def __init__(self):
    self.model = None
    self.observerTag = None
    self.lastObservedVolumeId = None
    self.lastClass = "-"
    self.lastClassProbability = "0.0"
    self.model_input_size = None
    self.classes = ['1', '2', '3', '4', 'None']
    self.predictionThreshold = 0.0
  
  def getLastClass(self):
    return [self.lastClass, self.lastClassProbability]
  
  def loadKerasModel(self, modelFilePath):
    """
    Tries to load Keras model for classifiation
    :param modelFilePath: full path to saved model file
    :return: True on success, False on error
    """
    try:
      self.model = load_model(modelFilePath, compile = False)
    except:
      self.model = None
      return False
    
    return True

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def run(self, inputVolumeNode, imageThreshold):
    """
    Run the classification algorithm on each new image
    """
    
    if self.model is None:
      logging.error('Cannot run classification without model!')
      return False
    
    self.predictionThreshold = imageThreshold
    
    image = inputVolumeNode.GetImageData()
    shape = list(image.GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)

    input_size = self.model.layers[0].input_shape[0][1]
    if input_size == None:
      self.model_input_size = 256
    else:
      self.model_input_size = input_size
    print('Model input size: ', self.model_input_size)
    
    if self.observerTag is None:
      self.lastObservedVolumeId = inputVolumeNode.GetID()
      self.observerTag = inputVolumeNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onImageModified)
      logging.info('Processing started')
    else:
      lastVolumeNode = slicer.util.getNode(self.lastObservedVolumeId)
      if lastVolumeNode is not None:
        lastVolumeNode.RemoveObserver(self.observerTag)
        self.observerTag = None
        self.lastObservedVolumeId = None
      logging.info('Processing ended')
    
    return True

  def preprocess_input(self, x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
  
  def onImageModified(self, caller, event):
    image_node = slicer.util.getNode(self.lastObservedVolumeId)
    image = image_node.GetImageData()
    shape = list(image.GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    input_array = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
    
    # Resize image and scale between 0.0 and 1.0
    resized_input_array = cv2.resize(input_array, (self.model_input_size, self.model_input_size))
    resized_input_array = self.preprocess_input(resized_input_array.astype('float'))
    #resized_input_array = resized_input_array / (resized_input_array.max())
    resized_input_array = np.expand_dims(resized_input_array, axis=0)
    #print('Resized input array: ', resized_input_array)
    
    # Run prediction and print result
    prediction = self.model.predict(resized_input_array)
    maxPredictionIndex = prediction.argmax()
    maxPrediction = prediction[0, maxPredictionIndex]
    
    if maxPrediction > self.predictionThreshold:
      self.lastClass = self.classes[maxPredictionIndex]
      self.lastClassProbability = "{:2.2%}".format(maxPrediction)
    else:
      self.lastClass = "None"      
      self.lastClassProbability = "-"

    print("Prediction: {} at {:2.2%} probability".format(self.lastClass, maxPrediction))


