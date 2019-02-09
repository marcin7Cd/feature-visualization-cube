import os
import sys
import pylab
import glob
#I use pytorch only for upscaling
#import torch
#import torch.nn.functional as F
import vtk
import numpy as np

version =7
IMAGE_SIZE = 64
size = 64
assert size >= IMAGE_SIZE, "siza has to larger or equal than IMAGE_SIZE"

scale_factor = size/IMAGE_SIZE

def saturation(array):
    max_color = np.amax(array,3,keepdims=True)
    min_color = np.amin(array,3,keepdims=True)
    diff = max_color - min_color
    return np.nan_to_num(diff/max_color)

def intensity(array):
    return np.sum(array,3,keepdims=True)/3

def function(array):
    xs = np.tile(np.linspace(-1.0, 1.0, size),(size,size,1))
    ys = np.transpose(xs, [0,2,1])
    zs = np.transpose(xs, [2,0,1])
    return np.expand_dims(np.abs(2-np.sqrt(xs**2+ys**2+zs**2)),3)
matrix_full = np.load(f"featureCube{IMAGE_SIZE}_{version}.npy")[0]

if scale_factor>1:
    #works only with pytorch
    matrix_full = matrix_full.transpose(3,0,1,2) #b,c,a,d
    with torch.no_grad():
        matrix_full = F.upsample(torch.Tensor(matrix_full).unsqueeze(0),scale_factor = (4,4,4), mode="trilinear")
        matrix_full = matrix_full.squeeze(0).numpy().transpose(1,2,3,0)


matrix_full = matrix_full.reshape([size,size,size,3])
#adds aplha channel by default it is intesity value
matrix_full = np.concatenate([matrix_full, intensity(matrix_full) ],-1)

#rescales to 0-255 range
matrix_full *=255

print(matrix_full.shape)
cuting_option = 3
print(cuting_option&2)
#0 no cuts |1 - custom cuts | 2 - sphere | 3- custom_cuts and sphere 
if cuting_option&1 ==1 :
    d = size//10
    for i in range((size//2)):
        for j in range((size//2)):
            matrix_full[i:(i+1), j:(j+1), 0:((size//2)) ] = 0.0

    for i in range((size//2)-d,(size//2)+d):
        for j in range((size//2)-d,(size//2)+d):
            matrix_full[i:(i+1) ,j:(j+1), 0:size ] = 0.0
        
    for i in range((size//2)-d,(size//2)):
        for j in range((size//2)-d,(size//2)):
            matrix_full[i:(i+1) ,j:(j+1), 0:size ] = 0.0



    for i in range(0,size):
        for j in range(size-(d),size):
            matrix_full[i:(i+1) ,j:(j+1), ((size//2)-d):((size//2)+d) ] = 0.0

    for i in range(size-(d),size):
        for j in range(0,size):
            matrix_full[i:(i+1) ,j:(j+1), ((size//2)-d):((size//2)+d) ] = 0.0


    for i in range((size//2)-d,(size//2)+d):
        for j in range((size//2)-d,(size//2)+d):
            matrix_full[i:(i+1), j:(j+1),  0:size] = 0.0

    for i in range((size//2)-d,(size//2)+d):
        for j in range((size//2)-d,(size//2)+d):
            matrix_full[0:size ,i:(i+1) ,j:(j+1) ] = 0.0
if cuting_option & 2 ==2:
    for y in range(0,size):
        r1 = np.sqrt((size//2)**2 - ((size//2)-y)**2).astype('int')

        for z in range( (size//2)-r1-1,(size//2)+r1+1):
            r = np.sqrt( (r1)**2 -(size//2-z)**2).astype('int')
            
            matrix_full[ 0:(size//2-r) ,y:(y+1) ,z:(z+1) ] = 0.0
            matrix_full[ (size//2+r):size ,y:(y+1) ,z:(z+1) ] = 0.0
            
        matrix_full[0:size, y:(y+1), 0:(size//2-r1)] = 0.0
        matrix_full[0:size, y:(y+1),(size//2+r1):size] = 0.0


print(matrix_full.shape)
matrix_full = matrix_full.astype('uint8')

print("DONE")
# For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
# imports raw data and stores it.
dataImporter = vtk.vtkImageImport()
# The previously created array is converted to a string of chars and imported.
data_string = matrix_full.tostring()
dataImporter.CopyImportVoidPointer(data_string, len(data_string))
# The type of the newly imported data is set to unsigned char (uint8)
dataImporter.SetDataScalarTypeToUnsignedChar()
# Because the data that is RGBA-coded, the importer
# must be told this is the case.
dataImporter.SetNumberOfScalarComponents(4)
# The following two functions describe how the data is stored and the dimensions of the array it is stored in.
w, h, d = size,size,size
dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)

# Create the standard renderer, render window and interactor
FXAAopt = vtk.vtkFXAAOptions()
print(FXAAopt.__dict__)

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)


# Create transfer mapping scalar value to opacity
#
opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(0, 0.0)
opacityTransferFunction.AddPoint(121, 0.3)

opacityTransferFunction.AddPoint(255, 0.5)


# The property describes how the data will look
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.IndependentComponentsOff()
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetAmbient(0.5)
volumeProperty.SetDiffuse(0.3)
volumeProperty.SetSpecular(0.2)

linearInterpolation = True #False for more blocky structure 
if linearInterpolation:
    volumeProperty.SetInterpolationTypeToLinear()
else:
    volumeProperty.SetInterpolationTypeToNearest()

# The mapper / ray cast function know how to render the data
volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
volumeMapper.SetBlendModeToComposite()

#allows to read data
volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
# The volume holds the mapper and the property and
# can be used to position/orient the volume
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

ren.AddVolume(volume)
ren.SetBackground(0, 0, 0)

camera =  ren.GetActiveCamera()
c = volume.GetCenter()
camera.SetFocalPoint(c[0]+ d/(2*c[0]), c[1]+d/(2*c[1]), c[2]+d/(2*c[2]))
camera.SetPosition(c[0] + 200, c[1], c[2])

camera.SetViewUp(0, 0, -1)

renWin.SetSize(800, 800)
renWin.Render()

def CheckAbort(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)

renWin.AddObserver("AbortCheckEvent", CheckAbort)

iren.Initialize()
renWin.Render()
iren.Start()
