# feature visualization cube
uses python libraries vtk, lucid, tensorflow

# usage
Run image_cube.py to generate tensor representing a cube of pixels. It will save it to a file named featureCube64_7 64 is image_size 7 is version number.
Then you can see the result as 3d visualization by running render3d.py.
You can also run slices_gif.py to generate animation (uses imageio library) of intersetions with three different planes moving respectively along x,y and z axis
