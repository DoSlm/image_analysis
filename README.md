GUITTARD Mehdi 
SALMI Dorian

# Region Growing

The objective of this project is to implement a region growing algorithme to segment images.

## Compilation
To compile this project, you need the following:
- OpenCV version 4.0 or more
- cmake version 3.21 or more

Then compile with the following lines, from the root of this project:
- cd region-growing
- cmake CMakeLists.txt
- make

Done !

## Running the program
When running the program, you need to give the path to the image you want to process as parameter, as such:
- ./region_growing the_path_to_your_image

The path can either be relative to the executable's position or absolute.

## Best-of

- ./region_growing Images/Cathedrale-Lyon.jpg (Growing: 18, Fusion 17)
- ./region_growing Images/Jotaro.jpeg (Growing: 10, Fusion 10)
- ./region_growing Images/Patineur-de-Cesar-Lyon.jpg (Growing: 25, Fusion 19)
