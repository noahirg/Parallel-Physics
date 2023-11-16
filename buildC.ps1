nvcc -c main.cpp --std c++17 -O3 Physics\libr\thread_object.cpp Physics\PhyCWorld.cu Physics\PhyCuda.cu -I.\SFML-2.6.0-M\include
nvcc main.obj thread_object.obj PhyCWorld.obj PhyCuda.obj  -o huh.exe -L.\SFML-2.6.0-M\lib -lsfml-graphics -lsfml-window -lsfml-system -lopengl32 -lwinmm -lgdi32
Remove-Item main.obj
Remove-Item PhyCWorld.obj
Remove-Item PhyCuda.obj
Remove-Item thread_object.obj