g++ -c main.cpp -O3 render.cpp Physics\libr\thread_object.cpp Physics\libr\thread_pool.cpp Physics\PhyWorld.cpp Physics\Body.cpp Physics\Circle.cpp -I.\SFML-2.6.0\include
g++ main.o render.o thread_object.o thread_pool.o PhyWorld.o Body.o Circle.o -o huh.exe -L.\SFML-2.6.0\lib -lsfml-graphics -lsfml-window -lsfml-system -lopengl32 -lwinmm -lgdi32
Remove-Item main.o
Remove-Item PhyWorld.o
Remove-Item Circle.o
Remove-Item render.o
Remove-Item Body.o
Remove-Item thread_object.o
Remove-Item thread_pool.o