import random

blue=[]
red =[]

n=10

x1=5
y1=5

class Seperator:
    def __init__(self,cx,cy,a):
        self.cx=cx
        self.cy=cy
        self.a=a
    def __call__(self,x,y):
        if self.cx*x+self.cy*y>self.a:
            return True
        return False

seperator=Seperator(1.,1.,0.)
    
while len(blue)<10:
    x=random.uniform(-x1, x1)
    y=random.uniform(-y1, y1)
    if seperator(x,y):
        blue.append([x,y])

while len(red)<10:
    x=random.uniform(-x1, x1)
    y=random.uniform(-y1, y1)
    if not seperator(x,y):
        red.append([x,y])


for b in blue:
    print(f"\\node[draw, circle, fill=blue] at ({b[0]},{b[1]}) {{}};")


for r in red:
    print(f"\\node[draw, circle, fill=red] at ({r[0]},{r[1]}) {{}};")
        
    
