import taichi as ti

@ti.data_oriented
class Pair :
    def __init__ (self , curr, next):
        self.curr = curr
        self.next = next
    
    def swap(self):
        self.curr , self.next = self.next , self.curr

