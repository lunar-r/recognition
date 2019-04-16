import copy


class Anmial():
    def __init__(self):
        self.name = "Animal"
        self.type = "Dynamic"
        self.names = ["an", "be"]

    def print(self):
        print("the id of list in class : " + str(id(self.names)))


a = Anmial()
a.print()

names = a.names
print("the id of list out class : " + str(id(names)))
others = []
print("the id of new list out class : " + str(id(others)))
others.append("one")
others.append("two")

names.clear()
for i in range(len(others)):
    names.append(others[i])


print("the id of list out class at the end: " + str(id(names)))

print(names[0])