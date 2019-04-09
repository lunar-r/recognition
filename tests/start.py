class Animal:
    def __init__(self):
        self.name = "Animal"
        self.age = 0

    def eat(self, food):
        print(self.name + " eat " + food)


class Cat(Animal):
    def __init__(self):
        super(Cat, self).__init__()
        print(self.name)


if __name__ == '__main__':
    cat = Cat()
