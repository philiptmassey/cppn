from model import CPPN, CPPNParameters

if __name__ == "__main__":
    params = CPPNParameters()
    model = CPPN(params)
    #model.save_gif("test.gif", 50)
    model.save_image("test.png")
