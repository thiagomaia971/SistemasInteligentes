class Log():
    def __init__(self):
        f = open("log.txt","w+")
        f.write('')
        f.close()

    def print(self, text):
        f = open("log.txt","a+")
        
        print(text)
        f.write(f'{text}\n')

        f.close()

    def printWeights(self, label, weights):
        arr = "\n".join(map(str, weights))
        self.print(f'{label}\n{arr}\n')