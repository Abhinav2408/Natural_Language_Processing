import torch
from torch import nn
from sklearn.metrics import f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_file(filename):
    with open(filename, "r") as file:
        text = file.readlines()
    return text

def process_text(text):
    X = []
    Y = []
    sentenceX = []
    sentenceY = []
    for line in text:
        split = line.split(" ")
        if len(split) > 1:
            sentenceX.append(split[0])
            sentenceY.append(split[1].replace("\n", ""))
        else:

            for i in range(len(sentenceX)):
                sentenceX[i] = sentenceX[i].lower()
                if sentenceX[i].replace('.','',1).isdigit():
                    sentenceX[i] = '<num>'
            X.append(sentenceX)
            Y.append(sentenceY)
            sentenceX = []
            sentenceY = []
    return X, Y

traintext = read_file("data/train.txt")
testtext = read_file("data/test.txt")
Xtrain, Ytrain = process_text(traintext)
Xtest,Ytest = process_text(testtext)


words_idx = {}
for sentence in Xtrain:
    for word in sentence:
        if word not in words_idx.keys():
            words_idx[word] = len(words_idx)

for sentence in Xtest:
    for word in sentence:
        if word not in words_idx.keys():
            words_idx[word] = len(words_idx)


tags_idx = {}

for tags in Ytrain:
    for tag in tags:
        if tag not in tags_idx.keys():
            tags_idx[tag] = len(tags_idx)

for tags in Ytest:
    for tag in tags:
        if tag not in tags_idx.keys():
            tags_idx[tag] = len(tags_idx)


def toidx(entity, map):
    return [map[i] for i in entity]

def get_scores(predY, trueY):
    trueY_O = [i for i, x in enumerate(trueY) if x == tags_idx["O"]]
    predY = [predY[i] for i in range(len(predY)) if i not in trueY_O]
    trueY = [trueY[i] for i in range(len(trueY)) if i not in trueY_O]
    print("Micro F1 score: ", f1_score(trueY, predY, average="micro"))
    print("Macro F1 score: ", f1_score(trueY, predY, average="macro"))
    print("Average F1 score: ", (f1_score(trueY, predY,average="micro") + f1_score(trueY, predY, average="macro")) / 2)

class Tagger(nn.Module):
    def __init__(self, embed_dim, hidden_states,vocablen,outlen ) -> None:
        super().__init__()

        self.hiddendim = hidden_states

        self.embed = nn.Embedding(vocablen,embed_dim)

        self.lstm = nn.LSTM(embed_dim,hidden_states,batch_first=True)

        self.tags = nn.Linear(hidden_states,outlen)

    def forward(self,sentence):

        embeds = self.embed(sentence)
        lstm_out,_ = self.lstm(embeds)

        tags = self.tags(lstm_out)

        tagprobs = nn.functional.softmax(tags,dim=1)
        return tags
    
EmbedDimension = 100
HiddenDimension = 200

model = Tagger(embed_dim=EmbedDimension, hidden_states=HiddenDimension, vocablen = len(words_idx.keys()),outlen = len(tags_idx.keys())).to(device)

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.3)


epochs = 3
for epoch in range(epochs):
    ### training
    train_loss = 0

    for x,y in zip(Xtrain,Ytrain):
        xval = toidx(x,words_idx)
        yval = toidx(y,tags_idx)
        xval = torch.tensor(xval,dtype=torch.long,device=device)
        yval = torch.tensor(yval,dtype=torch.long,device=device)

        ypred = model(xval)

        loss = lossfn(ypred,yval)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    

    train_loss /= len(Xtrain)


    with torch.no_grad():
        correcttags = 0
        totaltags = 0
        ypredforf1 = []
        ytrueforf1 = []
        for x,y in zip(Xtest,Ytest):
            xval = toidx(x,words_idx)
            yval = toidx(y,tags_idx)
            xval = torch.tensor(xval,dtype=torch.long,device=device)
            yval = torch.tensor(yval,dtype=torch.long,device=device)
            ypred = model(xval)
            ypred = ypred.argmax(dim=1)
            
            ypredforf1.extend(ypred.cpu())
            ytrueforf1.extend(yval.cpu())
            
            correcttags += torch.eq(ypred,yval).sum().item()
            totaltags += len(yval)

        acc = ((100.00*correcttags)/totaltags)
        print(f"Epoch: {epoch} | Loss : {train_loss:.5f} | TestAcc : {acc:.5f}")
        get_scores(ypredforf1,ytrueforf1)

            