---
layout: post
comments: true
title: Generating text in Dan Brown's style (Language Model with RNN)
---

RNNs (Recurrent Neural Networks) have been making some buzz for last 3-4 years. They have created some new benchmarks in learning sequential data that spans across multiple time steps like text and music. I can't really think of a better resource on understanding RNN than [Andrej Karpathy's blogpost](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

What I am going to be trying in this post is to have a language model trained on a book of Dan Brown. This exercise helps me solidify my knowledge around structuring RNN with Pytorch. Aside from the points that I will touch while putting the code, I leave it to the reader to gain in-depth knowledge on RNN as the internet is full of resources. My intention is to have an elaborate explanation of the code which I found missing in some of the existing implementations. So, let's get this started!!

<center><img src="http://karpathy.github.io/assets/rnn/charseq.jpeg"></center>
<center>Fig 1 <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/" target="_blank">source</a></center>

### Getting the dataset

<center><img src="https://static.rogerebert.com/uploads/movie/movie_poster/angels-and-demons-2009/large_grN4lETrHNejatQZFP2F2DAWMU4.jpg"></center>
<center><a href="https://www.rogerebert.com/reviews/angels-and-demons-2009" target="_blank">source</a></center>

The text input that I am going to be using here is a [book of Dan Brown](https://www.kaggle.com/bobita/sydneysheldon/data) made available through Kaggle datasets. We will work on making the network predict some sample text given a starting word.

As always, we start with the necessary imports.   

{% highlight  python %}
import torch
import time
import string
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
{% endhighlight %}

With Pytorch making our life easier with implementations of RNN, I find the most important and crucial part of code to be the preprocessing of our input. It took some time for me to wrap my head around how the network has to be fed with the text input in batches. It is probably much easier and straightforward to feed image data into dataloaders. We will go step-by-step over how I do this. There could be better ways of implementing this and I welcome your suggestions in the comments section. We create a LMDataset class that helps create corpus dictionaries and respective helper methods. I think it would be important to respect each of those methods with their own chunk of explanation. Following methods are a part of LMDataset class.

#### Create Corpus

{% highlight  python %}
def corpus(self):
        idx = 0
        with open(self.path,'r') as f:
            for line in f:
                words = line.split()
                words.append('<eos>')
                for word in words:
                    if word not in self.words2idx:
                        self.words2idx[word] = idx
                        self.idx2words[idx] = word
                        idx += 1
                    self.encodedtext.append(self.words2idx[word])
{% endhighlight %}

<em>corpus</em> method helps us map the words to dictionaries. Given a text we map each word to a numerical id and vice versa. The <em>encodedtext</em> list just adds numerical ids as per the respective word sequence in the text. Notice, we use end of the line - <em><eos></em> as a marker for our arbitrary length of a sentence. <em><eos></em> tag will help us create new lines when we sample our output from network model.

#### Encode and reshape text

{% highlight  python %}
def encodetext(self):
        sequence = len(self.encodedtext) // self.batch_size
        cropped_len = sequence * self.batch_size
        self.encodedtext = torch.LongTensor(self.encodedtext[:cropped_len]).view(self.batch_size,-1)
        return self.encodedtext
{% endhighlight %}

What we are trying to do here is to divide the id encoded text (encodetext python list) into a number of (batch_size) sequences of equal length (sequence). So, imagine the end result to be a matrix (encodedtext) with dimensions batch_size * sequence.

#### Number of sequences

{% highlight  python %}
def __len__(self):
        return (self.encodedtext.size(1) // self.sequence_len) - 1
{% endhighlight %}

One of the attributes of LMDataset class is sequence_len that determines the length of input sequence. Dividing the columns of encodetext with sequence_len will give us number of possible sequences to be fed to the network.

#### Input and Target

{% highlight  python %}
def __getitem__(self,index):
        i = index*self.sequence_len
        data = self.encodedtext[:,i:i+self.sequence_len]
        target = self.encodedtext[:,i+1:i+1+self.sequence_len]
        return data,target
{% endhighlight %}

Since LMDataset class inherits nn.utils.data.Dataset class, we have to implement __getitem__ to help deliver the input and target to our network. The idea here is to send a chunk of matrix (encodedtext) of size batch_size * sequence_len as input and the target would be of the same size but shifted one index towards right. For example, this would mean if there is a sentence "I live in Chicago and I love pets.", then the input would go in as "I live in Chicago and I love" while the target would go in as "live in Chicago and I love pets."

With the important methods in place, following is the full implementation of LMDataset class with few other supporting methods.

{% highlight  python %}
class LMDataset(Dataset):
    def __init__(self,path,sequence_len=30,batch_size=20):
        super(LMDataset,self).__init__()
        self.path = path
        self.words2idx = {}
        self.idx2words = {}
        self.encodedtext = []
        self.sequence_len = sequence_len
        self.batch_size = batch_size

        # preprocess text
        self.corpus()
        self.encodetext()

    def __len__(self):
        return (self.encodedtext.size(1) // self.sequence_len) - 1

    def __getitem__(self,index):
        i = index*self.sequence_len
        data = self.encodedtext[:,i:i+self.sequence_len]
        target = self.encodedtext[:,i+1:i+1+self.sequence_len]
        return data,target

    def corpus(self):
        idx = 0
        with open(self.path,'r') as f:
            for line in f:
                words = line.split()
                words.append('<eos>')
                for word in words:
                    if word not in self.words2idx:
                        self.words2idx[word] = idx
                        self.idx2words[idx] = word
                        idx += 1
                    self.encodedtext.append(self.words2idx[word])

    def wtoi(self):
        return self.words2idx

    def itow(self):
        return self.idx2words

    def encodetext(self):
        sequence = len(self.encodedtext) // self.batch_size
        cropped_len = sequence * self.batch_size
        self.encodedtext = torch.LongTensor(self.encodedtext[:cropped_len]).view(self.batch_size,-1)
        return self.encodedtext

    def vocablen(self):
        return len(self.words2idx)
{% endhighlight %}

### Network Architecture

The network that we use here has an embedding, LSTM and, a Linear block. We pass in a few hyperparameters to our network class, namely,

** vocab_size : size of vocabulary or len of wtoi dictionary.
** embed_size : size of embedding vector for each word id.
** hidden_size : size of hidden output vector.
** num_layers : number of layers of RNN blocks in the network.
** batch_size : number of sequences to be thrown as input in one batch.

{% highlight python %}
class LMRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, batch_size):
        super(LMRNN, self).__init__()

        # hyperparameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        #network layers
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.initiate_weights()

    def forward(self,x,hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x,hidden)
        out = out.contiguous().view(out.size(0)*out.size(1),out.size(2))   
        out = self.linear(out)  
        return out, hidden

    def initiate_weights(self):
        self.embed.weight.data.uniform_(-0.1,0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1,0.1)

    def initiate_hidden(self):
        return (Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)).cuda(),
               Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)).cuda())

    def detach(self, hidden):
        return [state.detach() for state in hidden]

    def sample(self, dataset, samples=100):
        # Set intial hidden ane memory states
        state = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).cuda(),
                 Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).cuda())

        # Select one word id randomly
        prob = torch.ones(self.vocab_size)
        input = Variable(torch.multinomial(prob, num_samples=1).unsqueeze(1),
                             volatile=True).cuda()

        sentence = []
        sentence.append(dataset.itow()[input.data[0][0]] + ' ')
        for i in range(samples):
            # Forward propagate rnn
            output, state = self.forward(input, state)

            # Sample a word id
            prob = output.squeeze().data.exp().cpu()
            word_id = torch.multinomial(prob, 1)[0]

            # Feed sampled word id to next time step
            input.data.fill_(word_id)

            # Write sentence
            word = dataset.itow()[word_id]
            if word == '<eos>':
                print(''.join(c for c in sentence))
                print()
                sentence = []
            else:
                word = word + ' '
                sentence.append(word)
{% endhighlight %}

#### Embedding block

The embedding module of Pytorch asks for a vocabulary size and dimensions of an embedding vector for each of the word id in the corpus. There are multiple ways one can derive embedding vectors for a word. Some of the popular techniques are Word2Vec and GloVe. Here, we are using a simple lookup techinque offered by Pytorch. The idea is very similar to one mentioned [here](https://stackoverflow.com/a/34877590) where a unique vector is generated for each word id.

#### LSTM blocks

Fig. 1 shows a simple character RNN where the network is trying to predict the next character in a given word. The key principle to understand here is that the output of hidden layer is fed back to hidden layer along with new input. The idea is to remember past inputs and predict future outputs based on relation between inputs at different time steps. One of the problems with the simplest version and character level RNNs is that they are not good at predicting future outputs based on a long sequence of inputs. [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) (Long short-term memory) networks counter this problem with a bunch of gates in the RNN unit. These gates basically control on what past input needs be remembered or forgotten for upcoming predictions. Following is probably one of the best resources to understand [how LSTM works](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). For our purpose, we use a 3-layered LSTM block that accepts an embedding vector and throws out a vector of hidden_size. The output of previous input is fed back into the LSTM block over the entire sequence of input.

#### Detach methods

The detach method of LMRNN class helps us detach the hidden layer weights from computation graph for gradient update after training a sequence. This is because we don't want to propogate gradients to hidden states from previous sequence.

### Loss, optimizer and training

We define a method <em>langmodel</em> that helps us create a dataloader based on LMDataset, a model with LMRNN, a CrossEntropyLoss criterion and an Adam optimizer. All of them are subsequently returned as a dictionary for a training method to use.

{% highlight python %}
def langmodel(path,embed_size,hidden_size,num_layers,batch_size,learning_rate):
    dataloader = DataLoader(dataset = LMDataset(path),batch_size=1,shuffle=True)
    model = LMRNN(dataloader.dataset.vocablen(), embed_size, hidden_size, num_layers, batch_size).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return {
        "dataloader": dataloader,
        "model" : model,
        "criterion" : criterion,
        "optimizer" : optimizer
    }
{% endhighlight %}

Further, we train the model for about 65 epochs to minimize our loss over the corpus. Notice, we clip the gradients to grad_clip value for backpropogation through time over every input sequence to avoid gradient explosion problem.

{% highlight python %}
def fitLM(path,embed_size=128,hidden_size=1024,num_layers=2,batch_size=20,learning_rate=0.001,grad_clip=0.5,epochs=10,save_as='lm'):
    rnn = langmodel(path,embed_size,hidden_size,num_layers,batch_size,learning_rate)

    since = time.time()
    losses = []
    for i in range(epochs):
        avg_loss = 0.0
        min_loss = 100
        hidden = rnn['model'].initiate_hidden()
        for data, target in rnn['dataloader']:
            data,target = Variable(data.view(data.size(1),data.size(2))).cuda(),Variable(target.view(target.size(1),target.size(2)).contiguous()).cuda()
            rnn['model'].zero_grad()
            hidden = rnn['model'].detach(hidden)
            out, hidden = rnn['model'](data,hidden)
            loss = rnn['criterion'](out,target.view(-1))
            avg_loss += loss.data[0]
            loss.backward()
            nn.utils.clip_grad_norm(rnn['model'].parameters(), grad_clip)
            rnn['optimizer'].step()
        avg_loss = avg_loss / len(rnn['dataloader'].dataset)
        losses.append(avg_loss)
        m,s = divmod(time.time() - since,60)
        print('Time {:.0f}m{:.0f}s Epoch {}/{} Loss {:.4f}'.format(m,s,i,epochs,avg_loss))
        print(20*'=','sample text',20*'=')

        if min_loss > avg_loss:
            torch.save(rnn['model'].state_dict(),save_as+'.pkl')
            min_loss = avg_loss

        avg_loss = 0.0
        rnn['model'].sample(rnn['dataloader'].dataset, samples=30)
        print(50*'=')

    rnn['model'].load_state_dict(torch.load(save_as+'.pkl'))
    return {
        "model": rnn['model'],
        "losses":losses,
        "dataset":rnn['dataloader'].dataset }
{% endhighlight %}

### Sample Predictions
{% highlight python %}
lm = fitLM(path,embed_size=256,hidden_size=1024,num_layers=3,epochs=65,learning_rate=0.001)
{% endhighlight %}

<center><img src="{{ site.baseurl }}/public/img/epoch1.png"></center>

<center><img src="{{ site.baseurl }}/public/img/epoch.png"></center>

{% highlight python %}
plt.ylabel('loss')
plt.xlabel('epochs')
plt.plot(lm['losses'])
plt.show()
{% endhighlight %}

<center><img src="{{ site.baseurl }}/public/img/lstmcurve.png"></center>

{% highlight python %}
lm['model'].sample(lm['dataset'],samples=100)
{% endhighlight %}

<center><img src="{{ site.baseurl }}/public/img/sample.png"></center>
