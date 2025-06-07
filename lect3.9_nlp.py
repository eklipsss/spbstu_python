

with open('C:/Users/Максим/.fastai/data/human_numbers/train.txt') as f:
    lines += L(*f.readlines())

text = ' '.join([l.strip() for l in lines])
# print(text[:50])

tokens = text.split(' ')
# print(tokens[:10])

vocab = L(*tokens).unique()
# print(vocab[:30])

word2index = {w: i for i, w in enumerate(vocab)}
# print(word2index)

nums = L(word2index[i] for i in tokens)
# print(nums[:30])

# 1. Список из всех последовательностей из трех слов

seq = L((tokens[i:i + 3], tokens[i + 3]) for i in range(0, len(tokens) - 4, 3))
# print(seq[:10])

seq = L((nums[i:i + 3], nums[i + 3]) for i in range(0, len(nums) - 4, 3))
# print(seq[:10])

seq = L((tensor(nums[i:i + 3]), nums[i + 3]) for i in range(0, len(nums) - 4, 3))
# print(seq[:10])

bs = 64
cut = int(len(seq) * 0.8)
dls = DataLoaders.from_dsets(seq[:cut], seq[cut:], bs=bs, shuffle=False)

class Model1(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
    
    def forward(self, x):
        # h = F.relu(self.h_h(self.i_h(x[:, 0])))
        # h = h + self.i_h(x[:, 1])
        # h = F.relu(self.h_h(h)) # h2
        # h = h + self.i_h(x[:, 2])
        # h = F.relu(self.h_h(h))
        
        h = 0
        for i in range(3):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))
        return self.h_o(h)
# Код может быть записан в виде цикла => переиспользуем слои.
# Рекуррентная НС - результат рефакторинга, коэффициенты одни и те же.


if __name__ == '__main__':
    freeze_support()
    learn = Learner(dls, Model1(len(vocab), bs), loss_func=F.cross_entropy, metrics=accuracy)

    learn.fit_one_cycle(4, 0.001)

    n = 0
    counts = torch.zeros(len(vocab))
    for x, y in dls.valid:
        n += y.shape[0]
        for i in range_of(vocab):
            counts[i] += (y == i).long().sum()

    print(counts)

    index = torch.argmax(counts)

    print(index, vocab[index.item()], count[index].item() / n)