# Generation of Image captioning using CNN and RNN
# Troy Cope
# Programming Assignment-02-CS4732-Troy Cope
import json
from torch_snippets import *

# Assigning variables and giving the images / captions list
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('open_images_train_captions.jsonl', 'r') as json_file:
    json_list = json_file.read().split('\n')

# filling out the data list with random choices
np.random.shuffle(json_list)
data = []

N = 100

# This will fetch the info for N images
for ix, json_str in Tqdm(enumerate(json_list), N):

    if ix == N: break

    try:
        result = json.loads(json_str)

        x = pd.DataFrame.from_dict(result, orient='index').T

        data.append(x)

    except:
        pass

# randomizing the set and segementing into two different sets
# this is our data -> training, validation
np.random.seed(10)

data = pd.concat(data)

data['train'] = np.random.choice([True, False], size=len(data), p=[0.95, 0.05])

data.to_csv('data.csv', index=False)

from openimages.download import _download_images_by_id

# !mkdir -p train-images val-images

# Now downloading the images necessary via JSON & ID
subset_imageIds = data[data['train']].image_id.tolist()

_download_images_by_id(subset_imageIds, 'train', './train-images/')

#
subset_imageIds = data[~data['train']].image_id.tolist()

_download_images_by_id(subset_imageIds, 'train', './val-images/')

# Creating all the data for a vocabulary
# This works by assigning each caption an unique integer
# and assigning them into counters

from collections import defaultdict
from torchtext.legacy.data import Field

#
captions = Field(sequential=False, init_token='<start>', eos_token='<end>')

all_captions = data[data['train']]['caption'].tolist()

all_tokens = [[w.lower() for w in c.split()] for c in all_captions]

all_tokens = [w for sublist in all_tokens for w in sublist]

captions.build_vocab(all_tokens)


# this section is making up for the lack of tokens
# in order to make a data set which can match strings
# and their integer mappings
class Vocab: pass


vocab = Vocab()

captions.vocab.itos.insert(0, '<pad>')

vocab.itos = captions.vocab.itos

vocab.stoi = defaultdict(lambda: captions.vocab.itos.index('<unk>'))

vocab.stoi['<pad>'] = 0

for s, i in captions.vocab.stoi.items():
    vocab.stoi[s] = i + 1

# this is going to give the dataframe, images
# and other important info for methods
from torchvision import transforms


class CaptioningData(Dataset):

    def __init__(self, root, df, vocab):
        self.df = df.reset_index(drop=True)

        self.root = root

        self.vocab = vocab

        self.transform = transforms.Compose([

            transforms.Resize(224),

            transforms.RandomCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize((0.485, 0.456, 0.406),

                                 (0.229, 0.224, 0.225))]

        )

    # image and caption are fetched
    # target is converted into word ID's
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        row = self.df.iloc[index].squeeze()

        id = row.image_id

        image_path = f'{self.root}/{id}.jpg'

        image = Image.open(os.path.join(image_path)) \
            .convert('RGB')

        caption = row.caption

        tokens = str(caption).lower().split()

        target = []

        target.append(vocab.stoi['<start>'])

        target.extend([vocab.stoi[token] for token in tokens])

        target.append(vocab.stoi['<end>'])

        target = torch.Tensor(target).long()

        return image, target, caption

    # choses a random int
    def choose(self):
        return self[np.random.randint(len(self))]

    # obviously returns the length of itself
    def __len__(self):
        return len(self.df)

    # finds the max length and pads rest of the data
    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)

        images, targets, captions = zip(*data)

        images = torch.stack([self.transform(image) for image in images], 0)

        lengths = [len(tar) for tar in targets]

        _targets = torch.zeros(len(captions), max(lengths)).long()

        for i, tar in enumerate(targets):
            end = lengths[i]

            _targets[i, :end] = tar[:end]

        return images.to(device), _targets.to(device), torch.tensor(lengths).long().to(device)


# Building a training and validation set
trn_ds = CaptioningData('train-images', data[data['train']], vocab)

val_ds = CaptioningData('val-images', data[~data['train']], vocab)
image, target, caption = trn_ds.choose()

show(image, title=caption, sz=5)
print(target)

# Creating loaders for each data set
trn_dl = DataLoader(trn_ds, 32, collate_fn=trn_ds.collate_fn)

val_dl = DataLoader(val_ds, 32, collate_fn=val_ds.collate_fn)

inspect(*next(iter(trn_dl)), names='images,targets,lengths')

#

from torch.nn.utils.rnn import pack_padded_sequence

from torchvision import models


# Building a CNN through encoders
class EncoderCNN(nn.Module):

    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace

        top fc layer."""

        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)

        # delete the last fc layer.

        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""

        with torch.no_grad():
            features = self.resnet(images)

        features = features.reshape(features.size(0), -1)

        features = self.bn(self.linear(features))

        return features


image, target, caption = trn_ds.choose()

show(image, title=caption, sz=5)
print(target)

# creating a summary of the encoder class
encoder = EncoderCNN(256).to(device)


# !pip install torch_summary


print(summary(encoder, torch.zeros(32, 3, 224, 224).to(device)))


# Building a decoder
# this decodes the CNN in a manageable way,
# using the word as the hidden stage of each step
class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=80):
        """Set the hyper-parameters and build the layers."""

        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)

        self.max_seq_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and

        generates captions."""

        embeddings = self.embed(captions)

        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True)

        outputs, _ = self.lstm(packed)

        outputs = self.linear(outputs[0])

        return outputs

    def predict(self, features, states=None):

        """Generate captions for given image

        features using greedy search."""

        sampled_ids = []

        inputs = features.unsqueeze(1)

        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)

            # hiddens: (batch_size, 1, hidden_size)

            outputs = self.linear(hiddens.squeeze(1))

            # outputs: (batch_size, vocab_size)

            _, predicted = outputs.max(1)

            # predicted: (batch_size)

            sampled_ids.append(predicted)

            inputs = self.embed(predicted)

            # inputs: (batch_size, embed_size)

            inputs = inputs.unsqueeze(1)

            # inputs: (batch_size, 1, embed_size)

        sampled_ids = torch.stack(sampled_ids, 1)

        # sampled_ids: (batch_size, max_seq_length)

        # convert predicted tokens to strings

        sentences = []

        for sampled_id in sampled_ids:

            sampled_id = sampled_id.cpu().numpy()

            sampled_caption = []

            for word_id in sampled_id:

                word = vocab.itos[word_id]

                sampled_caption.append(word)

                if word == '<end>':
                    break

            sentence = ' '.join(sampled_caption)

            sentences.append(sentence)

        return sentences


# building our training data
def train_batch(data, encoder, decoder, optimizer, criterion):
    encoder.train()

    decoder.train()

    images, captions, lengths = data

    images = images.to(device)

    captions = captions.to(device)

    targets = pack_padded_sequence(captions, lengths.cpu(), batch_first=True)[0]

    features = encoder(images)

    outputs = decoder(features, captions, lengths)

    loss = criterion(outputs, targets)

    decoder.zero_grad()

    encoder.zero_grad()

    loss.backward()

    optimizer.step()

    return loss


# This next set is somewhat standard, we are just validating the data
# This can be done by packing the data and evaluating loss
@torch.no_grad()
def validate_batch(data, encoder, decoder, criterion):
    encoder.eval()

    decoder.eval()

    images, captions, lengths = data

    images = images.to(device)

    captions = captions.to(device)

    targets = pack_padded_sequence(captions, lengths.cpu(), batch_first=True)[0]

    features = encoder(images)

    outputs = decoder(features, captions, lengths)

    loss = criterion(outputs, targets)

    return loss


# Using the previous functions as named
encoder = EncoderCNN(256).to(device)

decoder = DecoderRNN(256, 512, len(vocab.itos), 1).to(device)

criterion = nn.CrossEntropyLoss()

params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())

optimizer = torch.optim.AdamW(params, lr=1e-3)

n_epochs = 10

log = Report(n_epochs)

# the epochs are going to increase as they train!
# this can take a tremendous amount of time
for epoch in range(n_epochs):

    if epoch == 5: optimizer = torch.optim.AdamW(params, lr=1e-4)

    N = len(trn_dl)

    for i, data in enumerate(trn_dl):
        trn_loss = train_batch(data, encoder, decoder, optimizer, criterion)

        pos = epoch + (1 + i) / N

        log.record(pos=pos, trn_loss=trn_loss, end='\r')

    N = len(val_dl)

    for i, data in enumerate(val_dl):
        val_loss = validate_batch(data, encoder, decoder, criterion)

        pos = epoch + (1 + i) / N

        log.record(pos=pos, val_loss=val_loss, end='\r')
    log.report_avgs(epoch + 1)

log.plot_epochs(log=True)


# Building the generation of predictions via images
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')

    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        tfm_image = transform(image)[None]

    return image, tfm_image


def load_image_and_predict(image_path):
    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    org_image, tfm_image = load_image(image_path, transform)

    image_tensor = tfm_image.to(device)

    encoder.eval()

    decoder.eval()

    feature = encoder(image_tensor)

    sentence = decoder.predict(feature)[0]

    show(org_image, title=sentence)

    return sentence


files = Glob('val-images')

load_image_and_predict(choose(files))
#

# I mean it definetly works
# Most likely will not be accurate due to low epoch
# and/or the lower data sample size
