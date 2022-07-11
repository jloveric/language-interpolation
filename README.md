[![CI](https://github.com/jloveric/language-interpolation/actions/workflows/python-app.yml/badge.svg)](https://github.com/jloveric/language-interpolation/actions/workflows/python-app.yml)

# Natural Language Generation with Sparse High Order Layers
High order and piecewise networks for natural language generation (see [here](https://github.com/jloveric/high-order-layers-torch) for a description of High order layers being used).  The typical high order network design with piecewise polynomial
layers here is a fully connected network where each link has multiple segments.  Only one segment in
a link is active for each input so the network sparsity is determined by the number of segments. Although it looks like a standard MLP, the structure is more complicated and is a form of routing network with piecewise polynomials.


![image](images/language-interpolation-drawing.png)

I'm interested in creating larger language models from an ensemble of smaller models.  This would give better flexibility in adding or removing specific sources.

# Dataset

Data from project Gutenberg are used, either single or multiple books.  Training is done on the character level.  A pytorch lightning data module for project Gutenberg has been implemented for data loading.

# Language interpolation of Books
Run single case (data appears in outputs)
```
python examples/high_order_interpolation.py data.type=sequence
```
with nevergrad (data appears in multirun)
```
python examples/high_order_interpolation.py -m data.type=sequence
```
# Decent parameters
A few networks which are large enough to memorize "The Dunwich Horror" which is fairly short (120KB). Using Adam + learning rate scheduler.

1 hidden layer 2 segments per link
```
python examples/high_order_interpolation.py data.type=sequence net=large_single_layer net.hidden.width=200 max_epochs=100 optimizer.lr=1e-4 batch_size=1000
```
2 hidden layers 2 segments per link
```
python examples/high_order_interpolation.py data.type=sequence net=large_double_layer max_epochs=100 net.hidden.width=250 optimizer.lr=1e-5
``` 
1 hidden layer 2 segments per link
```
python examples/high_order_interpolation.py data.type=sequence net=large_single_layer net.hidden.width=250 max_epochs=100 net.n=3 optimizer.lr=1e-4
```
3 layers quadratic 2 segments per link
```
python examples/high_order_interpolation.py data.type=sequence net=small net.hidden.width=250 max_epochs=100 net.n=3 net.hidden.layers=3 optimizer.lr=1e-5
```
Standard ReLU network, however, the input layer is piecewise linear so that it can bin the characters into each segment.  The rest of the network
look like a standard net.
```
python examples/high_order_interpolation.py data.type=sequence net=large_standard net.hidden.width=1000 max_epochs=100 optimizer.lr=1e-4
```
can you memorize with just input and output layers with no hidden layers?  In this example we get to about 95% accuracy.  Discontinuous works
better than continuous as it has double the parameters for piecewise linear.  Increase the order of accuracy does nothing since the inputs are
discrete and not continuous - in this case we should have a piecewise constant option, but then the gradients would be 0.
```
python examples/high_order_interpolation.py data.type=sequence net=large_single_layer net.hidden.layers=0 max_epochs=100 optimizer.lr=1e-4 batch_size=1000 net.layer_type=discontinuous
```
Using conv layers
```
python examples/high_order_interpolation.py data.type=sequence net=conv max_epochs=100 optimizer.lr=1e-4 batch_size=1000 data.add_channel_dimension=true
```
Using tail focus network
```
python examples/high_order_interpolation.py data.type=sequence net=tail_focus max_epochs=100 optimizer.lr=1e-4 batch_size=1000 data.add_channel_dimension=true
```
## Notes
I use input layer (continuous or discontinuous) with 128 segments, one for each ASCII character.  You can bump this down to 64, but the convergence doesn't seem quite as good - presumably it still works because most books don't use all the ascii characters anyway.

## Apply a model using sequence model
```
python examples/high_order_interpolation.py train=False checkpoint=\"outputs/2022-06-15/16-13-08/lightning_logs/version_0/checkpoints/epoch=32-step=104577.ckpt\" topk=2 num_predict=1000 prompts=["Who are you?"]
```
example output (model trained to predict the next character given the preceeding 16) using a single hidden layer. The prompt is not in the
dataset, however, the data eventually evolves into something that is in the dataset.
```
prompt: Who are you? 
result: Who are you? I the set it wall night, and the whippoorwills in the glen, Selina Frye tottered to the telephone and spread what news she could of the second phase of the horror.  The next day all the countryside. Trees, grass, and underbrush were whipped into a fury; and the frightened crowd at the mountain\'s base huddled still closer, and winced as if in expectation of a blow.  "_Ygnaiih ... ygnaiih ... thflthkh\'ngha ... Yog-Sothoth...._" They trailed off into nothingness as the whippoorwills in the glen, Selina Frye tottered to the telephone and spread what news she could of the second phase of the horror.  The next day all the countryside. Trees, grass, and underbrush were whipped into a fury; and the frightened crowd at the mountain\'s base huddled still closer, and winced as if in expectation of a blow.  "_Ygnaiih ... ygnaiih ... thflthkh\'ngha ... Yog-Sothoth...._" They trailed off into nothingness as the whippoorwills in the glen, Selina Frye tottered to the telephone and spread what news she
```



# Other Stuff
## Run with centered model (language cellular automaton)
```
python examples/language_cellular_automata.py net.features=11 data.type=centered
```

## As cellular automaton
A centered model can be repeatedly applied to the same text as a moving window and will work with arbitrary length sentences.  This
approach is similar to stencils used in solving partial differential equations.
```
python examples/language_cellular_automata.py net.features=11 data.type=centered train=False checkpoint=\"outputs/2021-08-21/20-14-40/lightning_logs/version_0/checkpoints/epoch=20-step=35909.ckpt\" topk=2 num_predict=200 text="The monster awakes" topk=1 data.reapply=10
```

# Interesting papers related to sparse mlps for language

[Efficient Language Modeling with Sparse all-MLP](https://arxiv.org/pdf/2203.06850.pdf)