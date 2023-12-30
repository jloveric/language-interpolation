[![CI](https://github.com/jloveric/language-interpolation/actions/workflows/python-app.yml/badge.svg)](https://github.com/jloveric/language-interpolation/actions/workflows/python-app.yml)

# Natural Language Generation with Sparse High Order Layers
High order and piecewise networks for natural language generation (see [here](https://github.com/jloveric/high-order-layers-torch) for a description of High order layers being used).  The typical high order network design with piecewise polynomial
layers here is a fully connected network where each link has multiple segments.  Only one segment in
a link is active for each input so the network sparsity is determined by the number of segments. Although it looks like a standard MLP, the structure is more complicated and is a form of routing network with piecewise polynomials.

![image](images/language-interpolation-drawing.png)

I'm interested in creating larger language models from an ensemble of smaller models.  This would give better flexibility in adding or removing specific sources.

Working models for High Order MLPs, Mamba (SSM).

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
A few notes. For normalization, layer norm is the best, followed by maxabs and then no normalization . The only advantage
to maxabs is that there are no trainable parameters. The best optimizer is Lion, it seems by a long shot. I think for
these type of networks that have potentially steep gradients due to the polynomials, this is especially the case (since it uses the sign of the gradient). So far,
kaiming initialization seems to be performing better than linear initialization, but I need to investigate this further.

### sparse mlp
A few networks which are large enough to memorize "The Dunwich Horror" which is fairly short (120KB). Using Adam + learning rate scheduler.

#### Piecewise constant
Piecewise constant (requires discontinuous). Only the first layer can actually be optimized since derivatives beyond that are zero
```
python examples/high_order_interpolation.py data.type=sequence net=large_single_layer net.hidden.width=200 max_epochs=100 optimizer.lr=1e-3 batch_size=1000 net.layer_type=discontinuous net.n=1 net.segments=150 net.hidden.layers=3
```
#### Piecewise linear or better

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
#### Higher order version

3 hidden layers quadratic (n=3) 2 segments per link
```
python examples/high_order_interpolation.py data.type=sequence net=small net.hidden.width=250 max_epochs=100 net.n=3 net.hidden.layers=3 optimizer.lr=1e-5
```
3 hidden layers cubic (n=4) elements 2 segments per link
```
python examples/high_order_interpolation.py data.type=sequence net=small net.hidden.width=250 max_epochs=100 net.n=4 net.hidden.layers=3 optimizer.lr=1e-5
```
1 hidden layer with quartic (n=5) elements 2 segments per link
```
python examples/high_order_interpolation.py data.type=sequence net=large_single_layer net.hidden.width=200 max_epochs=100 optimizer.lr=1e-4 batch_size=1000 net.n=5
```
#### Only using high order for the input
Standard ReLU network, however, the input layer is piecewise linear so that it can bin the characters into each segment.  The rest of the network
look like a standard net.
```
python examples/high_order_interpolation.py data.type=sequence net=large_standard net.hidden.width=1000 max_epochs=100 optimizer.lr=1e-4
```
#### Without any hidden layers
can you memorize with just input and output layers with no hidden layers?  In this example we get to about 95% accuracy.  Discontinuous works
better than continuous as it has double the parameters for piecewise linear.  Increase the order of accuracy does nothing since the inputs are
discrete and not continuous - in this case we should have a piecewise constant option, but then the gradients would be 0.
```
python examples/high_order_interpolation.py data.type=sequence net=large_single_layer net.hidden.layers=0 max_epochs=100 optimizer.lr=1e-4 batch_size=1000 net.layer_type=discontinuous
```
### High order transformers
Using high order transformer blocks. These are in development and not as good as the MLPs above.
```
python examples/high_order_interpolation.py data.type=sequence net=transformer max_epochs=100 optimizer.lr=1e-3 batch_size=512 net.layer_type=continuous data.repeats=5 net.n=2 data.max_features=10 optimizer.patience=20 initialize.type=linear
```
Using only high order input, the rest being standard ReLU
```
python examples/high_order_interpolation.py data.type=sequence net=transformer max_epochs=100 optimizer.lr=1e-4 batch_size=16 net.layer_type=continuous data.repeats=1 net.n=3 data.max_features=20 optimizer.patience=20 initialize.type=linear accumulate_grad_batches=16 net.segments=2 net.model_type=high_order_input_transformer optimizer=lion
```
### sparse convolutional network
Using conv layers (not done too much here, see below for a possibly better network)
```
python examples/high_order_interpolation.py data.type=sequence net=conv max_epochs=100 optimizer.lr=1e-4 batch_size=1000 data.add_channel_dimension=true
```
### mamba

```
 python examples/high_order_interpolation.py data.type=sequence net=mamba optimizer.lr=1e-4 data.max_features=16 batch_size=1024
 ```

### tail focus network
Using tail focus network you can handle much much longer sequences, however the accuracy needs to be much higher to not get garbage (random ascii characters that don't look like any language) for a given input
```
python examples/high_order_interpolation.py data.type=sequence net=tail_focus max_epochs=100 optimizer.lr=1e-3 batch_size=8000 data.add_channel_dimension=true
```
to run a prediction
```
python examples/high_order_interpolation.py train=False checkpoint=\"outputs/2022-07-12/12-26-21/lightning_logs/version_0/checkpoints/'epoch=74-step=237075.ckpt'\" topk=1 num_predict=10000 prompts=["Who are you?"] data.add_channel_dimension=true net=tail_focus
```
and the result is jibberish sentences with a context of 256 characters and a single book and choosing the next best character.  I'll try this again at some point with a much larger context window and a much larger dataset.
```
Who are you?hhhIhr II hham hahe _I wa hhhar hit wohe _ower_ him hI whosow, hhough I wan og .hI. ht. __aday. Out hi sg them Cacler --a stabke wist his _Treag art unt.. The worn in the boumd di no ce. The hracises. Onte canheremhis counded teghing the hacling che conders. so collel, and as thing the eot to sheed an the wan or the lliused to the grom- of corning in. He who pout timetime, cu\' to e onled. The opar nor ly the notike.. The that ,uen\' forss will, liff us of that dert, the st bouthis spon the rills abec sire gors.  Then\'t alite alline the scomery dowped distured the _ anda stipy rouse pre. The comch bor, the tale hotives to frows in the cagane of cearsite to giss it a mameverise on the ping, as if withh doed crown onligriat ffmled afisht to crothin sing it aningstib catedep with tilled, and tather it nowms a sraned the eNid that hef on  follitines reas of in the lights he at the listent frog the and arenguy the consus dis, and it that himm--thoold y\'t heous. Ilikepad to was it ans on the tole of the shey. They mongite folker    sorece\'s abon the loud sote mathers verite to Corgass. Thele. Octereried to enttones of the vision inse frabity.
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
# Using lightgbm
Early experiments with lightgbm.

Training
```
python examples/language_interpolation_gbm.py 
```
Using
```
python examples/language_interpolation_gbm.py train=false checkpoint=outputs/2022-07-17/08-16-29/model.txt text="What did they say?"
```
sample output
```
prompt: What did they say? response: y.oon. These,ins onchased,by,sesionic subelaborishor, wondse trentel opowed my fould a midd, attrow 
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