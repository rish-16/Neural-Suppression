# Neural-Suppression
Soucre code for Neural Suppression, a regularisation technique inspired by Dropout

### What it does
Suppression randomly picks neurons and scales them down by a constant `gamma`. The other unpicked neurons are allowed to pass through to the next layer without being affected.

### Usage
You can import the `Suppress2d` layer like so:

```python
from suppress2d import Suppress2d
```

Like `Dropout2d`, you can incorporate it into your code like so:

```python
from suppress2d import Suppress2d

x = torch.rand(100, 3, 32, 32)
layer = Suppress2d(gamma=0.2, p=0.5)
y = layer(x) # (100, 3, 32, 32)
```

Check out `main.py` for an example of a `ConvNet` that uses `Suppress2d` instead of `Dropout2d`.

### License
[MIT](https://github.com/rish-16/Neural-Suppression/blob/main/LICENSE)