import torch, torchtext
from torchtext.functional import to_tensor

xlmr_base = torchtext.models.XLMR_BASE_ENCODER
model = xlmr_base.get_model()
transform = xlmr_base.transform()
input_batch = ["Hello world", "How are you!"]
model_input = to_tensor(transform(input_batch), padding_value=1)
output = model(model_input)
print(output.shape)
