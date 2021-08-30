from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.metrics.functional import accuracy
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
import torch
from high_order_layers_torch.networks import *
from single_text_dataset import SingleTextDataset
from torchsummary import summary
from single_text_dataset import dataset_from_file, encode_input_from_text, decode_output_to_text, ascii_to_float, dataset_centered_char, dataset_centered
import random
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
from language_interpolation import Net


@hydra.main(config_path="./config", config_name="language_config")
def run_language_cellular_automata(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        early_stopping = EarlyStopping(monitor='train_loss', patience=5)
        trainer = Trainer(
            callbacks=[early_stopping],
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            gradient_clip_val=cfg.gradient_clip
        )

        model = Net(cfg)
        trainer.fit(model)
        print('testing')

        result = trainer.test(model)
        print('result', result)
        print('finished testing')
        print('best check_point', trainer.checkpoint_callback.best_model_path)
        print('loss', result[0]['train_loss'])
        return result[0]['train_loss']
    else:
        # plot some data
        print('evaluating result')
        print('cfg.checkpoint', cfg.checkpoint)
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print('checkpoint_path', checkpoint_path)
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()

        text_in = cfg.text
        features = cfg.mlp.input.width
        result_list = []

        text = (cfg.text).center(len(cfg.text)+cfg.mlp.features+3)
        for step in range(cfg.data.reapply) :
            #print('text is : ', text)
            text=text.center(len(text)+cfg.mlp.features+3)
            dataset = list(SingleTextDataset(text=text, features=features))
            dataset = torch.stack([a[0] for a in dataset])
            model.eval()
            outputs = model(dataset)

            result = ''
            for output in outputs:
                values, indices, ascii = decode_output_to_text(
                    encoding=output, topk=cfg.topk)

                # pick the next character weighted by probabilities of each character
                # prevents the same response for every query.
                actual = random.choices(ascii, values.tolist())
                result += actual[0]

            text = ''.join(result.replace('\n',' '))
            result_list.append(text)
            #print('output:', text)
        print('\nresult_list\n', '\n'.join(result_list))

if __name__ == "__main__":
    run_language_cellular_automata()
