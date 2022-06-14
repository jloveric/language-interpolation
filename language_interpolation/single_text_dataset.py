import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Callable, Union, Any, Dict
import gutenbergpy.textget
from torch import Tensor
import logging
from multiprocessing import Pool
from functools import partial

logger = logging.getLogger(__name__)


def ascii_to_float(ascii_tensor: torch.Tensor):
    return (ascii_tensor - 64 + 0.5) / 64


def float_to_ascii(float_tensor: torch.Tensor):
    return ((float_tensor + 1.0) * 64 - 0.5).int()


def print_lines(text, size=100):
    short = text[:size].replace("\n", " ")
    logger.info(f"{short}")


def create_full_paths(root_dir: str, filenames: List[str] = None):
    """
    Construct global path from a list of local paths and a root directory.
    Args :
        root_dir : The directory containing the files
        filenames : A list of filenames within the root
    """
    if filenames is not None:
        full_paths = [f"{root_dir}/{path}" for path in filenames]
        return full_paths

    return None


def unify_ids(specific_ids: List[int], id_range: List[int]):
    """
    Create a single list from ids specified as a list and those
    specified as a range.
    Args :
        specific_ids : [1,2,10,20,100]
        range : [1,10] all values between 1 and 10 including 1 and 10
    """
    ids = []
    if specific_ids is not None:
        ids = specific_ids

    if id_range is not None:
        expand_ids = list(
            range(id_range[0], id_range[1] + 1)
        )  # User expects inclusive range
        ids.extend(expand_ids)

    return ids


def encode_input_from_text(text_in: str, features: int) -> Tuple[torch.tensor, str]:
    """
    Convert a string to input that the network can take.  Take the last "features" number
    of characters and convert to numbers.  Return those numbers as the network input, also
    return the raw_features (the text used to create the numbers).
    Args :
        text_in : input string.
        features : number of input features.
    Returns :
        tensor encoding, text used to create encoding.
    """
    text = text_in.encode("ascii", "ignore").decode("ascii")
    raw_sample = text[-(features):]
    encoding = [ord(val) for val in raw_sample]
    return torch.tensor(encoding), raw_sample


def decode_output_to_text(
    encoding: torch.tensor, topk: int = 1
) -> Tuple[torch.tensor, str]:
    """
    Takes an output from the network and converts to text.
    Args :
        encoding : Tensor of size 128 for each ascii character
        topk : The number of maximum values to report back
    Returns :
        Tuple of topk values and corresponding topk indices and list containing
        actual ascii values.
    """
    probabilities = torch.nn.Softmax(dim=0)(encoding)

    ascii_codes = torch.topk(probabilities, k=topk, dim=0)
    ascii_values = [
        chr(val).encode("ascii", "ignore").decode("ascii") for val in ascii_codes[1]
    ]

    return ascii_codes[0], ascii_codes[1], ascii_values


def generate_dataset(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset and convert to ordinal values for each character.
    This approach needs to be used for the neural network based approach.
    """
    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):
        n_feature = [ord(val) for val in text[i : (i + features)]]
        feature_list.append(n_feature)
        n_target = [ord(val) for val in text[(i + features) : (i + features + targets)]]
        target_list.append(n_target)

    return torch.tensor(feature_list), torch.tensor(target_list)


def dataset_centered(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a centered dataset as integers
    """

    f_left = features // 2
    f_right = features - f_left
    t_left = targets // 2
    t_right = targets - t_left

    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):
        feature_set = (
            text[i : (i + f_left)] + text[(f_left + 1) : (f_left + 1 + f_right)]
        )

        n_feature = [ord(val) for val in feature_set]
        feature_list.append(n_feature)
        n_target = [
            ord(val) for val in text[(i + f_left - t_left) : (i + f_left + t_right)]
        ]
        target_list.append(n_target)

    return torch.tensor(feature_list), torch.tensor(target_list)


def generate_dataset_char(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset as characters for use in random forest approaches.
    """
    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):

        n_feature = [ord(val) for val in text[i : (i + features)]]
        feature_list.append(n_feature)
        n_target = [ord(val) for val in text[(i + features) : (i + features + targets)]]
        target_list.append(n_target)

    return feature_list, target_list


def dataset_centered_char(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a centered dataset as char
    """

    f_left = features // 2
    f_right = features - f_left
    t_left = targets // 2
    t_right = targets - t_left

    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):
        feature_set = (
            text[i : (i + f_left)] + text[(f_left + 1) : (f_left + 1 + f_right)]
        )
        n_feature = [ord(val) for val in feature_set]
        feature_list.append(n_feature)
        n_target = [
            ord(val) for val in text[(i + f_left - t_left) : (i + f_left + t_right)]
        ]
        target_list.append(n_target)

    return feature_list, target_list


dataset_registry = {
    "sequence": generate_dataset,
    "centered": dataset_centered,
    "sequence_of_char": generate_dataset_char,
    "centered_char": dataset_centered_char,
}


def dataset_from_file(
    filename: str,
    features: int,
    targets: int,
    max_size: int = -1,
    dataset_generator=generate_dataset,
):
    with open(filename, "r") as f:
        return dataset_generator(
            text_in=f.read()[0:max_size], features=features, targets=targets
        )


def dataset_from_gutenberg(
    gutenberg_id: int,
    features: int,
    targets: int,
    max_size: int = -1,
    dataset_generator: Callable[[str, int, int], Tuple[Any, Any]] = generate_dataset,
) -> Union[Tuple[Tensor, Tensor], Any]:
    """
    Create a dataset from a book in project gutenberg https://www.gutenberg.org/
    Args :
        gutenberg_id : integer id of the book
        features : number of input features to use (number of characters)
        targets: number of targets to use (number of characters)
        datset_generator: formats the resulting dataset
    """
    raw_book = gutenbergpy.textget.get_text_by_id(gutenberg_id)
    clean_book = gutenbergpy.textget.strip_headers(raw_book)
    clean_book = clean_book.decode()

    return dataset_generator(
        text_in=clean_book[0:max_size], features=features, targets=targets
    )


class SingleTextDataset(Dataset):
    def __init__(
        self,
        filenames: List[str] = None,
        gutenberg_ids: List[int] = None,
        text: str = None,
        features: int = 10,
        targets: int = 1,
        max_size: int = -1,
        dataset_generator: Callable[
            [str, int, int], Tuple[Any, Any]
        ] = generate_dataset,
        num_workers: int = 0,
    ):
        """
        Args :
            filenames : List of filenames to load data from
            features : Number of input features (characters)
            targets : Number of output features (characters)
            max_size : Set the maximum number of characters to read from file.  Defaults
            to -1 which is to read everything.
            dataset_generator: A function that converts text into a tuple of features, targets
            num_workers: Number of parallel workers when more than one book is being
            processed.
        """
        if filenames is None and text is None and gutenberg_ids is None:
            raise ValueError(f"Must define either filenames, text or gutenberg ids.")
        if (filenames is not None) and (text is not None):
            raise ValueError(
                f"Either filenames, text, or gutenberg_ids must be defined."
            )

        list_features = []
        list_targets = []

        if filenames is not None:
            feature_list, target_list = dataset_from_file(
                filenames[0],
                features=features,
                targets=targets,
                max_size=max_size,
                dataset_generator=dataset_generator,
            )

            list_features.extend(feature_list)
            list_targets.extend(target_list)

        if text is not None:
            feature_list, target_list = dataset_generator(
                text_in=text, features=features, targets=targets
            )

            list_features.extend(feature_list)
            list_targets.extend(target_list)

        if gutenberg_ids is not None:

            if num_workers > 0:  # Run in parallel

                pdataset = partial(
                    dataset_from_gutenberg,
                    features=features,
                    targets=targets,
                    max_size=max_size,
                    dataset_generator=dataset_generator,
                )
                with Pool(num_workers) as p:
                    results = p.map(
                        pdataset,
                        gutenberg_ids,
                    )

                for feature_res, target_res in results:
                    list_features.extend(feature_res)
                    list_targets.extend(target_res)

            else:  # Run in serial
                for index in gutenberg_ids:

                    feature_list, target_list = dataset_from_gutenberg(
                        index,
                        features=features,
                        targets=targets,
                        max_size=max_size,
                        dataset_generator=dataset_generator,
                    )

                    list_features.extend(feature_list)
                    list_targets.extend(target_list)

        self.inputs = torch.stack(list_features)
        self.output = torch.stack(list_targets)
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.inputs[idx] - 64 + 0.5) / 64.0, self.output[idx]
