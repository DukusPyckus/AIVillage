import logging
import os
import torch
import numpy as np
import psutil
import shutil
from typing import List, Dict, Union, Callable, Tuple
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import traceback
from tqdm import tqdm

from .mask_weights_utils import mask_input_with_mask_rate
from .task_vector import TaskVector

logger = logging.getLogger(__name__)

class EvoMergeException(Exception):
    """Custom exception class for EvoMerge errors."""
    pass

class ModelReference(BaseModel):
    name: str
    path: str

class MergeConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Dict[str, Union[float, List[float], Dict[str, float]]]
    custom_dir: str
    ps_techniques: List[str]
    dfs_techniques: List[str]

def check_system_resources(model_paths: List[str]) -> bool:
    total_model_size = 0
    for path in model_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                total_model_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
    
    if total_model_size > 0:
        free_disk_space = shutil.disk_usage(os.path.dirname(next(path for path in model_paths if os.path.exists(path)))).free
    else:
        free_disk_space = shutil.disk_usage(".").free
    
    available_ram = psutil.virtual_memory().available

    logger.info(f"Total model size: {total_model_size / (1024**3):.2f} GB")
    logger.info(f"Free disk space: {free_disk_space / (1024**3):.2f} GB")
    logger.info(f"Available RAM: {available_ram / (1024**3):.2f} GB")

    if total_model_size > free_disk_space:
        logger.error("Not enough disk space to store merged models!")
        return False
    if total_model_size > available_ram:
        logger.warning("Available RAM might not be sufficient to load all models simultaneously!")
        # We'll continue with a warning, but you might want to implement a more sophisticated
        # memory management strategy if this is a common issue
    return True

def setup_gpu_if_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU available. Using device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available. Using CPU.")
    return device

def clean_up_models(model_paths: List[str]):
    for path in model_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Removing model: {path}")
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to remove model {path}: {str(e)}")
            logger.warning("This is not a critical error, but you may want to manually remove the file or directory.")

def load_models(model_references: List[ModelReference]) -> Tuple[List[torch.nn.Module], List[AutoTokenizer]]:
    logger.info("Starting to load models and tokenizers")
    models = []
    tokenizers = []
    for model_ref in tqdm(model_references, desc="Loading models and tokenizers"):
        try:
            logger.info(f"Loading model and tokenizer: {model_ref.name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_ref.path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_ref.path)
            models.append(model)
            tokenizers.append(tokenizer)
            logger.info(f"Successfully loaded model and tokenizer: {model_ref.name}")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer {model_ref.name}: {str(e)}")
    return models, tokenizers

def save_model(model: torch.nn.Module, path: str) -> None:
    logger.info(f"Saving merged model to: {path}")
    try:
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        tokenizer.save_pretrained(path)
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise EvoMergeException(f"Error saving model: {str(e)}")

def generate_text(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str, max_length: int = 100) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise EvoMergeException(f"Error generating text: {str(e)}")

def evaluate_model(model_path: str) -> Dict[str, Union[float, str]]:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Implement your evaluation logic here
        # This is a placeholder implementation
        return {
            "perplexity": 100.0,
            "accuracy": 0.8,
            "overall_score": 0.9
        }

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise EvoMergeException(f"Error evaluating model: {str(e)}")

def parallel_evaluate_models(model_paths: List[str], max_workers: int = None) -> List[Dict[str, Union[float, str]]]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(evaluate_model, model_paths))

def mask_model_weights(finetuned_model: torch.nn.Module, pretrained_model: torch.nn.Module, exclude_param_names_regex: list, 
                       weight_format: str, weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str) -> Dict[str, torch.Tensor]:
    """
    Mask model weights based on the specified parameters.
    
    :param finetuned_model: The finetuned model
    :param pretrained_model: The pretrained model
    :param exclude_param_names_regex: List of regex patterns for parameter names to exclude
    :param weight_format: Format of weights to be masked ("finetuned_weight" or "delta_weight")
    :param weight_mask_rate: Rate of weights to mask
    :param use_weight_rescale: Whether to rescale weights after masking
    :param mask_strategy: Strategy for masking ("random" or "magnitude")
    :return: Dictionary of masked parameters
    """
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"Unsupported weight_format: {weight_format}"
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items(), desc="Masking weights"):
            masked_param_dict[param_name] = mask_input_with_mask_rate(
                input_tensor=param_value,
                mask_rate=weight_mask_rate,
                use_rescale=use_weight_rescale,
                mask_strategy=mask_strategy
            )

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict

def get_param_names_to_merge(input_param_names: List[str], exclude_param_names_regex: List[str]) -> List[str]:
    """
    Get the list of parameter names to merge, excluding those that match the given regex patterns.
    
    :param input_param_names: List of all parameter names
    :param exclude_param_names_regex: List of regex patterns for parameter names to exclude
    :return: List of parameter names to merge
    """
    import re
    param_names_to_merge = []
    for param_name in input_param_names:
        if not any(re.match(regex, param_name) for regex in exclude_param_names_regex):
            param_names_to_merge.append(param_name)
    return param_names_to_merge
