"""Utility functions for dPrune package."""

import argparse
import subprocess
from typing import List, Optional
import pandas as pd
import os


# KenLM utility modified from: https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/languages_id.py

# Supported languages for KenLM models
SUPPORTED_LANGUAGES = [
    {
        "lang": "Arabic",
        "dataset_id": "ar",
        "kenlm_id": "ar",
    },
    {
        "lang": "Bengali", 
        "dataset_id": "bn",
        "kenlm_id": "bn",
    },
    {
        "lang": "Catalan",
        "dataset_id": "ca", 
        "kenlm_id": "ca",
    },
    {
        "lang": "English",
        "dataset_id": "en",
        "kenlm_id": "en",
    },
    {
        "lang": "Spanish",
        "dataset_id": "es",
        "kenlm_id": "es", 
    },
    {
        "lang": "Basque",
        "dataset_id": "eu",
        "kenlm_id": "eu",
    },
    {
        "lang": "French", 
        "dataset_id": "fr",
        "kenlm_id": "fr",
    },
    {
        "lang": "Hindi",
        "dataset_id": "hi",
        "kenlm_id": "hi",
    },
    {
        "lang": "Indonesian",
        "dataset_id": "id",
        "kenlm_id": "id",
    },
    {
        "lang": "Portuguese",
        "dataset_id": "pt", 
        "kenlm_id": "pt",
    },
    {
        "lang": "Urdu",
        "dataset_id": "ur",
        "kenlm_id": "ur",
    },
    {
        "lang": "Vietnamese",
        "dataset_id": "vi",
        "kenlm_id": "vi",
    },
    {
        "lang": "Chinese",
        "dataset_id": "zh",
        "kenlm_id": "zh",
    },
]


def get_supported_languages() -> List[str]:
    """Get list of supported language IDs for KenLM models.
    
    Returns:
        List of supported language IDs (e.g., ['en', 'fr', 'es', ...])
    """
    return [lang["kenlm_id"] for lang in SUPPORTED_LANGUAGES]


def download_kenlm_model(
    output_dir_path: str, 
    lang_id: str = "en", 
    source: str = "oscar",
    verbose: bool = True
) -> str:
    """Download KenLM model for a specific language.
    
    Args:
        output_dir_path: Directory to save the model
        lang_id: Language ID (e.g., 'en' for English)
        source: Source dataset ('oscar' or 'wikipedia')
        verbose: Whether to print progress messages
        
    Returns:
        Path to the downloaded model file
        
    Raises:
        ValueError: If language is not supported
        RuntimeError: If download fails
    """
    # Validate language
    supported_langs = get_supported_languages()
    if lang_id not in supported_langs:
        raise ValueError(
            f"Language '{lang_id}' is not supported. "
            f"Supported languages: {', '.join(supported_langs)}"
        )
    
    # Validate source
    if source not in ["oscar", "wikipedia"]:
        raise ValueError("Source must be 'oscar' or 'wikipedia'")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Construct model filename and path
    model_filename = f"{lang_id}.arpa.bin"
    model_path = os.path.join(output_dir_path, model_filename)
    
    # Skip download if model already exists
    if os.path.exists(model_path):
        if verbose:
            print(f"KenLM model for {lang_id} already exists at {model_path}")
        return model_path
    
    try:
        if verbose:
            print(f"Downloading KenLM model for language: {lang_id}")
            
        # Download the model
        url = f"https://huggingface.co/edugp/kenlm/resolve/main/{source}/{model_filename}"
        subprocess.check_output(
            f"wget {url} -P {output_dir_path}",
            shell=True,
            stderr=subprocess.STDOUT
        )
        
        if verbose:
            print(f"Successfully downloaded KenLM model for {lang_id} to {model_path}")
            
        return model_path
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Download failed for KenLM model for language {lang_id}. "
            f"Error: {e.output.decode() if e.output else str(e)}"
        )


def download_multiple_kenlm_models(
    output_dir_path: str,
    lang_ids: Optional[List[str]] = None,
    source: str = "oscar",
    verbose: bool = True
) -> List[str]:
    """Download KenLM models for multiple languages.
    
    Args:
        output_dir_path: Directory to save the models
        lang_ids: List of language IDs to download. If None, downloads all supported languages.
        source: Source dataset ('oscar' or 'wikipedia')
        verbose: Whether to print progress messages
        
    Returns:
        List of paths to downloaded model files
        
    Raises:
        ValueError: If any language is not supported
    """
    if lang_ids is None:
        lang_ids = get_supported_languages()
    
    model_paths = []
    failed_downloads = []
    
    for lang_id in lang_ids:
        try:
            model_path = download_kenlm_model(
                output_dir_path=output_dir_path,
                lang_id=lang_id,
                source=source,
                verbose=verbose
            )
            model_paths.append(model_path)
        except (ValueError, RuntimeError) as e:
            failed_downloads.append((lang_id, str(e)))
            if verbose:
                print(f"Failed to download {lang_id}: {e}")
    
    if failed_downloads and verbose:
        print(f"\nDownload summary:")
        print(f"  Successful: {len(model_paths)}")
        print(f"  Failed: {len(failed_downloads)}")
        
    return model_paths


def get_kenlm_model_path(output_dir_path: str, lang_id: str = "en") -> Optional[str]:
    """Get path to KenLM model file if it exists.
    
    Args:
        output_dir_path: Directory where models are stored
        lang_id: Language ID
        
    Returns:
        Path to model file if it exists, None otherwise
    """
    model_path = os.path.join(output_dir_path, f"{lang_id}.arpa.bin")
    return model_path if os.path.exists(model_path) else None 