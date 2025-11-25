"""Text processing utilities."""

from typing import List, Union


def prepare_text_targets(answers: Union[List[str], str], tokenizer, max_length=512):
    """
    Properly tokenize answers for text generation loss.
    
    Args:
        answers: Answer string or list of answer strings
        tokenizer: Tokenizer to use (from transformers)
        max_length: Maximum sequence length
    
    Returns:
        Tokenized input IDs as tensor
    """
    # Ensure answers is a list
    if isinstance(answers, str):
        answers = [answers]
    
    # Tokenize answers with the same tokenizer used in LLaVA-Med
    tokenized = tokenizer(
        answers,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return tokenized.input_ids

